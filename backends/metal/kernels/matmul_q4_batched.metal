// ============================================================
// matmul_q4_batched_kernel — BATCH M THẬT cho prefill (M=T token > 1)
// ------------------------------------------------------------
// BUG GỐC đã tìm ra bằng cách đọc matmul_q4_kernel_ndst2_paired (kernel
// decode M=1 đang dùng, KHÔNG đụng vào): dispatch grid Y = m (từng hàng
// M một threadgroup riêng), khiến MỖI hàng đọc lại TOÀN BỘ weight Q4 liên
// quan từ đầu, hoàn toàn độc lập với hàng khác - với M=92 (prefill), toàn
// bộ ma trận weight bị đọc lại 92 LẦN thay vì 1 lần. Đây là lý do prefill
// (944ms/token) gần bằng decode (1121ms/token) dù prefill lẽ ra phải nhanh
// hơn hàng chục lần (chi phí đọc weight được khấu hao qua nhiều token).
//
// KERNEL NÀY: đọc mỗi byte weight ĐÚNG 1 LẦN, dùng lại cho MBLOCK=4 hàng
// liên tiếp (dispatch grid Y = ceil(M/MBLOCK) thay vì Y=M) - giảm ~4 lần
// lưu lượng đọc bộ nhớ cho phần weight. MBLOCK=4 (không lớn hơn) để giữ áp
// lực thanh ghi ở mức an toàn - CHÍNH kernel decode M=1 (NDST=2) đã được
// chọn thay vì NDST=8 VÌ NDST=8 gây register spill trên Intel Iris 640
// (xem lịch sử comment ở matmul_q4_kernel_ndst2_paired) - kernel này thêm
// chiều MBLOCK nên cố định NDST=2 (thấp nhất) để tổng thanh ghi vẫn nằm
// trong vùng an toàn đã được CHÍNH kernel kia chứng minh không spill.
//
// TOÁN HỌC GIỐNG HỆT kernel gốc (sc[rr]*dot[rr] - zp[rr]*sc[rr]*sumA[mm]),
// chỉ khác: mảng dot/sum/sumA có thêm chiều mm (0..<MBLOCK), và byte weight
// wq[...] chỉ đọc 1 lần/k2 rồi TÁI SỬ DỤNG cho mọi mm trong vòng lặp con.
//
// CHỈ DÙNG CHO M>1 (prefill) - decode (M=1) vẫn dùng nguyên
// matmul_q4_kernel_ndst2_paired như cũ, KHÔNG đổi gì đường đó.
// ============================================================

#define BB_MATMUL_Q4_BATCHED(NDST_VAL, MBLOCK_VAL, KFUNC)                    \
kernel void KFUNC(                                                          \
    device const float* a          [[buffer(0)]],                          \
    device const uchar* wq         [[buffer(1)]],                          \
    device const float* scales     [[buffer(2)]],                          \
    device const float* zeros      [[buffer(3)]],                          \
    device float* c                [[buffer(4)]],                          \
    device const int& M            [[buffer(5)]],                          \
    device const int& K            [[buffer(6)]],                          \
    device const int& N            [[buffer(7)]],                          \
    device const int& groupSize    [[buffer(8)]],                          \
    device const int& nGroupsPerRow [[buffer(9)]],                         \
    device const int& simdWidth    [[buffer(10)]],                         \
    threadgroup float* partial [[threadgroup(0)]],                         \
    uint3 tgpig [[threadgroup_position_in_grid]],                          \
    uint3 tiisgv [[thread_position_in_threadgroup]]                        \
) {                                                                          \
    uint tiisg = tiisgv.x;                                                  \
    const int NDST = NDST_VAL;                                              \
    const int MBLOCK = MBLOCK_VAL;                                          \
    int m0 = int(tgpig.y) * MBLOCK;                                         \
    if (m0 >= M) return;                                                    \
                                                                              \
    int rowBlock = int(tgpig.x);                                            \
    int n0 = rowBlock * NDST;                                               \
    int bytesPerRow = (K + 1) / 2;                                          \
                                                                              \
    float sum[MBLOCK_VAL][NDST_VAL];                                        \
    _Pragma("unroll")                                                       \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) sum[mm][rr] = 0.0f;          \
    }                                                                        \
                                                                              \
    int prevGr = -1;                                                        \
    float sc[NDST_VAL];                                                     \
    float zp[NDST_VAL];                                                     \
    float dot[MBLOCK_VAL][NDST_VAL];                                        \
    _Pragma("unroll")                                                       \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) dot[mm][rr] = 0.0f;          \
    }                                                                        \
    float sumA[MBLOCK_VAL];                                                 \
    _Pragma("unroll")                                                       \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) sumA[mm] = 0.0f;                \
                                                                              \
    for (int k2 = int(tiisg) * 2; k2 < K; k2 += int(simdWidth) * 2) {       \
        bool hasK1 = (k2 + 1) < K;                                          \
        int gr = groupSize > 0 ? (k2 / groupSize) : 0;                      \
                                                                              \
        if (gr != prevGr) {                                                 \
            if (prevGr != -1) {                                             \
                _Pragma("unroll")                                           \
                for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                  \
                    _Pragma("unroll")                                       \
                    for (int rr = 0; rr < NDST_VAL; ++rr) {                \
                        sum[mm][rr] += sc[rr] * dot[mm][rr] - zp[rr] * sc[rr] * sumA[mm]; \
                        dot[mm][rr] = 0.0f;                                  \
                    }                                                        \
                }                                                            \
                _Pragma("unroll")                                           \
                for (int mm = 0; mm < MBLOCK_VAL; ++mm) sumA[mm] = 0.0f;    \
            }                                                               \
            prevGr = gr;                                                    \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                int n = n0 + rr;                                            \
                if (n < N) {                                                \
                    sc[rr] = scales[n * nGroupsPerRow + gr];                \
                    zp[rr] = zeros[n * nGroupsPerRow + gr];                 \
                } else {                                                    \
                    sc[rr] = 0.0f;                                          \
                    zp[rr] = 0.0f;                                          \
                }                                                           \
            }                                                               \
        }                                                                   \
                                                                              \
        int byteIdx = k2 >> 1;                                              \
        /* MẤU CHỐT: đọc byte weight ĐÚNG 1 LẦN/k2, TÁI SỬ DỤNG cho toàn bộ \
           MBLOCK hàng bên dưới - đây là điểm khác biệt DUY NHẤT về mặt    \
           lưu lượng bộ nhớ so với kernel decode gốc (nơi đọc lại byte này \
           riêng cho MỖI hàng vì mỗi hàng là 1 threadgroup độc lập).       \
           THÊM MỚI (sparsity): nếu sc[rr]==0 (khối này đã bị Nim đánh dấu \
           cat luc uploadResident - xem nimformer.nim), SKIP HẲN việc      \
           đọc byte weight cho cột rr đó - tiết kiệm băng thông đọc THẬT,  \
           không chỉ tiết kiệm phép tính (dù không skip, kết quả toán học  \
           vẫn = 0 vì nhân với sc=0, nhưng SKIP mới thực sự đỡ phải đọc    \
           byte đó từ bộ nhớ GPU - đây là nguồn lợi tốc độ thật của        \
           sparsity, khong chi tinh ra 0 nhanh hon). */                    \
        uchar wbyte[NDST_VAL];                                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) {                             \
            int n = n0 + rr;                                                \
            wbyte[rr] = (n < N && sc[rr] != 0.0f) ? wq[n * bytesPerRow + byteIdx] : uchar(0); \
        }                                                                    \
                                                                              \
        _Pragma("unroll")                                                   \
        for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                          \
            int m = m0 + mm;                                                \
            if (m >= M) continue;                                          \
            float aVal0 = a[m * K + k2];                                    \
            float aVal1 = hasK1 ? a[m * K + k2 + 1] : 0.0f;                 \
            sumA[mm] += aVal0 + (hasK1 ? aVal1 : 0.0f);                     \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                int n = n0 + rr;                                            \
                if (n >= N) continue;                                      \
                uchar nib0 = wbyte[rr] & 0x0Fu;                             \
                uchar nib1 = (wbyte[rr] >> 4) & 0x0Fu;                      \
                dot[mm][rr] += aVal0 * float(nib0);                         \
                if (hasK1) dot[mm][rr] += aVal1 * float(nib1);              \
            }                                                                \
        }                                                                    \
    }                                                                        \
    if (prevGr != -1) {                                                     \
        _Pragma("unroll")                                                   \
        for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                          \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                sum[mm][rr] += sc[rr] * dot[mm][rr] - zp[rr] * sc[rr] * sumA[mm]; \
            }                                                                \
        }                                                                    \
    }                                                                         \
                                                                              \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) \
        for (int rr = 0; rr < NDST_VAL; ++rr) \
            partial[(mm * NDST_VAL + rr) * simdWidth + tiisg] = sum[mm][rr]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint stride = uint(simdWidth) / 2; stride > 0; stride /= 2) { \
        if (tiisg < stride) { \
            for (int i = 0; i < MBLOCK_VAL * NDST_VAL; ++i) \
                partial[i * simdWidth + tiisg] += partial[i * simdWidth + tiisg + stride]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (tiisg == 0) { \
        for (int mm = 0; mm < MBLOCK_VAL; ++mm) \
            for (int rr = 0; rr < NDST_VAL; ++rr) \
                if (m0 + mm < M && n0 + rr < N) \
                    c[(m0 + mm) * N + n0 + rr] = partial[(mm * NDST_VAL + rr) * simdWidth]; \
    } \
}

// MBLOCK=4, NDST=2 - tổng thanh ghi thêm (dot[4][2]+sum[4][2]+sumA[4]=20)
// so với bản gốc NDST=2 (dot[2]+sum[2]+sumA=5) - tăng vừa phải, KHÔNG áp
// dụng cấu hình NDST cao hơn (đã biết gây spill ở kernel decode).
BB_MATMUL_Q4_BATCHED(2, 4, matmul_q4_batched_m4_kernel)
