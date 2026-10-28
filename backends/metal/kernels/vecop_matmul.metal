#include <metal_stdlib>
using namespace metal;

// ============================================================
// VECTOR OPERATIONS
// ============================================================
kernel void vecop_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const int& op [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (op == 0) c[id] = a[id] + b[id];
    else if (op == 1) c[id] = a[id] - b[id];
    else if (op == 2) c[id] = a[id] * b[id];
    else if (op == 3) c[id] = a[id] / b[id];
}

// ============================================================
// COPY INTO (GPU-side concat) - THÊM MỚI: dst[dstOffset + id] = src[id],
// dùng để ghép nhiều buffer con (vd outHeads của từng KV-head trong
// attention decode) thành 1 buffer liên tục [H*D] NGAY TRÊN GPU, tránh
// phải sessionRead() về CPU rồi sessionUpload() lại chỉ để nối mảng -
// đây là điểm chặn cuối cùng khiến session attention-decode và session
// oProj không gộp được làm 1. dstOffset tính bằng SỐ PHẦN TỬ (không phải
// byte) - metal_copy_into_enc (metal_shim.m) tự nhân sizeof(float) khi
// bind buffer offset.
kernel void copy_into_kernel(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    dst[id] = src[id];
}

// ============================================================
// SCALE (x *= scalar, in-place) - THÊM MỚI: dùng để nhân "scale" của
// attention (1/sqrt(D)) vào TOÀN BỘ buffer Q [H*D] MỘT LẦN trên GPU, thay
// vì đọc Q về CPU rồi nhân từng phần tử trong vòng for (xem qGroups ở
// nim_inference.nim). Sau khi có scale_kernel, việc "gather theo group"
// cho Q không cần kernel scatter riêng: vì layout Q là [H,D] với
// h = hkv*nRep + r liên tục theo hkv, mỗi group hkv chỉ là 1 LÁT CẮT LIÊN
// TỤC dài nRep*D bắt đầu tại offset hkv*nRep*D - dùng thẳng
// copy_into_kernel (sessionCopyInto, đã có sẵn, tham số srcOffsetElems)
// để trích ra, không cần kernel gather/scatter mới.
kernel void scale_kernel(
    device float* x               [[buffer(0)]],
    device const float& scalar    [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    x[id] *= scalar;
}

// ============================================================
// MATMUL TILED
// ============================================================
// SỬA (thay cho fix "hạ cứng BB_TILE=8 cho MỌI GPU" trước đây):
// Root cause thật sự KHÔNG phải "tile 32 luôn nguy hiểm" mà là
// metal_matmul()/metal_matmul_enc() dispatch groupSize=32x32=1024 threads
// mà KHÔNG kiểm tra pipeline.maxTotalThreadsPerThreadgroup của GPU đang
// chạy trước (khác metal_vecop cùng file đã làm đúng bằng MIN(...)). Hạ
// tile về 8 cho MỌI GPU để né bug đó là đúng an toàn nhưng sai hiệu năng:
// trên Apple Silicon / GPU rời mạnh, threadgroup 1024 threads hoàn toàn
// hợp lệ và tile 32 nhanh hơn tile 8 nhiều lần (data reuse tốt hơn hẳn:
// mỗi phần tử tileA/tileB được dùng lại 32 lần thay vì 8 lần trước khi
// nạp lại từ device memory).
// Fix ĐÚNG: build sẵn 3 biến thể kernel tile 8/16/32 (macro sinh code,
// KHÔNG dùng function constant vì threadgroup array size cần literal rõ
// ràng lúc build MSL, tránh rủi ro không tương thích driver cũ). Phía
// metal_shim.m sẽ tự đo pipeline.maxTotalThreadsPerThreadgroup của TỪNG
// pipeline sau khi tạo (không đoán trước theo tên GPU) rồi chọn variant
// lớn nhất mà threadgroup thật sự cho phép, fallback dần xuống nếu không.
#define BB_MATMUL_VARIANT(TILE, KFUNC)                                       \
kernel void KFUNC(                                                           \
    device const float* a [[buffer(0)]],                                    \
    device const float* b [[buffer(1)]],                                    \
    device float* c [[buffer(2)]],                                          \
    device const int& m [[buffer(3)]],                                      \
    device const int& k [[buffer(4)]],                                      \
    device const int& n [[buffer(5)]],                                      \
    uint2 gid [[thread_position_in_grid]],                                  \
    uint2 tid [[thread_position_in_threadgroup]]                            \
) {                                                                          \
    threadgroup float tileA[TILE][TILE];                                    \
    threadgroup float tileB[TILE][TILE];                                    \
    int row = int(gid.y);                                                   \
    int col = int(gid.x);                                                   \
    int tx = int(tid.x), ty = int(tid.y);                                   \
    float sum = 0.0f;                                                       \
    int numTiles = (k + TILE - 1) / TILE;                                   \
    for (int t = 0; t < numTiles; ++t) {                                    \
        int aCol = t * TILE + tx;                                           \
        int bRow = t * TILE + ty;                                           \
        tileA[ty][tx] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;    \
        tileB[ty][tx] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        for (int p = 0; p < TILE; ++p) {                                    \
            sum += tileA[ty][p] * tileB[p][tx];                             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                        \
    if (row < m && col < n) {                                               \
        c[row * n + col] = sum;                                             \
    }                                                                        \
}

// 3 biến thể - metal_shim.m chọn 1 trong 3 tuỳ pipeline.maxTotalThreadsPerThreadgroup
// thật sự của GPU đang chạy (8x8=64, 16x16=256, 32x32=1024 threads/threadgroup).
BB_MATMUL_VARIANT(8,  matmul_kernel_t8)
BB_MATMUL_VARIANT(16, matmul_kernel_t16)
BB_MATMUL_VARIANT(32, matmul_kernel_t32)

// Giữ nguyên tên "matmul_kernel" = alias tile 8 để mọi chỗ gọi cũ (nếu còn
// sót nơi nào không qua đường chọn-tile-động) vẫn compile được, KHÔNG BAO
// GIỜ dispatch với groupSize > 8x8 cho tên này.
BB_MATMUL_VARIANT(8, matmul_kernel)

// ============================================================
// MATMUL TILED "NT" (C = A @ B^T, B lưu row-major [N,K] - KHÔNG transpose
// vật lý). THÊM MỚI: dùng cho attention decode - K-cache/V-cache lưu tự
// nhiên theo layout [S,D] (N=S, K=headDim) - loại bỏ hoàn toàn bước
// transpose CPU O(S*D)/token trước đây phải làm để build kBlockT.
// ============================================================
#define BB_MATMUL_NT_VARIANT(TILE, KFUNC)                                    \
kernel void KFUNC(                                                           \
    device const float* a [[buffer(0)]],                                    \
    device const float* b [[buffer(1)]],                                    \
    device float* c [[buffer(2)]],                                          \
    device const int& m [[buffer(3)]],                                      \
    device const int& k [[buffer(4)]],                                      \
    device const int& n [[buffer(5)]],                                      \
    uint2 gid [[thread_position_in_grid]],                                  \
    uint2 tid [[thread_position_in_threadgroup]]                            \
) {                                                                          \
    threadgroup float tileA[TILE][TILE];                                    \
    threadgroup float tileB[TILE][TILE];                                    \
    int tx = int(tid.x), ty = int(tid.y);                                   \
    int rowBase = int(gid.y) - ty;                                          \
    int colBase = int(gid.x) - tx;                                          \
    int row = rowBase + ty;                                                 \
    int col = colBase + tx;                                                 \
    float sum = 0.0f;                                                       \
    int numTiles = (k + TILE - 1) / TILE;                                   \
    for (int t = 0; t < numTiles; ++t) {                                    \
        int aK = t * TILE + tx;                                             \
        int bCol = colBase + ty; /* doc lap voi 'col' - dung ty lam cot khi load B */ \
        int bK = t * TILE + tx;                                             \
        tileA[ty][tx] = (row < m && aK < k) ? a[row * k + aK] : 0.0f;        \
        tileB[ty][tx] = (bCol < n && bK < k) ? b[bCol * k + bK] : 0.0f;      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        for (int p = 0; p < TILE; ++p) {                                    \
            sum += tileA[ty][p] * tileB[tx][p];                             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                        \
    if (row < m && col < n) {                                               \
        c[row * n + col] = sum;                                             \
    }                                                                        \
}

BB_MATMUL_NT_VARIANT(8, matmul_nt_kernel)

// ============================================================
// MATMUL2 - DUAL MATMUL
// ============================================================
kernel void matmul2_kernel(
    device const float* a1 [[buffer(0)]],
    device const float* b1 [[buffer(1)]],
    device float* c1 [[buffer(2)]],
    device const int32_t* dims1 [[buffer(3)]],
    device const float* a2 [[buffer(4)]],
    device const float* b2 [[buffer(5)]],
    device float* c2 [[buffer(6)]],
    device const int32_t* dims2 [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]
) {
    int m1 = dims1[0], k1 = dims1[1], n1 = dims1[2];
    int m2 = dims2[0], k2 = dims2[1], n2 = dims2[2];
    
    int row = position.y;
    int col = position.x;
    
    if (row < m1 && col < n1) {
        float sum = 0.0f;
        for (int i = 0; i < k1; ++i) {
            sum += a1[row * k1 + i] * b1[i * n1 + col];
        }
        c1[row * n1 + col] = sum;
    }
    
    if (row < m2 && col < n2) {
        float sum = 0.0f;
        for (int i = 0; i < k2; ++i) {
            sum += a2[row * k2 + i] * b2[i * n2 + col];
        }
        c2[row * n2 + col] = sum;
    }
}

// ============================================================
// ACTIVATIONS
// ============================================================
kernel void activation_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int& op [[buffer(2)]],
    device const float& p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float val = x[id];
    if (op == 0) {
        y[id] = val > 0.0f ? val : 0.0f;
    } else if (op == 1) {
        y[id] = 1.0f / (1.0f + exp(-val));
    } else if (op == 2) {
        y[id] = tanh(val);
    } else if (op == 3) {
        float alpha = 0.1f;
        float beta = 1.0f;
        y[id] = val > 0.0f ? val * (1.0f + alpha * val) : beta * val * exp(val);
    } else if (op == 4) {
        y[id] = exp(val);
    } else if (op == 5) {
        y[id] = log(val);
    } else if (op == 6) {
        y[id] = sqrt(val);
    } else if (op == 7) {
        y[id] = pow(val, p);
    }
}

// ============================================================
// ACTIVATION BACKWARD
// ============================================================
kernel void activation_backward_kernel(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    device const int& op [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float val = x[id];
    float d = dy[id];
    if (op == 0) {
        dx[id] = val > 0.0f ? d : 0.0f;
    } else if (op == 1) {
        float sig = 1.0f / (1.0f + exp(-val));
        dx[id] = d * sig * (1.0f - sig);
    } else if (op == 2) {
        float t = tanh(val);
        dx[id] = d * (1.0f - t * t);
    } else if (op == 3) {
        float alpha = 0.1f;
        float beta = 1.0f;
        dx[id] = val > 0.0f ? d * (1.0f + 2.0f * alpha * val) : d * beta * exp(val) * (1.0f + val);
    }
}

// ============================================================
// ROPE (rotate_half, in-place) - THÊM MỚI
// ============================================================
// Trước đây RoPE phải: đọc Q/K về CPU, tự build mảng cosFull/sinFull/rotHalf
// kích thước [T,H,D] (lãng phí, lặp giá trị cos/sin H lần), upload lại, làm
// 3 sessionVecOp (mul,mul,add), rồi lại đọc về CPU lần nữa để dùng cho
// attention - tốn 1 session GPU + 2 lượt round-trip CPU RIÊNG chỉ cho RoPE.
// Kernel này áp dụng rotate_half TRỰC TIẾP trong lúc Q/K còn nằm trên GPU
// (ngay sau QKV matmul, TRONG CÙNG session), không cần build mảng lặp lại,
// không cần đọc CPU giữa chừng. cosT/sinT là đoạn [T, D/2] đã cắt sẵn theo
// posOffset ở phía CPU (bảng gốc rope.cosT/sinT không đổi giữa các
// layer/token nên phần cắt cũng rất nhỏ so với Q/K).
kernel void rope_kernel(
    device float* x               [[buffer(0)]],   // [T, Hn, D] in-place
    device const float* cosT      [[buffer(1)]],   // [T, D/2]
    device const float* sinT      [[buffer(2)]],   // [T, D/2]
    device const int& Hn          [[buffer(3)]],
    device const int& D           [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]          // gid.x = i in [0, D/2), gid.y = h in [0, Hn), gid.z = t in [0, T)
) {
    int half_ = D / 2;
    int i = int(gid.x);
    if (i >= half_) return;
    int h = int(gid.y);
    int t = int(gid.z);
    int base = (t * Hn + h) * D;
    float c = cosT[t * half_ + i];
    float s = sinT[t * half_ + i];
    float x1 = x[base + i];
    float x2 = x[base + i + half_];
    x[base + i]         = x1 * c - x2 * s;
    x[base + i + half_] = x2 * c + x1 * s;
}

// ============================================================
// SOFTMAX
// ============================================================
#define BB_WG 256

// SUA (fix bug tinh toan sai nhat quan - xem bao cao debug): CA 3 kernel
// duoi day (softmax/rmsnorm/layernorm) gia dinh CO DINH BB_WG=256 thread/
// threadgroup cho ca kich thuoc mang reduction "red[]" LAN buoc nhay vong
// lap - nhung phia dispatch (metal_shim.m) lai tinh threadsPerGroup =
// MIN(pipeline.maxTotalThreadsPerThreadgroup, 256) O RUNTIME. Neu GPU/
// pipeline nao co maxTotalThreadsPerThreadgroup < 256 (hoan toan co the
// tren GPU yeu/cu nhu Iris 640 - da thay nhieu dau hieu bat thuong tren
// GPU nay suot qua trinh debug), so thread THAT dispatch < 256, nghia la
// cac o red[thread That..255] KHONG BAO GIO duoc thread nao ghi vao - la
// threadgroup memory CHUA KHOI TAO. Buoc reduction van co dinh bat dau tu
// BB_WG/2=128 nen se CONG NHAM cac o rac nay vao tong (sum-of-squares cho
// rmsnorm, max/sum cho softmax) -> ket qua sai NHAT QUAN (deterministic,
// vi rac trong threadgroup memory thuong on dinh giua cac lan goi tren
// cung 1 tien trinh) cho MOI layer, MOI token dung kernel nay. Fix: dung
// [[threads_per_threadgroup]] (Metal cap san, biet DUNG so thread that su
// dang chay) de CHINH cac thread dang co chu dong zero not phan red[] con
// lai (tu tgSize den BB_WG-1) TRUOC buoc reduction, dam bao KHONG con o
// nao chua khoi tao du threadgroup dispatch nho hon BB_WG bao nhieu.

kernel void softmax_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float m = -INFINITY;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        m = max(m, x[off + c]);
    }
    red[tid] = m;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        float e = exp(x[off + c] - maxVal);
        y[off + c] = e;
        sum += e;
    }
    red[tid] = sum;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invSum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        y[off + c] *= invSum;
    }
}

// ============================================================
// RMSNORM FORWARD (mirror layernorm_kernel ở trên, bỏ mean-subtract + beta -
// RMSNorm chỉ scale theo sqrt(mean(x^2)), không center theo mean, không có
// bias). Cơ chế threadgroup-reduction giữ NGUYÊN VẸN từ layernorm_kernel đã
// chạy thật (cùng BB_WG, cùng pattern tid += BB_WG, cùng thứ tự barrier) -
// phần duy nhất khác là công thức elementwise cuối và bớt 1 buffer (beta).
// ============================================================
kernel void rmsnorm_kernel(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int& cols [[buffer(3)]],
    device const float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float sumSq = 0.0f;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        float v = x[off + c];
        sumSq += v * v;
    }
    red[tid] = sumSq;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float meanSq = red[0] / float(cols);
    float invStd = 1.0f / sqrt(meanSq + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        y[off + c] = x[off + c] * invStd * gamma[c];
    }
}

// ============================================================
// LAYERNORM FORWARD
// ============================================================
kernel void layernorm_kernel(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const int& cols [[buffer(4)]],
    device const float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float sum = 0.0f, sumSq = 0.0f;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        float v = x[off + c];
        sum += v;
        sumSq += v * v;
    }
    red[tid] = sum;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[tid] = sumSq;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float meanSq = red[0] / float(cols);
    float var = meanSq - mean * mean;
    float invStd = 1.0f / sqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: buoc nhay = SO THREAD THAT (tgSize), khong phai BB_WG
        y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
}

// ============================================================
// LAYERNORM BACKWARD
// ============================================================
kernel void layernorm_backward_kernel(
    device const float* dy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* dx [[buffer(4)]],
    device float* dgamma [[buffer(5)]],
    device float* dbeta [[buffer(6)]],
    device const int& rows [[buffer(7)]],
    device const int& cols [[buffer(8)]],
    device const float& eps [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < (uint)cols) {
        float dg = 0.0f;
        float db = 0.0f;
        for (int r = 0; r < rows; ++r) {
            uint off = r * cols;
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += x[off + c];
            mean /= float(cols);
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float diff = x[off + c] - mean;
                var += diff * diff;
            }
            var /= float(cols);
            float invStd = 1.0f / sqrt(var + eps);
            
            float norm = (x[off + id] - mean) * invStd;
            dg += dy[off + id] * norm;
            db += dy[off + id];
        }
        dgamma[id] = dg;
        dbeta[id] = db;
    }
    
    uint row = id;
    if (row < (uint)rows) {
        uint off = row * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += x[off + c];
        mean /= float(cols);
        
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = x[off + c] - mean;
            var += diff * diff;
        }
        var /= float(cols);
        float invStd = 1.0f / sqrt(var + eps);
        
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float grad = dy[off + c] * gamma[c] * invStd;
            sum1 += grad;
            sum2 += grad * (x[off + c] - mean);
        }
        sum2 *= (-invStd * invStd / float(cols));
        
        for (int c = 0; c < cols; ++c) {
            float term1 = dy[off + c] * gamma[c] * invStd;
            float term2 = sum1 / float(cols);
            // sum2 = -invStd^3 * mean(dy*gamma*(x-mean)) (da chia cols o tren): KHONG nhan them 2/cols
            // (ban cu lam so hang hieu chinh phuong sai nho di cols/2 lan -> dx sai).
            float term3 = (x[off + c] - mean) * sum2;
            dx[off + c] = term1 - term2 + term3;
        }
    }
}

// ============================================================
// EMBEDDING LOOKUP
// ============================================================
kernel void embedding_lookup_kernel(
    device const float* table [[buffer(0)]],
    device const int32_t* indices [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int& dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    int32_t idx = indices[id];
    uint dst_off = id * dim;
    uint src_off = idx * dim;
    for (int d = 0; d < dim; ++d) {
        y[dst_off + d] = table[src_off + d];
    }
}

// ============================================================
// MATMUL TRUC TIEP TREN INT4 ASYMMETRIC (packed, per-group scale/zero_point)
// ============================================================
// SUA (item 3 trong ghi chu hieu nang): TRUOC DAY moi lan Linear.forward()
// phai dequantizeTensorTransposed() TOAN BO weight int4 ra 1 mang fp32 day
// du tren CPU (vd 4096x11008 ~180MB fp32) roi UPLOAD ca mang fp32 do len
// GPU chi de nhan 1 lan roi vut - dung y nhu llama.cpp KHONG lam (ho nhan
// truc tiep tren du lieu quantized). Kernel nay nhan thang tren buffer
// int4 da pack (2 gia tri/byte) + scale/zero_point theo group, KHONG can
// dequant rieng tren CPU va KHONG can upload mang fp32 day du - giam bang
// thong upload ~4-8 lan (dung tinh than "giam bang thong memory" cua
// llama.cpp), giai phong luon phan CPU-time danh cho vong for dequant.
// Layout phai khop CHINH XAC voi quant.nim (quantizeInt4Asymmetric /
// dequantizeTensorTransposed):
//   - wq: [N, K] row-major, nibble-packed 2 gia tri/byte, byte thap = phan
//     tu chan (k%2==0), byte cao = phan tu le (k%2==1).
//   - scales/zeros: neu groupSize>0, moi hang N chia thanh
//     ceil(K/groupSize) group, group[g] dung scales[n*nGroupsPerRow+g].
//   - c[m,n] = sum_k a[m,k] * ((nibble(n,k) - zero(n,g)) * scale(n,g))
//     -- dung 100% cong thuc trong dequantizeTensorTransposed() + beMatmul
//     hien tai, chi khac la khong vat chat hoa wT ra fp32 truoc.
// CHUA CHAY THU TREN GPU METAL THAT (moi truong sinh code la Linux, khong
// co macOS/Metal that de bien dich+chay) - viet bam sat dung cong thuc
// tham chieu CPU (quant.nim) nhung BAT BUOC phai tu build+test tren May
// that truoc khi dung cho sinh token that. Neu ra token rac, day la nghi
// ngo dau tien (sai offset nibble hoac sai thu tu group).
// SUA (theo yêu cầu "sửa cách đi như llama.cpp" - kernel CŨ bên dưới đã đổi
// tên còn matmul_q4_kernel_naive để tham khảo/rollback):
// Kernel CŨ: 1 thread = 1 phần tử output, tự đọc & cộng dồn HẾT K phần tử
// tuần tự (không SIMD reduction), dispatch threadgroup 16x1 = chỉ 16 thread/
// group -> occupancy cực thấp trên GPU thật (Iris có hàng trăm ALU rảnh).
// Kernel MỚI theo đúng chiến thuật kernel_mul_mv_q4_K của llama.cpp/ggml
// Metal backend:
//   1. Mỗi threadgroup = ĐÚNG 1 simdgroup (32 thread), xử lý MMQ4_NDST=4
//      hàng output (n) cùng lúc cho 1 hàng activation (m) - dispatch theo
//      (rowBlock = ceil(N/4), M).
//   2. Activation a[m,:] (K phần tử) được 32 thread NẠP 1 LẦN vào threadgroup
//      shared memory rồi DÙNG LẠI cho cả 4 hàng - trước đây mỗi hàng n tự
//      đọc lại a[] từ device memory (băng thông lãng phí gấp N/4 lần).
//   3. Mỗi hàng: 32 thread chia nhau K phần tử (k = lane, lane+32, ...),
//      cộng dồn partial rồi simd_sum() reduce chỉ trong 1 lệnh phần cứng
//      (thay vì 1 thread loop tuần tự K lần) - đây là phần tăng tốc chính.
// Kết quả: số threadgroup dispatch ra nhiều hơn N/4 lần (đủ để lấp đầy toàn
// bộ EU của GPU dù yếu), mỗi threadgroup làm việc song song thật (32-way)
// thay vì 1 thread gánh hết K.
// SUA BUG NGHIÊM TRỌNG (gây "GPU Timeout Error" thật trên máy Intel Iris):
// Bản trước dùng threadgroup shared memory kích thước K*4 byte để cache cả
// hàng activation - với K = intermediate_size (vd down_proj, K~11008+) thì
// cần ~44KB, VƯỢT giới hạn shared-memory/threadgroup của Iris (thường chỉ
// 16-32KB tuỳ gen). Metal không báo lỗi rõ ràng lúc setThreadgroupMemoryLength
// trên GPU cũ kiểu này - nó treo hẳn GPU (timeout), không phải lỗi biên dịch.
// SỬA: bỏ hẳn threadgroup memory. Đổi thứ tự vòng lặp: k làm vòng NGOÀI (mỗi
// thread đọc a[m,k] từ device memory ĐÚNG 1 LẦN, giữ trong 1 register cục bộ
// `aVal`), rr (MMQ4_NDST hàng) làm vòng TRONG dùng lại chính register đó -
// vẫn đạt mục tiêu "không đọc lại activation N/4 lần" như bản shared-memory,
// nhưng an toàn tuyệt đối vì không cần cấp phát shared memory nào cả.
// SUA BUG NGHIEM TRONG (token rac tren Intel Iris Plus 640 that qua Metal):
// Ban truoc dung simd_sum() + thread_index_in_simdgroup, GIA DINH 1
// threadgroup 32-thread == DUNG 1 simdgroup phan cung (nhu comment cu ghi
// "CHUA CHAY THU TREN GPU METAL THAT" - chua bao gio duoc kiem chung tren
// may that). Tren GPU Intel (Gen9 va nhieu Intel GPU khac) qua Metal,
// threads_per_simdgroup THUC TE co the la 8 hoac 16, KHONG dam bao 32 -
// khi do 1 threadgroup 32 thread bi chia thanh NHIEU simdgroup phan cung
// rieng biet, moi simdgroup co tiisg==0 RIENG -> NHIEU thread cung ghi de
// c[m*N+n] (race condition, thread ghi sau cung thang) VA moi simd_sum()
// chi cong duoc PHAN K ma simdgroup DO xu ly, khong phai tong ca K -> ket
// qua SAI cho MOI phep Linear dung weight Q4 (gan nhu toan bo model luong
// tu hoa) - day la nguyen nhan "token rac" tren Metal (dung y het bug da
// tim thay va sua o ban Vulkan - matmul_q4.comp).
// FIX: bo simd_sum/thread_index_in_simdgroup, dung threadgroup (shared)
// memory + tree reduction theo DUNG thread_position_in_threadgroup (luon
// la 0..31 duy nhat trong CA threadgroup, khong phu thuoc simdgroup phan
// cung report bao nhieu) - dung nguyen tac da ap dung cho ban Vulkan.
#define MMQ4_NDST 8
// SUA: tang tu 4 len 8 (dong bo voi Vulkan + const MMQ4_NDST trong
// vulkan_driver.nim) - tang tai su dung moi lan doc a[k] tren 8 cot thay
// vi 4, giam ~50% so threadgroup dispatch, cai thien ty le FLOPs/byte
// doc tu buffer A. AN TOAN tang vi kernel nay dung threadgroup shared-
// memory reduction (khong dua vao simd_sum/thread_index_in_simdgroup),
// khong con phu thuoc gia dinh simdgroup phan cung nao ca.
#define MMQ4_LANES 32
// SUA HIEU NANG (kernel nay CHAM hon ban simd_sum cu vi 2 ly do): (1) moi
// threadgroup chi co 32 thread -> so threadgroup dispatch nhieu gap 4 lan
// so voi ban cu (ban cu gom MMQ4_SG_PER_TG=4 simdgroup/threadgroup), (2)
// 6 threadgroup_barrier/block (1 build tier2 + 5 tang tree-reduce) tren
// GPU it thread song song hon vi threadgroup memory 9.7KB/threadgroup gioi
// han occupancy. FIX: gom MMQ4_GROUPS_PER_TG=4 "logical group" 32-lane vao
// CHUNG 1 threadgroup 128 thread - moi logical group tu xu ly 1 rowBlock
// rieng, dung "lane = lid % 32" thay cho "lid" lam chi so trong nhom -
// giam so threadgroup dispatch 4 lan (bang ban simd_sum cu) MA VAN giu
// nguyen tac barrier + threadgroup memory (khong quay lai gia dinh
// simdgroup phan cung = 32). tier2 (tach nibble) dung CHUNG ca 128 thread
// (khong doi theo group). tier1 (dequant LUT) + curGroup gio la
// threadgroup memory dung chung cho CA logical group (khong con rieng
// tung lane) - AN TOAN vi gr=k/groupSize CHI phu thuoc k, ma k tai 1 vong
// lap la GIONG HET nhau cho ca 128 thread (moi logical group deu lap
// k=lane, lane+32,... voi lane=0..31 nhu nhau) => dieu kien "gr doi" la
// UNIFORM tren toan bo threadgroup, threadgroup_barrier ben trong nhanh
// re nay van HOP LE (khong bi thread nao di vao nhanh khac thread khac).
// GIA DINH: groupSize % 32 == 0 (dung voi cac groupSize Q4 pho bien 32/
// 64/128) de dam bao 32 gia tri k lien tiep trong 1 vong lap khong bao
// gio bi cat ngang boi 1 ranh gioi group - neu groupSize khac boi so 32,
// build lai tier1 co the xay ra tre 1 vong doi voi vai lane (van ĐÚNG ve
// gia tri cuoi cung, chi mat bot loi ich cache, khong sai ket qua).
#define MMQ4_GROUPS_PER_TG 4
#define MMQ4_TG_SIZE (MMQ4_LANES * MMQ4_GROUPS_PER_TG)   // 128

// ============================================================
// SUA (muon Y TUONG tu llama.cpp/ggml Metal backend - kernel_mul_mv_q4_K_f32
// - xem ggml-metal.metal, N_SIMDWIDTH=32 "assuming SIMD group size is 32"):
// ggml GIA DINH CUNG simdgroup=32 tren MOI GPU Apple (ke ca Intel iGPU qua
// Metal) va dung THANG simd_sum()/thread_index_in_simdgroup - DA XAC NHAN
// CHAY DUNG + NHANH (3 tok/s) tren CHINH Intel Iris Plus 640 (bang chung
// thuc te tu llama-cli chay that, khong phai suy doan). Dieu nay cho thay
// noi lo truoc day cua kernel nay ("threads_per_simdgroup Intel Gen9 co the
// la 8/16 khong dam bao 32" - xem lich su sua doi ban duoi threadgroup-
// memory) la THUA/qua an toan tren driver Iris 640 hien tai - gay cham hon
// can thiet (threadgroup shared-memory + nhieu barrier hon simd_sum that).
// GIU NGUYEN layout du lieu rieng cua project (wq nibble-pack + scales/
// zeros theo group, KHONG doi sang block_q4_K cua ggml - doi layout se can
// viet lai ca quant.nim, rui ro bug bit-packing, khong dang de "tien").
// CHI muon KY THUAT dispatch: 1 threadgroup = DUNG 1 simdgroup (32 thread,
// dung y N_SIMDWIDTH cua ggml), MMQ4_NDST=8 hang output/threadgroup, dung
// simd_sum() that thay vi threadgroup tree-reduce thu cong - bo HOAN TOAN
// threadgroup memory (tier1Tg/tier2Lo/tier2Hi/sdata cua ban truoc), vi
// simd_sum khong can shared memory nao ca.
// AN TOAN ve dynamic-index-register (nguyen nhan Timeout THAT truoc day):
// sc[]/zp[] la private array KICH THUOC CO DINH (MMQ4_NDST=8), CHI duoc
// index bang #pragma unroll (compile-time constant rr), KHONG BAO GIO
// index bang bien runtime (nibble/k) - khac han tier1[16] ban cu bi index
// bang "nibble" (runtime) gay tran thanh ghi/scratch-memory-cham.
kernel void matmul_q4_kernel(
    device const float* a          [[buffer(0)]],   // [M,K]
    device const uchar* wq         [[buffer(1)]],   // [N, ceil(K/2)] packed int4
    device const float* scales     [[buffer(2)]],   // [N * nGroupsPerRow]
    device const float* zeros      [[buffer(3)]],   // [N * nGroupsPerRow]
    device float* c                [[buffer(4)]],   // [M,N]
    device const int& M            [[buffer(5)]],
    device const int& K            [[buffer(6)]],
    device const int& N            [[buffer(7)]],
    device const int& groupSize    [[buffer(8)]],
    device const int& nGroupsPerRow [[buffer(9)]],
    device const int& simdWidth    [[buffer(10)]],  // THEM MOI - xem giai thich duoi
    uint3 tgpig [[threadgroup_position_in_grid]],   // (rowBlock, m, 0)
    uint3 tiisgv [[thread_position_in_threadgroup]] // .x = 0..simdWidth-1 - BAT BUOC dispatch threadgroup size = (simdWidth,1,1)
    // uint3 (khong phai ushort/uint don) vi tron vector+scalar attribute
    // cung argument list gay loi bien dich MSL (da gap va sua o cac kernel
    // khac trong file nay - xem attention_fused_flash_kernel).
    // SUA BUG NGHIEM TRONG (nguyen nhan THAT cua "token rac" sau khi doi
    // sang simd_sum): ban truoc HARDCODE 32 lam ca stride vong lap LAN
    // dieu kien ghi ket qua (tiisg==0), TU Y GIA DINH "threadgroup 32
    // thread = DUNG 1 simdgroup phan cung". Day la gia dinh CHUA TUNG duoc
    // kiem chung that (chi dua vao dong comment "assuming SIMD group size
    // is 32" cua ggml - do la GIA DINH DINH CO DISPATCH cua ho, KHONG
    // phai bang chung phan cung, va code ggml VAN DUNG du gia dinh do sai
    // vi ho dung dung builtin [[thread_index_in_simdgroup]] + tu xu ly
    // nhieu simdgroup/threadgroup qua sgitg - con ban truoc thi KHONG).
    // Neu simdgroup phan cung THAT cua Iris 640 la 8 hoac 16 (rat co the
    // - kien truc Gen9 Intel), threadgroup 32 thread se TU DONG bi chia
    // thanh 2-4 simdgroup phan cung RIENG BIET - simd_sum() CHI cong
    // trong PHAM VI 1 simdgroup phan cung do, va "tiisg==0" (vi tri 0
    // TRONG THREADGROUP) chi la lane-0 cua simdgroup DAU TIEN - BO SOT
    // hoan toan dong gop cua cac simdgroup con lai -> tong SAI -> token
    // rac (khong crash vi day la tinh toan hop le nhung SAI, khong phai
    // truy cap bo nho ngoai pham vi).
    // FIX DUNG GOC: KHONG doan so 32 nua - phia host (metal_shim.m) doc
    // THAT pipeline.threadExecutionWidth (Metal API chinh thuc, tra ve
    // dung do rong simdgroup phan cung cua PIPELINE DO tren GPU DANG
    // CHAY) roi dispatch CHINH XAC bang so do cho threadgroup - dam bao
    // threadgroup LUON DUNG BANG 1 simdgroup phan cung THAT, bat ke phan
    // cung la 8/16/32. Khi do "thread_position_in_threadgroup" va
    // "thread_index_in_simdgroup" TRUNG NHAU (0..simdWidth-1) - tiisg==0
    // luc nay moi DUNG la lane-0 DUY NHAT cua CA threadgroup.
) {
    uint tiisg = tiisgv.x;   // 0..simdWidth-1
    int m = int(tgpig.y);
    if (m >= M) return;

    #define MMQ4_NDST 8   // so hang output / threadgroup (= / simdgroup)
    int rowBlock = int(tgpig.x);
    int n0 = rowBlock * MMQ4_NDST;
    int bytesPerRow = (K + 1) / 2;

    float sum[MMQ4_NDST];
    #pragma unroll
    for (int rr = 0; rr < MMQ4_NDST; ++rr) sum[rr] = 0.0f;

    int prevGr = -1;
    float sc[MMQ4_NDST];
    float zp[MMQ4_NDST];

    for (int k = int(tiisg); k < K; k += simdWidth) {
        float aVal = a[m * K + k];
        int gr = groupSize > 0 ? (k / groupSize) : 0;

        // Cache scale/zero theo group HIEN TAI - CHI doc lai device memory
        // khi group thuc su doi (dung y het tier1 ban cu, nhung gio la
        // private array/thread thay vi threadgroup - AN TOAN vi index rr
        // la compile-time qua #pragma unroll, KHONG phai runtime).
        if (gr != prevGr) {
            prevGr = gr;
            #pragma unroll
            for (int rr = 0; rr < MMQ4_NDST; ++rr) {
                int n = n0 + rr;
                if (n < N) {
                    sc[rr] = scales[n * nGroupsPerRow + gr];
                    zp[rr] = zeros[n * nGroupsPerRow + gr];
                } else {
                    sc[rr] = 0.0f;
                    zp[rr] = 0.0f;
                }
            }
        }

        int byteIdx = k >> 1;
        bool lowNibble = (k & 1) == 0;
        #pragma unroll
        for (int rr = 0; rr < MMQ4_NDST; ++rr) {
            int n = n0 + rr;
            if (n >= N) continue;
            uchar byteVal = wq[n * bytesPerRow + byteIdx];
            uchar nibble = lowNibble ? (byteVal & 0x0Fu) : ((byteVal >> 4) & 0x0Fu);
            sum[rr] += aVal * ((float(nibble) - zp[rr]) * sc[rr]);
        }
    }

    // simd_sum() THAT - dung y ggml kernel_mul_mv_q4_K_f32 (khong con
    // threadgroup_barrier/tree-reduce thu cong nao ca). DUNG vi threadgroup
    // = DUNG 1 simdgroup phan cung THAT (dispatch tu host bang chinh
    // simdWidth nay), khong con gia dinh nao ca.
    #pragma unroll
    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        float total = simd_sum(sum[rr]);
        if (tiisg == 0u) {
            int n = n0 + rr;
            if (n < N) c[m * N + n] = total;
        }
    }
    #undef MMQ4_NDST
}

// ============================================================
// BIEN THE THU NGHIEM - GIAM MMQ4_NDST (8 -> 2/4, theo dung
// N_R0_Q4_K=2 cua ggml kernel_mul_mv_q4_K_f32_impl) + doc nibble
// THEO CAP (k2,k2+1) bang 1 lan doc uchar + bitmask thuan, KHONG
// nhanh runtime nao (dung y "diem khac biet #1" nguoi dung neu ra
// khi doi chieu voi ggml).
//
// LICH SU: ban dau thu NDST=8 + pairing -> DO THAT tren may Intel
// Iris Plus 640 CHAM HON ban goc (xem log nguoi dung gui: ~2-3x
// cham hon o cung vi tri token). Nghi ngo goc: NDST=8 nghia la
// MOI iteration phai giu song CUNG LUC 8 phan tu sum[]/sc[]/zp[]
// CONG THEM aVal0/aVal1/nib0/nib1/hasK1 moi vong lap - Iris co
// register file rat nho (da tung gay bug tran thanh ghi khac trong
// chinh file nay - xem lich su sua doi tier1[]/acc[] o cac kernel
// khac) => nhieu kha nang bi SPILL, mat het loi ich giam so lan
// doc bo nho.
//
// ggml chon N_R0_Q4_K CHI = 2 (khong phai 8) - it ap luc thanh ghi
// hon HAN. Bien the nay lam DUNG y ggml: NDST nho (2 hoac 4) +
// pairing, tach rieng thanh kernel MOI (KHONG dung ten
// "matmul_q4_kernel" cu) de A/B test that tren may that, khong
// ghi de len ban da duoc do la ON DINH (0.86 tok/s).
//
// CHUA CHAY THU TREN GPU METAL THAT - giong tinh trang ban dau cua
// matmul_q4_kernel truoc day, BAT BUOC tu do lai tren may that
// truoc khi dung cho sinh token that. Neu cham hon/sai, quay lai
// "matmul_q4_kernel" (khong doi gi ca, van con nguyen ben tren).
#define BB_MATMUL_Q4_VARIANT(NDST_VAL, KFUNC)                                \
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
    int m = int(tgpig.y);                                                   \
    if (m >= M) return;                                                     \
                                                                              \
    const int NDST = NDST_VAL;                                              \
    int rowBlock = int(tgpig.x);                                            \
    int n0 = rowBlock * NDST;                                               \
    int bytesPerRow = (K + 1) / 2;                                          \
                                                                              \
    float sum[NDST_VAL];                                                    \
    _Pragma("unroll")                                                       \
    for (int rr = 0; rr < NDST_VAL; ++rr) sum[rr] = 0.0f;                   \
                                                                              \
    int prevGr = -1;                                                        \
    float sc[NDST_VAL];                                                     \
    float zp[NDST_VAL];                                                     \
                                                                              \
    /* SUA (tach zero-point ra khoi vong lap chinh - hoc tu cach ggml     \
       xu ly q4_1 bang sumy): thay vi tru zp[rr] roi nhan sc[rr] MOI      \
       phan tu MOI rr (1 tru + 2 nhan x NDST lan/phan tu), gio chi cong  \
       don dot[rr] += aVal*nibble (1 nhan x NDST lan/phan tu) va mot     \
       sumA (Sigma aVal) DUNG CHUNG cho moi rr trong CUNG 1 group. Khi   \
       group doi (hoac cuoi vong lap), moi flush 1 lan:                  \
       sum[rr] += sc[rr]*dot[rr] - zp[rr]*sc[rr]*sumA - toan hoc y het   \
       cong thuc cu (phan phoi phep nhan), chi doi THU TU cong don de   \
       giam ALU trong vong lap nong. An toan voi NDST=2/4 (register     \
       them: dot[NDST_VAL] + 1 scalar sumA, nhe hon nhieu so luc NDST=8 \
       gay spill - xem comment lich su o tren). */                      \
    float dot[NDST_VAL];                                                    \
    _Pragma("unroll")                                                       \
    for (int rr = 0; rr < NDST_VAL; ++rr) dot[rr] = 0.0f;                   \
    float sumA = 0.0f;                                                      \
                                                                              \
    /* groupSize duoc gia dinh CHAN (dung voi 32/64/128/256 pho bien)      \
       de k2 va k2+1 luon cung 1 group - xem giai thich chi tiet o         \
       phien ban NDST=8 da thu (comment o tren, cung logic). */            \
    for (int k2 = int(tiisg) * 2; k2 < K; k2 += int(simdWidth) * 2) {       \
        float aVal0 = a[m * K + k2];                                        \
        bool hasK1 = (k2 + 1) < K;                                          \
        float aVal1 = hasK1 ? a[m * K + k2 + 1] : 0.0f;                     \
        int gr = groupSize > 0 ? (k2 / groupSize) : 0;                      \
                                                                              \
        if (gr != prevGr) {                                                 \
            if (prevGr != -1) {                                             \
                _Pragma("unroll")                                           \
                for (int rr = 0; rr < NDST_VAL; ++rr) {                     \
                    sum[rr] += sc[rr] * dot[rr] - zp[rr] * sc[rr] * sumA;   \
                    dot[rr] = 0.0f;                                          \
                }                                                            \
                sumA = 0.0f;                                                \
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
        sumA += aVal0 + (hasK1 ? aVal1 : 0.0f);                             \
                                                                              \
        int byteIdx = k2 >> 1;                                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) {                             \
            int n = n0 + rr;                                                \
            if (n >= N) continue;                                          \
            uchar byteVal = wq[n * bytesPerRow + byteIdx];                  \
            uchar nib0 = byteVal & 0x0Fu;                                   \
            uchar nib1 = (byteVal >> 4) & 0x0Fu;                            \
            dot[rr] += aVal0 * float(nib0);                                 \
            if (hasK1) dot[rr] += aVal1 * float(nib1);                     \
        }                                                                    \
    }                                                                        \
    /* flush group cuoi cung (vong lap ket thuc ma khong co lan doi group \
       nao nua de trigger flush o tren). */                                \
    if (prevGr != -1) {                                                     \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) {                             \
            sum[rr] += sc[rr] * dot[rr] - zp[rr] * sc[rr] * sumA;          \
        }                                                                    \
    }                                                                        \
                                                                              \
    for (int rr = 0; rr < NDST_VAL; ++rr) { \
        partial[rr * simdWidth + tiisg] = sum[rr]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint stride = uint(simdWidth) / 2; stride > 0; stride /= 2) { \
        if (tiisg < stride) { \
            for (int rr = 0; rr < NDST_VAL; ++rr) \
                partial[rr * simdWidth + tiisg] += partial[rr * simdWidth + tiisg + stride]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (tiisg == 0) { \
        for (int rr = 0; rr < NDST_VAL; ++rr) \
            if (n0 + rr < N) c[m * N + n0 + rr] = partial[rr * simdWidth]; \
    } \
}

// 2 bien the de do that: NDST=2 (dung khop ggml N_R0_Q4_K), NDST=4
// (trung gian). host (metal_shim.m) PHAI dung dung so nay khi tinh
// rowBlocks (kMMQ4_NDST) cho DUNG ten kernel dang goi - xem comment
// trong metal_shim.m cho huong dan doi ten kernel de test.
BB_MATMUL_Q4_VARIANT(2, matmul_q4_kernel_ndst2_paired)
BB_MATMUL_Q4_VARIANT(4, matmul_q4_kernel_ndst4_paired)
#undef BB_MATMUL_Q4_VARIANT

// Kernel gốc (naive) giữ lại để đối chiếu/rollback nếu kernel simd-reduction
// phía trên ra kết quả sai trên máy thật (nghi ngờ đầu tiên: layout nibble
// hoặc offset group sai - xem cùng công thức y hệt, chỉ khác cách reduce).
kernel void matmul_q4_kernel_naive(
    device const float* a          [[buffer(0)]],
    device const uchar* wq         [[buffer(1)]],
    device const float* scales     [[buffer(2)]],
    device const float* zeros      [[buffer(3)]],
    device float* c                [[buffer(4)]],
    device const int& M            [[buffer(5)]],
    device const int& K            [[buffer(6)]],
    device const int& N            [[buffer(7)]],
    device const int& groupSize    [[buffer(8)]],
    device const int& nGroupsPerRow [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int n = int(gid.x);
    int m = int(gid.y);
    if (m >= M || n >= N) return;

    int bytesPerRow = (K + 1) / 2;
    int rowByteOff = n * bytesPerRow;
    int rowGroupOff = n * nGroupsPerRow;

    // TIER1 giong ban toi uu o tren (khong can TIER2 o day - kernel naive
    // la 1 thread/row/col nen khong co gi de "dung chung giua threads" ca,
    // tach bit bang shift/mask van re nhu tra bang cho 1 thread don le).
    float sum = 0.0f;
    int curGroup = -1;
    float tier1[16];
    for (int k = 0; k < K; ++k) {
        int g = groupSize > 0 ? (k / groupSize) : 0;
        if (g != curGroup) {
            curGroup = g;
            float sc = scales[rowGroupOff + g];
            float zp = zeros[rowGroupOff + g];
            for (int v = 0; v < 16; ++v) tier1[v] = (float(v) - zp) * sc;
        }
        uchar byteVal = wq[rowByteOff + (k >> 1)];
        uchar nibble = (k & 1) == 0 ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
        sum += a[m * K + k] * tier1[nibble];
    }
    c[m * N + n] = sum;
}

// ============================================================
// ATOMIC ADD FLOAT
// ============================================================
inline void atomicAddFloatDevice(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    float expectedF = as_type<float>(expected);
    while (!atomic_compare_exchange_weak_explicit(
              addr, &expected,
              as_type<uint>(expectedF + val),
              memory_order_relaxed, memory_order_relaxed)) {
        expectedF = as_type<float>(expected);
    }
}

// ============================================================
// ATTENTION FUSED FORWARD
// ============================================================
kernel void attention_fused_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device float* s_matrix [[buffer(4)]],
    device const int& B [[buffer(5)]],
    device const int& H [[buffer(6)]],
    device const int& S [[buffer(7)]],
    device const int& D [[buffer(8)]],
    device const float& scale [[buffer(9)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint bh = position.x;
    uint ti = position.y;
    
    if (bh < (uint)(B * H) && ti < (uint)S) {
        uint base_idx = bh * S * D;
        uint base_s = bh * S * S;
        uint row = base_s + ti * S;

        // SUA: bo local array "float scores[256]" (tran bo nho neu S > 256,
        // vi day la mang co dinh tren stack cua thread, khong lien quan gi
        // toi kich thuoc thuc te cua S). Dung thang s_matrix (buffer global,
        // da duoc cap phat dung S*S theo tung lan goi thuc te tu phia Nim)
        // lam bo nho tam -> khong con gioi han cung nao ve S nua.
        float mx = -1e30f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d];
            }
            float sc = dot * scale;
            s_matrix[row + tj] = sc;
            if (sc > mx) mx = sc;
        }

        float sum_exp = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float e = exp(s_matrix[row + tj] - mx);
            s_matrix[row + tj] = e;
            sum_exp += e;
        }

        for (uint tj = 0; tj <= ti; ++tj) {
            s_matrix[row + tj] /= sum_exp;
        }
        for (uint tj = ti + 1; tj < (uint)S; ++tj) {
            s_matrix[row + tj] = 0.0f;
        }

        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (uint tj = 0; tj <= ti; ++tj) {
                acc += s_matrix[row + tj] * v[base_idx + tj * D + d];
            }
            o[base_idx + ti * D + d] = acc;
        }
    }
}

// ============================================================
// ATTENTION FUSED - FLASH (prefill, causal), MOI THEM.
// attention_fused_kernel (ban tren) materialize toan bo ma tran s_matrix
// [S,S] ra device memory - O(S^2) bo nho + 3 lan doc/ghi toan bo ma tran
// (tinh score, exp, chia sum). Kernel nay thay bang online-softmax MOT
// LAN duy nhat qua tj=0..ti (dung dinh nghia flash attention: giu running
// max m_i, running sum l_i, va running output accumulator acc[d], moi
// khi gap score moi lon hon max cu thi RESCALE acc/l theo he so
// exp(m_cu - m_moi) thay vi tinh lai tu dau) - KHONG can buffer S*S nao,
// chi 1 lan doc q/k/v, khong doc/ghi lai bat cu gia tri trung gian nao.
// GIU NGUYEN dispatch pattern cu (1 thread / (bh,ti), khong dung
// threadgroup) - CHI doi than kernel, khong doi cach goi.
// KHONG dung lai attention_fused_kernel cu (van giu nguyen o tren) vi
// attention_fused_backward_kernel (huan luyen) can s_matrix (xac suat
// softmax da materialize) lam input - kernel flash nay CHI dung cho suy
// luan (inference forward), khong co gradient.
// GIOI HAN: D (head_dim) <= FLASH_MAX_D (128, du cho hau het kien truc
// pho bien: Llama/Mistral D=128, nhieu model khac D=64/80/96). Neu D>128,
// kernel tra ve som (khong ghi gi vao o[]) - fallback ve
// attention_fused_kernel cu cho truong hop nay.
// LY DO dung #pragma unroll cho vong lap d thay vi "for (int d=0;d<D;++d)"
// binh thuong: acc[] la private array/thread - index BANG BIEN CHAY THOI
// (runtime d) tren mang private se KHONG co dynamic-index register tren
// Intel Iris (dung y het nguyen nhan "GPU Timeout" da gap voi tier1[] o
// matmul_q4_kernel truoc day) -> BAT BUOC unroll ve compile-time index.
#define FLASH_MAX_D 128

// SUA LAI HOAN TOAN (nguyen nhan that cua GPU treo/timeout): ban truoc
// dung mang PRIVATE acc[FLASH_MAX_D] (128 float = 512 byte) SONG SUOT ca
// vong lap tj (chay toi S lan) TRONG TUNG THREAD RIENG LE - Intel Iris co
// register file rat nho, 512 byte/thread cho rieng acc[] (chua ke
// dot/sc/m_i/l_i/aOld/p) chac chan TRAN REGISTER, driver phai spill xuong
// bo nho rieng tung thread - DUNG Y HET lop bug da gap voi tier1[] o
// matmul_q4_kernel (khac o cho tier1[] la LUT nho, con acc[] o day lon hon
// va song lau hon nhieu). FIX THAT SU (khong phai roi ve non-flash): chia
// D ra cho CA MOT THREADGROUP lam chung (giong cach attention_decode_gqa_
// kernel da lam), acc[] chuyen vao THREADGROUP MEMORY (khong phai register
// rieng tung thread nua - threadgroup memory KHONG co van de dynamic-index-
// gay-tran-register, day la vung nho that su tren chip, khong phai
// register file). Van la flash attention DUNG NGHIA: 1 lan duy nhat qua
// tj=0..ti, running max/sum, KHONG materialize S*S, KHONG mang private
// lon nao ca.
#define ATTN_FLASH_TG 64

kernel void attention_fused_flash_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device const int& B [[buffer(4)]],
    device const int& H [[buffer(5)]],
    device const int& S [[buffer(6)]],
    device const int& D [[buffer(7)]],
    device const float& scale [[buffer(8)]],
    uint3 tgpigv [[threadgroup_position_in_grid]],   // .x = bh, .y = ti (uint3 - xem ly do o duoi)
    uint3 tidv   [[thread_position_in_threadgroup]]  // .x = 0..ATTN_FLASH_TG-1 - BAT BUOC dispatch threadgroup size = (ATTN_FLASH_TG,1,1)
    // SUA loi bien dich MSL (dung y het fix da ap dung cho matmul_q4_kernel):
    // "expecting input declarations with either all scalar types or all
    // vector types with the same number of elements" - trinh bien dich
    // Metal KHONG cho phep tron 1 tham so vector (uint2 tgpig) voi 1 tham
    // so scalar (uint tid) cung co attribute vi tri thread trong CUNG
    // argument list. Fix: khai ca 2 deu la uint3, roi lay .xy/.x tuong ung.
) {
    uint2 tgpig = tgpigv.xy;
    uint tid = tidv.x;
    uint bh = tgpig.x;
    uint ti = tgpig.y;
    if (bh >= (uint)(B * H) || ti >= (uint)S) return;
    if (D > FLASH_MAX_D) return;   // xem GIOI HAN o dinh file (128, du cho hau het kien truc pho bien)

    // SUA (theo yeu cau: luu o "RAM nho cua moi thread" = thanh ghi/private,
    // KHONG dung threadgroup memory cho acc nua). FLASH_MAX_D=128,
    // ATTN_FLASH_TG=64 => moi thread giu TOI DA 2 phan tu (FLASH_MAX_D/
    // ATTN_FLASH_TG). Dung mang private co KICH THUOC CO DINH bang so
    // "slot" nay (khong phai FLASH_MAX_D) va CHI truy cap bang #pragma
    // unroll voi index SLOT tinh (0,1,...) - KHONG BAO GIO index bang bien
    // chay "d" truc tiep - de trinh bien dich gan duoc vao thanh ghi thay
    // vi spill ra scratch memory (day chinh la nguyen nhan GPU Timeout da
    // gay ra truoc do voi tier1[]/acc[] kieu dynamic-index).
    #define ACC_SLOTS ((FLASH_MAX_D + ATTN_FLASH_TG - 1) / ATTN_FLASH_TG)
    float acc[ACC_SLOTS];
    #pragma unroll
    for (int s = 0; s < ACC_SLOTS; ++s) acc[s] = 0.0f;

    threadgroup float redBuf[ATTN_FLASH_TG];
    threadgroup float sM, sL, sAOld, sP;

    if (tid == 0u) { sM = -1e30f; sL = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = bh * (uint)S * (uint)D;
    uint qOff = base + ti * (uint)D;

    for (uint tj = 0u; tj <= ti; ++tj) {
        uint kOff = base + tj * (uint)D;

        // --- dot(q_ti, k_tj) cooperatively: moi thread cong 1 phan D, roi
        // tree-reduce trong threadgroup memory (KHONG phai mang private) ---
        float partial = 0.0f;
        for (int d = int(tid); d < D; d += ATTN_FLASH_TG) partial += q[qOff + (uint)d] * k[kOff + (uint)d];
        redBuf[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_FLASH_TG / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) redBuf[tid] += redBuf[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float dot = redBuf[0] * scale;   // moi thread deu doc duoc gia tri giong nhau sau barrier cuoi

        // --- cap nhat online-softmax: CHI thread 0 tinh (scalar, khong can
        // song song), broadcast qua threadgroup scalar cho ca nhom dung ---
        if (tid == 0u) {
            float m_old = sM;
            float new_m = max(m_old, dot);
            float aOld = exp(m_old - new_m);
            float p = exp(dot - new_m);
            sL = sL * aOld + p;
            sM = new_m;
            sAOld = aOld;
            sP = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float aOld = sAOld;
        float p = sP;
        // s la SLOT tinh (compile-time qua unroll) - d = tid + s*ATTN_FLASH_TG
        // van la runtime nhung CHI dung de doc/ghi device memory (v[], hop le
        // du D thay doi), KHONG dung de index vao mang private acc[] nua.
        #pragma unroll
        for (int s = 0; s < ACC_SLOTS; ++s) {
            int d = int(tid) + s * ATTN_FLASH_TG;
            if (d < D) acc[s] = acc[s] * aOld + p * v[kOff + (uint)d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // xong tj nay, an toan ghi de redBuf o vong sau
    }

    float invL = 1.0f / sL;
    #pragma unroll
    for (int s = 0; s < ACC_SLOTS; ++s) {
        int d = int(tid) + s * ATTN_FLASH_TG;
        if (d < D) o[qOff + (uint)d] = acc[s] * invL;
    }
}
#undef ACC_SLOTS

// ============================================================
// ATTENTION DECODE (GQA) - v2, SONG SONG THẬT + SỬA LỖI FENCE.
// v1 dùng 1 thread/head (chỉ nRep=4-8 thread tổng) - lãng phí gần hết GPU,
// đo thực tế chậm hơn ~4x so với đường cũ (1.9s/token -> 8s/token).
// v2 dùng 1 threadgroup (128 thread)/head + tree-reduction, NHƯNG lần đầu
// viết bị sai: dùng threadgroup_barrier(mem_flags::mem_threadgroup) trong
// khi `scores` là bộ nhớ `device` (không phải `threadgroup`) - cờ
// mem_threadgroup CHỈ đảm bảo đồng bộ cho biến threadgroup (shared_reduce),
// KHÔNG đảm bảo các thread khác đã ghi xong vào `scores` (device memory)
// trước khi đọc chéo ở Bước 3 - gây sai lệch ~0.02-0.1 (đọc phải giá trị
// scores cũ/dở dang). ĐÃ SỬA: dùng mem_flags::mem_device_and_threadgroup
// cho MỌI barrier trong kernel này (fence cả threadgroup lẫn device memory).
//   q: [H, D], k/v: [S, D] (dùng chung cho H head), o: [H, D]
//   scores: [H, S] scratch (bộ nhớ device, ĐỌC/GHI CHÉO giữa các thread)
// ============================================================
#define ATTN_DECODE_TG 128

// SUA (flash attention, MOI THEM): ban tren dung 3 vong lap qua S + 1
// buffer device `scores[H,S]` lam trung gian (ghi o Buoc 1, ghi de o Buoc
// 2, doc cheo o Buoc 3) - dung 1 lan device-memory fence rong (memory
// traffic ~3x kich thuoc scores). Ban FLASH nay chi 1 vong lap DUY NHAT
// qua cac KHOI (block) TG=128 key mot luc, giu running max/sum/output
// (dung y het thuat toan flash attention, ap dung cho truong hop 1 query
// - S key cua decode) - KHONG con doc/ghi `scores` (device) nua, thay
// bang `pBlock` (threadgroup memory, chi 512 byte) lam cho xac suat cua
// DUY NHAT 1 block dang xu ly. Tham so `scores` GIU LAI trong signature
// (khong dung nua) de KHONG phai doi shim/Nim binding (ABI cu van goi
// duoc y nguyen).
kernel void attention_decode_gqa_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device float* scores [[buffer(4)]],   // KHONG con dung - giu de ABI/shim khong doi
    device const int& H [[buffer(5)]],
    device const int& S [[buffer(6)]],
    device const int& D [[buffer(7)]],
    device const float& scale [[buffer(8)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_reduce[ATTN_DECODE_TG];
    threadgroup float pBlock[ATTN_DECODE_TG];   // xac suat (chua chuan hoa) cua block key dang xu ly
    if (h >= (uint)H) return;
    if (D > FLASH_MAX_D) return;   // xem gioi han o attention_fused_flash_kernel

    uint qBase = h * (uint)D;

    float m_i = -1e30f;    // running max
    float l_i = 0.0f;      // running sum (da theo thang m_i)
    // Moi thread chi giu PHAN acc cua (toi da) 1 chi so d = tid (vi
    // ATTN_DECODE_TG=128 >= FLASH_MAX_D=128 => moi d duoc dung DUNG 1
    // thread, khong can mang private nao ca -> khong co van de dynamic-
    // index register nhu ben attention_fused_flash_kernel).
    float accD = 0.0f;
    bool hasD = (tid < (uint)D);

    for (uint blockStart = 0u; blockStart < (uint)S; blockStart += ATTN_DECODE_TG) {
        uint tj = blockStart + tid;
        bool valid = tj < (uint)S;

        float dot = -1e30f;
        if (valid) {
            dot = 0.0f;
            uint kBase = tj * (uint)D;
            for (int d = 0; d < D; ++d) dot += q[qBase + (uint)d] * k[kBase + (uint)d];
            dot *= scale;
        }

        // --- tim max cua BLOCK nay (khong phai toan cuc - flash attention
        // rescale dan, khong can biet max toan cuc truoc) ---
        shared_reduce[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_DECODE_TG / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float blockMax = shared_reduce[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float new_m = max(m_i, blockMax);
        float aOld = exp(m_i - new_m);
        float p = valid ? exp(dot - new_m) : 0.0f;
        pBlock[tid] = p;

        // --- tong p cua block (tree-reduce) ---
        shared_reduce[tid] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_DECODE_TG / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_reduce[tid] += shared_reduce[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float blockSum = shared_reduce[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // dam bao pBlock[] da ghi xong truoc khi doc cheo ben duoi

        l_i = l_i * aOld + blockSum;

        // --- cap nhat output: moi thread (neu hasD) doc CHEO toan bo
        // pBlock[] (cua block nay, TG phan tu) nhan voi v[.., d=tid] ---
        if (hasD) {
            float blockAcc = 0.0f;
            uint blockLen = min((uint)ATTN_DECODE_TG, (uint)S - blockStart);
            for (uint j = 0u; j < blockLen; ++j) {
                blockAcc += pBlock[j] * v[(blockStart + j) * (uint)D + tid];
            }
            accD = accD * aOld + blockAcc;
        }
        m_i = new_m;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // xong block, an toan ghi de shared_reduce/pBlock o vong sau
    }

    if (hasD) {
        o[qBase + tid] = accD / l_i;
    }
}

// ============================================================
// ATTENTION FUSED BACKWARD
// ============================================================
kernel void attention_fused_backward_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* s_matrix [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* dq [[buffer(5)]],
    device atomic_uint* dk [[buffer(6)]],
    device atomic_uint* dv [[buffer(7)]],
    device const int& B [[buffer(8)]],
    device const int& H [[buffer(9)]],
    device const int& S [[buffer(10)]],
    device const int& D [[buffer(11)]],
    device const float& scale [[buffer(12)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint bh = position.x;
    uint ti = position.y;
    
    if (bh < (uint)(B * H) && ti < (uint)S) {
        uint base_idx = bh * S * D;
        uint base_s = bh * S * S;
        
        float softmaxW[256];
        for (uint tj = 0; tj <= ti; ++tj) {
            softmaxW[tj] = s_matrix[base_s + ti * S + tj];
        }
        
        float dSoftmax[256];
        for (uint tj = 0; tj <= ti; ++tj) {
            float dotVal = 0.0f;
            for (int d = 0; d < D; ++d) {
                float dyVal = dy[base_idx + ti * D + d];
                atomicAddFloatDevice(&dv[base_idx + tj * D + d], softmaxW[tj] * dyVal);
                dotVal += dyVal * v[base_idx + tj * D + d];
            }
            dSoftmax[tj] = dotVal;
        }
        
        float dotSum = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            dotSum += softmaxW[tj] * dSoftmax[tj];
        }
        
        for (uint tj = 0; tj <= ti; ++tj) {
            float dScore = softmaxW[tj] * (dSoftmax[tj] - dotSum) * scale;
            for (int d = 0; d < D; ++d) {
                dq[base_idx + ti * D + d] += dScore * k[base_idx + tj * D + d];
                atomicAddFloatDevice(&dk[base_idx + tj * D + d], dScore * q[base_idx + ti * D + d]);
            }
        }
    }
}
// ============================================================
// GATHER - dung cho permute/concat/split/stack tren GPU thay vi CPU loop.
// dst[id] = src[idx[id]].
// ============================================================
kernel void gather_kernel(
    device const float* src [[buffer(0)]],
    device const int*   idx [[buffer(1)]],
    device float*       dst [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    dst[id] = src[idx[id]];
}

// ============================================================
// ARGMAX (gia tri + chi so) theo threadgroup, cung thuat toan tree-reduce
// nhu argmax_reduce (CUDA) / argmax.comp (Vulkan). Moi threadgroup xu ly
// 1 cap (o,i); threadgroup size phai >= dimSize.
// ============================================================
kernel void argmax_kernel(
    device const float* x       [[buffer(0)]],
    device float*       outVal  [[buffer(1)]],
    device int*         outIdx  [[buffer(2)]],
    constant int&       outer   [[buffer(3)]],
    constant int&       dimSize [[buffer(4)]],
    constant int&       inner   [[buffer(5)]],
    uint pair [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    threadgroup float* sVal [[threadgroup(0)]],
    threadgroup int*   sIdx [[threadgroup(1)]]
) {
    if (pair >= uint(outer * inner)) return;
    int o = int(pair) / inner;
    int i = int(pair) % inner;

    if (int(d) < dimSize) {
        sVal[d] = x[(o * dimSize + int(d)) * inner + i];
        sIdx[d] = int(d);
    } else {
        sVal[d] = -FLT_MAX;
        sIdx[d] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            if (sVal[d + stride] > sVal[d]) {
                sVal[d] = sVal[d + stride];
                sIdx[d] = sIdx[d + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d == 0) {
        outVal[pair] = sVal[0];
        outIdx[pair] = sIdx[0];
    }
}

// ============================================================
// COMPARE - bitmask that tren GPU. op: 0=gt 1=lt 2=eq 3=ge 4=le.
// ============================================================
kernel void compare_kernel(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device uchar*       mask [[buffer(2)]],
    constant int&        op  [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float av = a[id], bv = b[id];
    uchar r = 0;
    if (op == 0) r = (av >  bv) ? 1 : 0;
    else if (op == 1) r = (av <  bv) ? 1 : 0;
    else if (op == 2) r = (av == bv) ? 1 : 0;
    else if (op == 3) r = (av >= bv) ? 1 : 0;
    else if (op == 4) r = (av <= bv) ? 1 : 0;
    mask[id] = r;
}

// ============================================================
// DROPOUT RNG - counter-based xorshift32, cung cong thuc voi CUDA/Vulkan.
// ============================================================
inline uint xorshift32_m(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

kernel void dropout_rng_kernel(
    device float*  mask   [[buffer(0)]],
    constant uint& seed   [[buffer(1)]],
    constant float& p     [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint state = seed ^ (id * 2654435761u + 1u);
    state = xorshift32_m(state);
    state = xorshift32_m(state);
    float u = float(state >> 8) * (1.0f / 16777216.0f);
    mask[id] = (u < p) ? 0.0f : scale;
}

// ============================================================
// PERMUTE - device tự giải mã chỉ số từ strides. Không có mảng chỉ số
// O(n) build trên host.
// ============================================================
kernel void permute_strided(
    device const float* src        [[buffer(0)]],
    device float*       dst        [[buffer(1)]],
    constant int*       newStrides [[buffer(2)]],
    constant int*       oldStrides [[buffer(3)]],
    constant int*       dimsMap    [[buffer(4)]],
    constant int&       ndim       [[buffer(5)]],
    uint linear [[thread_position_in_grid]]
) {
    int rem = int(linear);
    int oldLinear = 0;
    for (int i = 0; i < ndim; i++) {
        int idxI = rem / newStrides[i];
        rem = rem % newStrides[i];
        oldLinear += idxI * oldStrides[dimsMap[i]];
    }
    dst[linear] = src[oldLinear];
}

// ============================================================
// SPLIT - lát cắt liên tục theo dim, device tự tính offset.
// ============================================================
kernel void split_slice(
    device const float* src      [[buffer(0)]],
    device float*       dst      [[buffer(1)]],
    constant int&       dimTotal [[buffer(2)]],
    constant int&       thisSize [[buffer(3)]],
    constant int&       offset   [[buffer(4)]],
    constant int&       inner    [[buffer(5)]],
    uint linear [[thread_position_in_grid]]
) {
    int i = int(linear) % inner;
    int rem = int(linear) / inner;
    int d = rem % thisSize;
    int o = rem / thisSize;
    int srcLinear = (o * dimTotal + offset + d) * inner + i;
    dst[linear] = src[srcLinear];
}

// ============================================================
// CONCAT - device tự chọn nguồn qua bảng nhỏ O(numTensors). flatSrc do
// host nối phẳng (memcpy nối tiếp, không phải tính chỉ số per-element).
// ============================================================
kernel void concat_gather(
    device const float* flatSrc      [[buffer(0)]],
    device const int*   srcBase      [[buffer(1)]],
    device const int*   srcDimSize   [[buffer(2)]],
    device const int*   srcDimOffset [[buffer(3)]],
    constant int&       numTensors   [[buffer(4)]],
    device float*       dst          [[buffer(5)]],
    constant int&       totalDim     [[buffer(6)]],
    constant int&       inner        [[buffer(7)]],
    uint linear [[thread_position_in_grid]]
) {
    int i = int(linear) % inner;
    int rem = int(linear) / inner;
    int dGlobal = rem % totalDim;
    int o = rem / totalDim;

    int ti = 0;
    for (int k = 0; k < numTensors; k++) {
        if (dGlobal < srcDimOffset[k] + srcDimSize[k]) { ti = k; break; }
    }
    int dLocal = dGlobal - srcDimOffset[ti];
    int srcLinear = srcBase[ti] + (o * srcDimSize[ti] + dLocal) * inner + i;
    dst[linear] = flatSrc[srcLinear];
}
