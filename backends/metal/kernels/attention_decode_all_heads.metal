// ============================================================
// attention_decode_all_heads_kernel
// ------------------------------------------------------------
// MỤC ĐÍCH: thay thế vòng lặp CPU "for hkv in 0..<HKV: copyInto+matmulNT+
// softmax+matmul+copyInto" (32 head x 5 lệnh dispatch = 160 lệnh/layer) bằng
// ĐÚNG 1 LỆNH DISPATCH duy nhất, xử lý TOÀN BỘ H head cùng lúc trên GPU
// (mỗi threadgroup phụ trách 1 query head, chạy song song với nhau).
//
// KHÔNG tái dùng logic của attention_decode_gqa_kernel (kernel cũ, đã bị
// chính codebase ghi nhận là "đã xác nhận sinh sai" - xem gAttnDecodeGqaDisabled
// ở nim_inference.nim) - viết lại từ đầu, dùng thuật toán softmax cổ điển
// 4 pha (không dùng kiểu "flash attention" streaming phức tạp của kernel cũ)
// để dễ soát lỗi bằng mắt, đúng CHÍNH XÁC theo toán học mà đường
// matmulNT+softmax+matmul (đã chạy đúng, đang dùng làm baseline) thực hiện:
//   scores[s] = dot(Q, K[s])
//   probs     = softmax(scores)            (trừ max để ổn định số học)
//   out[d]    = sum_s probs[s] * V[s][d]
//
// Bố cục buffer: Q[H*D] (đã nhân RoPE + scale sẵn), K_all/V_all
// [HKV*maxSeqLen*D] (LAYOUT LIÊN TỤC theo head - offset head hkv LUÔN LÀ
// hkv*maxSeqLen*D, xem giải thích ở kResidentCombined trong nim_inference.nim
// vì sao KHÔNG dùng kResident/vResident cũ - suballocate từ heap không đảm
// bảo khoảng cách đều giữa các head).
//
// nRep = H / HKV (hỗ trợ GQA tổng quát, dù model hiện tại H==HKV nên nRep=1):
// query head h dùng KV head (h / nRep) - ĐÚNG NHƯ quy ước hkv*nRep*D trong
// vòng lặp CPU cũ (mỗi khối nRep query head liên tiếp dùng chung 1 KV head).
//
// Dispatch: H threadgroup (1 threadgroup/query head), TG_SIZE thread/nhóm.
// Threadgroup memory "scores" cấp phát ĐỘNG từ host (setThreadgroupMemoryLength)
// vì S (độ dài chuỗi) thay đổi mỗi token - dùng con trỏ threadgroup, không
// phải mảng cỡ cố định.
// ============================================================

#define ATTN_ALL_TG 128

kernel void attention_decode_all_heads_kernel(
    device const float* Qbuf        [[buffer(0)]],   // [H*D]
    device const float* Kbuf        [[buffer(1)]],   // [HKV*maxSeqLen*D]
    device const float* Vbuf        [[buffer(2)]],   // [HKV*maxSeqLen*D]
    device float*       Obuf        [[buffer(3)]],   // [H*D]
    constant int&       H           [[buffer(4)]],
    constant int&       HKV         [[buffer(5)]],
    constant int&       D           [[buffer(6)]],
    constant int&       S           [[buffer(7)]],
    constant int&       maxSeqLen   [[buffer(8)]],
    threadgroup float*  scores      [[threadgroup(0)]],   // độ dài S, cấp động từ host
    uint headIdx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]])
{
    threadgroup float shared_reduce[ATTN_ALL_TG];

    int nRep = max(1, H / max(1, HKV));
    int hkv = int(headIdx) / nRep;
    if (hkv >= HKV) hkv = HKV - 1;   // an toàn phòng chia lệch (không nên xảy ra nếu H%HKV==0)

    device const float* q    = Qbuf + uint(headIdx) * uint(D);
    device const float* kBase = Kbuf + uint(hkv) * uint(maxSeqLen) * uint(D);
    device const float* vBase = Vbuf + uint(hkv) * uint(maxSeqLen) * uint(D);
    device float*       out  = Obuf + uint(headIdx) * uint(D);

    // ---- Pha 1: scores[s] = dot(Q, K[s]) cho s = tid, tid+TG, tid+2*TG, ... ----
    for (uint s = tid; s < uint(S); s += ATTN_ALL_TG) {
        device const float* k = kBase + s * uint(D);
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q[d] * k[d];
        }
        scores[s] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Pha 2: tìm max (giảm dần song song) để trừ vào exp cho ổn định số học ----
    float localMax = -INFINITY;
    for (uint s = tid; s < uint(S); s += ATTN_ALL_TG) {
        localMax = max(localMax, scores[s]);
    }
    shared_reduce[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTN_ALL_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Pha 3: scores[s] = exp(scores[s]-maxVal), cộng dồn tổng (giảm dần song song) ----
    float localSum = 0.0f;
    for (uint s = tid; s < uint(S); s += ATTN_ALL_TG) {
        float e = exp(scores[s] - maxVal);
        scores[s] = e;
        localSum += e;
    }
    shared_reduce[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ATTN_ALL_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_reduce[tid] += shared_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sumVal = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = 1.0f / max(sumVal, 1e-20f);   // tránh chia 0 nếu S=0 (không nên xảy ra)

    // ---- Pha 4: out[d] = invSum * sum_s scores[s]*V[s][d], song song theo d ----
    for (int d = int(tid); d < D; d += ATTN_ALL_TG) {
        float acc = 0.0f;
        for (uint s = 0; s < uint(S); s++) {
            acc += scores[s] * vBase[s * uint(D) + uint(d)];
        }
        out[d] = acc * invSum;
    }
}
