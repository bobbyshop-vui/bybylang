// ============================================================
// attention_decode_all_heads_kernel
// ------------------------------------------------------------
// PURPOSE: replaces the CPU loop "for hkv in 0..<HKV: copyInto+matmulNT+
// softmax+matmul+copyInto" (32 heads x 5 dispatches = 160 dispatches/layer) with
// EXACTLY ONE dispatch handling ALL H heads at once on the GPU
// (one threadgroup per query head, running in parallel).
//
// Does NOT reuse attention_decode_gqa_kernel (the old kernel, which the
// codebase itself records as "confirmed to produce wrong results" - see gAttnDecodeGqaDisabled
// in nim_inference.nim) - written from scratch with the classic 4-phase
// softmax (not the old kernel's complex streaming "flash attention" style)
// so it is easy to check by eye, matching EXACTLY the math of the
// matmulNT+softmax+matmul path (known correct, used as the baseline):
//   scores[s] = dot(Q, K[s])
//   probs     = softmax(scores)            (max subtracted for numerical stability)
//   out[d]    = sum_s probs[s] * V[s][d]
//
// Buffer layout: Q[H*D] (RoPE + scale already applied), K_all/V_all
// [HKV*maxSeqLen*D] (CONTIGUOUS per head - the offset of head hkv is ALWAYS
// hkv*maxSeqLen*D; see kResidentCombined in nim_inference.nim for
// why the old kResident/vResident are NOT used - heap suballocation does not
// guarantee even spacing between heads).
//
// nRep = H / HKV (general GQA, although the current model has H==HKV so nRep=1):
// query head h uses KV head (h / nRep) - SAME as the hkv*nRep*D convention of the
// old CPU loop (each block of nRep consecutive query heads shares one KV head).
//
// Dispatch: H threadgroups (1 per query head), TG_SIZE threads per group.
// The "scores" threadgroup memory is sized DYNAMICALLY by the host (setThreadgroupMemoryLength)
// because S (sequence length) changes every token - a threadgroup pointer, not
// a fixed-size array.
// ============================================================

#define ATTN_ALL_TG 128

template <typename KV>
kernel void attention_decode_all_heads_t(
    device const float* Qbuf        [[buffer(0)]],   // [H*D]
    device const KV*    Kbuf        [[buffer(1)]],   // [HKV*maxSeqLen*D]
    device const KV*    Vbuf        [[buffer(2)]],   // [HKV*maxSeqLen*D]
    device float*       Obuf        [[buffer(3)]],   // [H*D]
    constant int&       H           [[buffer(4)]],
    constant int&       HKV         [[buffer(5)]],
    constant int&       D           [[buffer(6)]],
    constant int&       S           [[buffer(7)]],
    constant int&       maxSeqLen   [[buffer(8)]],
    threadgroup float*  scores      [[threadgroup(0)]],   // length S, sized by the host
    uint headIdx  [[threadgroup_position_in_grid]],
    uint tid      [[thread_position_in_threadgroup]])
{
    threadgroup float shared_reduce[ATTN_ALL_TG];

    int nRep = max(1, H / max(1, HKV));
    int hkv = int(headIdx) / nRep;
    if (hkv >= HKV) hkv = HKV - 1;   // guard against an uneven split (should not happen when H%HKV==0)

    device const float* q    = Qbuf + uint(headIdx) * uint(D);
    device const KV*    kBase = Kbuf + uint(hkv) * uint(maxSeqLen) * uint(D);
    device const KV*    vBase = Vbuf + uint(hkv) * uint(maxSeqLen) * uint(D);
    device float*       out  = Obuf + uint(headIdx) * uint(D);

    // ---- Pha 1: scores[s] = dot(Q, K[s]) cho s = tid, tid+TG, tid+2*TG, ... ----
    for (uint s = tid; s < uint(S); s += ATTN_ALL_TG) {
        device const KV* k = kBase + s * uint(D);
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += q[d] * float(k[d]);
        }
        scores[s] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 2: find the max (parallel reduction) to subtract before exp for stability ----
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

    // ---- Phase 3: scores[s] = exp(scores[s]-maxVal), accumulate the sum (parallel reduction) ----
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
    float invSum = 1.0f / max(sumVal, 1e-20f);   // avoid dividing by 0 if S=0 (should not happen)

    // ---- Pha 4: out[d] = invSum * sum_s scores[s]*V[s][d], song song theo d ----
    for (int d = int(tid); d < D; d += ATTN_ALL_TG) {
        float acc = 0.0f;
        for (uint s = 0; s < uint(S); s++) {
            acc += scores[s] * float(vBase[s * uint(D) + uint(d)]);
        }
        out[d] = acc * invSum;
    }
}
template [[host_name("attention_decode_all_heads_kernel")]] kernel decltype(attention_decode_all_heads_t<float>) attention_decode_all_heads_t<float>;
template [[host_name("attention_decode_all_heads_kernel_h")]] kernel decltype(attention_decode_all_heads_t<half>) attention_decode_all_heads_t<half>;
