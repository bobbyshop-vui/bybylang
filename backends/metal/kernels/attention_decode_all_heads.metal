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

#define ATTN_ALL_TG_MAX 1024

// One threadgroup per query head, a multiple of 128 threads (the host picks up to 512).
// Scores: one simdgroup per key, lanes split D (coalesced loads, simd_sum). Output: the
// threads form tgSize/128 key groups, each thread owns one d of the group's keys, and the
// groups are summed through threadgroup memory. D <= 128 * 8 per pass, D <= 16 * simd width.
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
    uint tid      [[thread_position_in_threadgroup]],
    uint tgSize   [[threads_per_threadgroup]],
    uint lane     [[thread_index_in_simdgroup]],
    uint sg       [[simdgroup_index_in_threadgroup]],
    uint W        [[threads_per_simdgroup]],
    uint nsg      [[simdgroups_per_threadgroup]])
{
    threadgroup float shared_reduce[ATTN_ALL_TG_MAX];
    threadgroup float red[8 * 128];

    int nRep = max(1, H / max(1, HKV));
    int hkv = int(headIdx) / nRep;
    if (hkv >= HKV) hkv = HKV - 1;

    device const float* q     = Qbuf + uint(headIdx) * uint(D);
    device const KV*    kBase = Kbuf + uint(hkv) * uint(maxSeqLen) * uint(D);
    device const KV*    vBase = Vbuf + uint(hkv) * uint(maxSeqLen) * uint(D);
    device float*       out   = Obuf + uint(headIdx) * uint(D);

    // ---- scores[s] = q . K[s], one simdgroup per key ----
    float qv[16];
    for (uint i = 0; i < 16; ++i) {
        const uint d = lane + i * W;
        qv[i] = d < uint(D) ? q[d] : 0.0f;
    }
    for (uint s = sg; s < uint(S); s += nsg) {
        device const KV* k = kBase + s * uint(D);
        float dot = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            const uint d = lane + i * W;
            if (d < uint(D)) dot += qv[i] * float(k[d]);
        }
        dot = simd_sum(dot);
        if (lane == 0) scores[s] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- max ----
    float localMax = -INFINITY;
    for (uint s = tid; s < uint(S); s += tgSize) localMax = max(localMax, scores[s]);
    shared_reduce[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- exp and sum ----
    float localSum = 0.0f;
    for (uint s = tid; s < uint(S); s += tgSize) {
        float e = exp(scores[s] - maxVal);
        scores[s] = e;
        localSum += e;
    }
    shared_reduce[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_reduce[tid] += shared_reduce[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sumVal = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = 1.0f / max(sumVal, 1e-20f);

    // ---- out[d] = sum_s scores[s] * V[s][d]: key groups x 128 d per pass ----
    const uint kgN = max(1u, tgSize / 128u);
    const uint dl = tid % 128u, kg = tid / 128u;
    for (uint d0 = 0; d0 < uint(D); d0 += 128u) {
        const uint d = d0 + dl;
        float acc = 0.0f;
        if (d < uint(D))
            for (uint s = kg; s < uint(S); s += kgN) acc += scores[s] * float(vBase[s * uint(D) + d]);
        red[kg * 128u + dl] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg == 0 && d < uint(D)) {
            float t = 0.0f;
            for (uint j = 0; j < kgN; ++j) t += red[j * 128u + dl];
            out[d] = t * invSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
template [[host_name("attention_decode_all_heads_kernel")]] kernel decltype(attention_decode_all_heads_t<float>) attention_decode_all_heads_t<float>;
template [[host_name("attention_decode_all_heads_kernel_h")]] kernel decltype(attention_decode_all_heads_t<half>) attention_decode_all_heads_t<half>;
