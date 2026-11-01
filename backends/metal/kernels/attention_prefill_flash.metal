// ============================================================
// attention_prefill_flash_kernel / kv_cache_scatter_kernel
// ------------------------------------------------------------
// Causal prefill flash attention, the Metal twin of
// backends/vulkan/kernels/attention_prefill_flash.glsl (same tiling, same math):
//   o[t, h, :] = softmax_k(q[t, h, :] . K[hkv, k, :] * scale) . V[hkv, k, :]
//   over k <= pos + t (k < kvLen = pos + T), hkv = h / (H / HKV).
// q and o are position-major [T, H, D]; K/V are the resident cache
// [HKV, maxSeqLen, D] already holding positions 0 .. pos + T - 1.
// One threadgroup = 16 query rows of one head, 128 threads (8 per row); keys
// stream through threadgroup memory 16 at a time with an online softmax in
// base 2 (scale * log2(e) folded into the scores). D <= 128.
// ============================================================

#define FA_BR 16
#define FA_BC 16
#define FA_MAXD 128
#define FA_NCOL (FA_MAXD / 8)
#define FA_NEG (-1.0e30f)

struct FlashParams {
    uint T, H, HKV, D, pos, maxSeqLen;
    float scaleLog2;
};

kernel void attention_prefill_flash_kernel(
    device const float* q        [[buffer(0)]],
    device const float* kc       [[buffer(1)]],
    device const float* vc       [[buffer(2)]],
    device float* o              [[buffer(3)]],
    constant FlashParams& p      [[buffer(4)]],
    uint3 tg                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]])
{
    threadgroup float qs[FA_BR * FA_MAXD];
    threadgroup float ks[FA_BC * (FA_MAXD + 1)];
    threadgroup float vs[FA_BC * FA_MAXD];
    threadgroup float ss[FA_BR * (FA_BC + 1)];

    uint r = tid >> 3;
    uint sub = tid & 7u;
    uint row0 = tg.x * FA_BR;
    uint h = tg.y;
    uint D = p.D;
    uint hkv = min(h / max(1u, p.H / p.HKV), p.HKV - 1u);
    uint kvBase = hkv * p.maxSeqLen * D;
    uint kvLen = p.pos + p.T;
    uint t = row0 + r;
    uint rowPos = p.pos + t;

    for (uint i = tid; i < FA_BR * D; i += 128u) {
        uint rr = i / D;
        uint d = i - rr * D;
        uint tt = row0 + rr;
        qs[rr * FA_MAXD + d] = tt < p.T ? q[(tt * p.H + h) * D + d] : 0.0f;
    }

    float acc[FA_NCOL];
    for (uint c = 0u; c < FA_NCOL; ++c) acc[c] = 0.0f;
    float m = FA_NEG;
    float l = 0.0f;
    uint kEnd = min(kvLen, p.pos + row0 + FA_BR);

    for (uint kb = 0u; kb < kEnd; kb += FA_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < FA_BC * D; i += 128u) {
            uint j = i / D;
            uint d = i - j * D;
            uint key = kb + j;
            bool ok = key < kvLen;
            ks[j * (FA_MAXD + 1u) + d] = ok ? kc[kvBase + key * D + d] : 0.0f;
            vs[j * FA_MAXD + d] = ok ? vc[kvBase + key * D + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j0 = sub;
        uint j1 = sub + 8u;
        float d0 = 0.0f;
        float d1 = 0.0f;
        for (uint d = 0u; d < D; ++d) {
            float qv = qs[r * FA_MAXD + d];
            d0 += qv * ks[j0 * (FA_MAXD + 1u) + d];
            d1 += qv * ks[j1 * (FA_MAXD + 1u) + d];
        }
        uint key0 = kb + j0;
        uint key1 = kb + j1;
        float s0 = (key0 < kvLen && key0 <= rowPos) ? d0 * p.scaleLog2 : FA_NEG;
        float s1 = (key1 < kvLen && key1 <= rowPos) ? d1 * p.scaleLog2 : FA_NEG;
        ss[r * (FA_BC + 1u) + j0] = s0;
        ss[r * (FA_BC + 1u) + j1] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rowMax = FA_NEG;
        for (uint j = 0u; j < FA_BC; ++j) rowMax = max(rowMax, ss[r * (FA_BC + 1u) + j]);
        float mNew = max(m, rowMax);
        float corr = exp2(m - mNew);
        float rowSum = 0.0f;
        for (uint j = 0u; j < FA_BC; ++j) {
            float sv = ss[r * (FA_BC + 1u) + j];
            rowSum += sv > 0.5f * FA_NEG ? exp2(sv - mNew) : 0.0f;
        }
        l = l * corr + rowSum;
        m = mNew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss[r * (FA_BC + 1u) + j0] = s0 > 0.5f * FA_NEG ? exp2(s0 - mNew) : 0.0f;
        ss[r * (FA_BC + 1u) + j1] = s1 > 0.5f * FA_NEG ? exp2(s1 - mNew) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0u; c < FA_NCOL; ++c) {
            uint d = sub + 8u * c;
            if (d < D) {
                float a = acc[c] * corr;
                for (uint j = 0u; j < FA_BC; ++j) a += ss[r * (FA_BC + 1u) + j] * vs[j * FA_MAXD + d];
                acc[c] = a;
            }
        }
    }

    if (t < p.T) {
        float inv = 1.0f / max(l, 1.0e-30f);
        for (uint c = 0u; c < FA_NCOL; ++c) {
            uint d = sub + 8u * c;
            if (d < D) o[(t * p.H + h) * D + d] = acc[c] * inv;
        }
    }
}

struct KvScatterParams {
    uint T, HKV, D, pos, maxSeqLen;
};

// src [T, HKV, D] (the K or V projection) -> dst [HKV, maxSeqLen, D] at pos.
kernel void kv_cache_scatter_kernel(
    device const float* src      [[buffer(0)]],
    device float* dst            [[buffer(1)]],
    constant KvScatterParams& p  [[buffer(2)]],
    uint i                       [[thread_position_in_grid]])
{
    uint total = p.T * p.HKV * p.D;
    if (i >= total) return;
    uint d = i % p.D;
    uint th = i / p.D;
    uint h = th % p.HKV;
    uint t = th / p.HKV;
    dst[(h * p.maxSeqLen + p.pos + t) * p.D + d] = src[i];
}
