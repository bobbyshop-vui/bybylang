// ============================================================
// attention_prefill_flash_kernel / kv_cache_scatter_kernel
// ------------------------------------------------------------
// Causal prefill flash attention using FP16 operands and FP32 softmax/accumulation.
// This is the FP16 sibling of
// backends/vulkan/kernels/attention_prefill_flash.glsl (same tiling, same math):
//   o[t, h, :] = softmax_k(q[t, h, :] . K[hkv, k, :] * scale) . V[hkv, k, :]
//   over k <= pos + t (k < kvLen = pos + T), hkv = h / (H / HKV).
// q and o are position-major [T, H, D]; K/V are the resident cache
// [HKV, maxSeqLen, D] already holding positions 0 .. pos + T - 1.
// One threadgroup = 16 query rows of one head, 128 threads (8 per row); keys
// stream through threadgroup memory 16 at a time with an online softmax in
// base 2 (scale * log2(e) folded into the scores). D <= 128.
// ============================================================

#define FA16_BR 16
#define FA16_BC 16
#define FA16_MAXD 128
#define FA16_NCOL (FA16_MAXD / 8)
#define FA16_NEG (-1.0e30f)

struct FlashFp16Params {
    uint T, H, HKV, D, pos, maxSeqLen;
    float scaleLog2;
};

kernel void attention_prefill_flash_fp16_kernel(
    device const float* q        [[buffer(0)]],
    device const float* kc       [[buffer(1)]],
    device const float* vc       [[buffer(2)]],
    device float* o              [[buffer(3)]],
    constant FlashFp16Params& p      [[buffer(4)]],
    uint3 tg                     [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]])
{
    threadgroup half qs[FA16_BR * FA16_MAXD];
    threadgroup half ks[FA16_BC * (FA16_MAXD + 1)];
    threadgroup half vs[FA16_BC * FA16_MAXD];
    threadgroup float ss[FA16_BR * (FA16_BC + 1)];

    uint r = tid >> 3;
    uint sub = tid & 7u;
    uint row0 = tg.x * FA16_BR;
    uint h = tg.y;
    uint D = p.D;
    uint hkv = min(h / max(1u, p.H / p.HKV), p.HKV - 1u);
    uint kvBase = hkv * p.maxSeqLen * D;
    uint kvLen = p.pos + p.T;
    uint t = row0 + r;
    uint rowPos = p.pos + t;

    for (uint i = tid; i < FA16_BR * D; i += 128u) {
        uint rr = i / D;
        uint d = i - rr * D;
        uint tt = row0 + rr;
        qs[rr * FA16_MAXD + d] = half(tt < p.T ? q[(tt * p.H + h) * D + d] : 0.0f);
    }

    float acc[FA16_NCOL];
    for (uint c = 0u; c < FA16_NCOL; ++c) acc[c] = 0.0f;
    float m = FA16_NEG;
    float l = 0.0f;
    uint kEnd = min(kvLen, p.pos + row0 + FA16_BR);

    for (uint kb = 0u; kb < kEnd; kb += FA16_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < FA16_BC * D; i += 128u) {
            uint j = i / D;
            uint d = i - j * D;
            uint key = kb + j;
            bool ok = key < kvLen;
            ks[j * (FA16_MAXD + 1u) + d] = half(ok ? kc[kvBase + key * D + d] : 0.0f);
            vs[j * FA16_MAXD + d] = half(ok ? vc[kvBase + key * D + d] : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j0 = sub;
        uint j1 = sub + 8u;
        float d0 = 0.0f;
        float d1 = 0.0f;
        for (uint d = 0u; d < D; ++d) {
            half qv = qs[r * FA16_MAXD + d];
            d0 += float(qv * ks[j0 * (FA16_MAXD + 1u) + d]);
            d1 += float(qv * ks[j1 * (FA16_MAXD + 1u) + d]);
        }
        uint key0 = kb + j0;
        uint key1 = kb + j1;
        float s0 = (key0 < kvLen && key0 <= rowPos) ? d0 * p.scaleLog2 : FA16_NEG;
        float s1 = (key1 < kvLen && key1 <= rowPos) ? d1 * p.scaleLog2 : FA16_NEG;
        ss[r * (FA16_BC + 1u) + j0] = s0;
        ss[r * (FA16_BC + 1u) + j1] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rowMax = FA16_NEG;
        for (uint j = 0u; j < FA16_BC; ++j) rowMax = max(rowMax, ss[r * (FA16_BC + 1u) + j]);
        float mNew = max(m, rowMax);
        float corr = exp2(m - mNew);
        float rowSum = 0.0f;
        for (uint j = 0u; j < FA16_BC; ++j) {
            float sv = ss[r * (FA16_BC + 1u) + j];
            rowSum += sv > 0.5f * FA16_NEG ? exp2(sv - mNew) : 0.0f;
        }
        l = l * corr + rowSum;
        m = mNew;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ss[r * (FA16_BC + 1u) + j0] = s0 > 0.5f * FA16_NEG ? exp2(s0 - mNew) : 0.0f;
        ss[r * (FA16_BC + 1u) + j1] = s1 > 0.5f * FA16_NEG ? exp2(s1 - mNew) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0u; c < FA16_NCOL; ++c) {
            uint d = sub + 8u * c;
            if (d < D) {
                float a = acc[c] * corr;
                for (uint j = 0u; j < FA16_BC; ++j)
                    a += float(half(ss[r * (FA16_BC + 1u) + j]) * vs[j * FA16_MAXD + d]);
                acc[c] = a;
            }
        }
    }

    if (t < p.T) {
        float inv = 1.0f / max(l, 1.0e-30f);
        for (uint c = 0u; c < FA16_NCOL; ++c) {
            uint d = sub + 8u * c;
            if (d < D) o[(t * p.H + h) * D + d] = acc[c] * inv;
        }
    }
}

