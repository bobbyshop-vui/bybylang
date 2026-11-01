#version 450
// Causal prefill flash attention (scalar FP32), GQA, one pass over the KV cache.
//
//   o[t, h, :] = softmax_k(q[t, h, :] . K[hkv, k, :] * scale) . V[hkv, k, :]
//   over k <= pos + t (k < kvLen = pos + T), hkv = h / (H / HKV).
//
// q and o are position-major [T, H, D] (the layout of the Q projection and
// the input of oProj); K/V are the resident cache [HKV, maxSeqLen, D] that
// already holds positions 0 .. pos + T - 1, so a continued prompt (pos > 0)
// attends to the earlier turns too.
//
// One workgroup = BR query rows of one head; it streams BC keys at a time
// through shared memory with an online softmax in base 2 (scale * log2(e)
// folded into the scores), so nothing of size T x T is ever stored.
// 128 threads: 8 per query row. Thread (r, sub) scores keys sub and sub + 8 of
// each block and owns output columns sub, sub + 8, ... of row r.
// Host contract: D <= MAXD, local size 128, ~27 KiB of shared memory.

#define BR 16
#define BC 16
#define MAXD 128
#define NCOL (MAXD / 8)
#define NEG (-1.0e30)

layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer BufQ { float q[]; };
layout(std430, binding = 1) readonly buffer BufK { float kc[]; };
layout(std430, binding = 2) readonly buffer BufV { float vc[]; };
layout(std430, binding = 3) writeonly buffer BufO { float o[]; };

layout(push_constant) uniform Params {
    uint T, H, HKV, D, pos, maxSeqLen;
    float scaleLog2;   // softmax scale * log2(e)
} p;

shared float qs[BR * MAXD];
shared float ks[BC * (MAXD + 1)];
shared float vs[BC * MAXD];
shared float ss[BR * (BC + 1)];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint r = tid >> 3;
    uint sub = tid & 7u;
    uint row0 = gl_WorkGroupID.x * BR;
    uint h = gl_WorkGroupID.y;
    uint D = p.D;
    uint hkv = min(h / max(1u, p.H / p.HKV), p.HKV - 1u);
    uint kvBase = hkv * p.maxSeqLen * D;
    uint kvLen = p.pos + p.T;
    uint t = row0 + r;
    uint rowPos = p.pos + t;

    for (uint i = tid; i < BR * D; i += 128u) {
        uint rr = i / D;
        uint d = i - rr * D;
        uint tt = row0 + rr;
        qs[rr * MAXD + d] = tt < p.T ? q[(tt * p.H + h) * D + d] : 0.0;
    }

    float acc[NCOL];
    for (uint c = 0u; c < NCOL; ++c) acc[c] = 0.0;
    float m = NEG;
    float l = 0.0;
    // The last row of the block sees keys up to pos + row0 + BR - 1.
    uint kEnd = min(kvLen, p.pos + row0 + BR);

    for (uint kb = 0u; kb < kEnd; kb += BC) {
        barrier();   // previous block done with ks/vs/ss (and qs written)
        for (uint i = tid; i < BC * D; i += 128u) {
            uint j = i / D;
            uint d = i - j * D;
            uint key = kb + j;
            bool ok = key < kvLen;
            ks[j * (MAXD + 1u) + d] = ok ? kc[kvBase + key * D + d] : 0.0;
            vs[j * MAXD + d] = ok ? vc[kvBase + key * D + d] : 0.0;
        }
        barrier();

        float s0 = NEG;
        float s1 = NEG;
        {
            uint j0 = sub;
            uint j1 = sub + 8u;
            float d0 = 0.0;
            float d1 = 0.0;
            for (uint d = 0u; d < D; ++d) {
                float qv = qs[r * MAXD + d];
                d0 += qv * ks[j0 * (MAXD + 1u) + d];
                d1 += qv * ks[j1 * (MAXD + 1u) + d];
            }
            uint key0 = kb + j0;
            uint key1 = kb + j1;
            if (key0 < kvLen && key0 <= rowPos) s0 = d0 * p.scaleLog2;
            if (key1 < kvLen && key1 <= rowPos) s1 = d1 * p.scaleLog2;
            ss[r * (BC + 1u) + j0] = s0;
            ss[r * (BC + 1u) + j1] = s1;
        }
        barrier();

        float rowMax = NEG;
        for (uint j = 0u; j < BC; ++j) rowMax = max(rowMax, ss[r * (BC + 1u) + j]);
        float mNew = max(m, rowMax);
        float corr = exp2(m - mNew);
        float rowSum = 0.0;
        for (uint j = 0u; j < BC; ++j) {
            float sv = ss[r * (BC + 1u) + j];
            rowSum += sv > 0.5 * NEG ? exp2(sv - mNew) : 0.0;
        }
        l = l * corr + rowSum;
        m = mNew;
        barrier();   // every thread of the row has read the scores
        ss[r * (BC + 1u) + sub] = s0 > 0.5 * NEG ? exp2(s0 - mNew) : 0.0;
        ss[r * (BC + 1u) + sub + 8u] = s1 > 0.5 * NEG ? exp2(s1 - mNew) : 0.0;
        barrier();

        for (uint c = 0u; c < NCOL; ++c) {
            uint d = sub + 8u * c;
            if (d < D) {
                float a = acc[c] * corr;
                for (uint j = 0u; j < BC; ++j) a += ss[r * (BC + 1u) + j] * vs[j * MAXD + d];
                acc[c] = a;
            }
        }
    }

    if (t < p.T) {
        float inv = 1.0 / max(l, 1.0e-30);
        for (uint c = 0u; c < NCOL; ++c) {
            uint d = sub + 8u * c;
            if (d < D) o[(t * p.H + h) * D + d] = acc[c] * inv;
        }
    }
}
