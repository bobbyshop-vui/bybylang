#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_control_flow_attributes : require

// Causal prefill flash attention on tensor cores (KHR_cooperative_matrix,
// FP16 MMA, FP32 accumulate). Same contract as attention_prefill_flash.glsl:
// q/o [T, H, D] position-major, K/V cache [HKV, maxSeqLen, D] already holding
// positions 0 .. pos + T - 1, causal over k <= pos + t, GQA hkv = h / (H / HKV).
//
// One workgroup = one subgroup = 16 query rows of one head. Per 16-key block:
//   S = Q K^T     D/16 MMAs (Q stays in registers as A matrices)
//   softmax       one lane per row, base 2 (scale * log2(e) folded in)
//   O += P V      D/16 MMAs into accumulators that never leave registers
// The output accumulators are rescaled only when a row max grows by more than
// RESCALE_LOG2 (2^8): until then P is taken against the stale max, its
// values stay <= 256 (exact in FP16) and l uses the same reference, so the
// result is unchanged (the lazy rescale of FlashAttention-4). A rescale goes
// through shared memory, so skipping it saves most of the softmax overhead.
// HEAD_DIM is a compile-time multiple of 16, at most 128.

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#define DT (HEAD_DIM / 16)
#define LDH (HEAD_DIM + 8)    // padded FP16 row: 16-byte aligned for coopMatLoad
#define LDP 24
#define NEG (-1.0e30)
#define RESCALE_LOG2 8.0

layout(local_size_x_id = 0) in;

layout(std430, binding = 0) readonly buffer BufQ { vec4 q4[]; };
layout(std430, binding = 1) readonly buffer BufK { vec4 k4[]; };
layout(std430, binding = 2) readonly buffer BufV { vec4 v4[]; };
layout(std430, binding = 3) writeonly buffer BufO { float o[]; };

layout(push_constant) uniform Params {
    uint T, H, HKV, D, pos, maxSeqLen;
    float scaleLog2;
} p;

shared float16_t qs[16 * LDH];
shared float16_t ks[16 * LDH];
shared float16_t vs[16 * LDH];
shared float16_t ps[16 * LDP];
shared float sTile[256];
shared float oStage[256];
shared float rowScale[16];

void main() {
    uint lane = gl_SubgroupInvocationID;
    uint nLanes = gl_SubgroupSize;
    uint row0 = gl_WorkGroupID.x * 16u;
    uint h = gl_WorkGroupID.y;
    uint hkv = min(h / max(1u, p.H / p.HKV), p.HKV - 1u);
    uint kvBase4 = hkv * p.maxSeqLen * (HEAD_DIM / 4);
    uint kvLen = p.pos + p.T;
    const uint vecsPerRow = HEAD_DIM / 4;

    for (uint i = lane; i < 16u * vecsPerRow; i += nLanes) {
        uint rr = i / vecsPerRow;
        uint c = i - rr * vecsPerRow;
        uint t = row0 + rr;
        vec4 v = t < p.T ? q4[(t * p.H + h) * vecsPerRow + c] : vec4(0.0);
        uint off = rr * LDH + c * 4u;
        qs[off] = float16_t(v.x);
        qs[off + 1u] = float16_t(v.y);
        qs[off + 2u] = float16_t(v.z);
        qs[off + 3u] = float16_t(v.w);
    }
    barrier();

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> qa[DT];
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[DT];
    [[unroll]] for (uint dt = 0u; dt < DT; ++dt) {
        coopMatLoad(qa[dt], qs, dt * 16u, LDH, gl_CooperativeMatrixLayoutRowMajor);
        acc[dt] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    }

    // Lane r < 16 keeps the running max (reference) and sum of row r.
    float m = NEG;
    float l = 0.0;
    uint rowPos = p.pos + row0 + lane;
    bool first = true;
    uint kEnd = min(kvLen, p.pos + row0 + 16u);

    for (uint kb = 0u; kb < kEnd; kb += 16u) {
        barrier();   // previous block done with ks/vs/ps
        for (uint i = lane; i < 16u * vecsPerRow; i += nLanes) {
            uint j = i / vecsPerRow;
            uint c = i - j * vecsPerRow;
            uint key = kb + j;
            vec4 kv = vec4(0.0);
            vec4 vv = vec4(0.0);
            if (key < kvLen) {
                kv = k4[kvBase4 + key * vecsPerRow + c];
                vv = v4[kvBase4 + key * vecsPerRow + c];
            }
            uint off = j * LDH + c * 4u;
            ks[off] = float16_t(kv.x);
            ks[off + 1u] = float16_t(kv.y);
            ks[off + 2u] = float16_t(kv.z);
            ks[off + 3u] = float16_t(kv.w);
            vs[off] = float16_t(vv.x);
            vs[off + 1u] = float16_t(vv.y);
            vs[off + 2u] = float16_t(vv.z);
            vs[off + 3u] = float16_t(vv.w);
        }
        barrier();

        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> s =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for (uint dt = 0u; dt < DT; ++dt) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> kbm;
            // B[d][j] = K[kb + j][d]: column j is key j, contiguous along d.
            coopMatLoad(kbm, ks, dt * 16u, LDH, gl_CooperativeMatrixLayoutColumnMajor);
            s = coopMatMulAdd(qa[dt], kbm, s);
        }
        coopMatStore(s, sTile, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        barrier();

        float corr = 1.0;
        if (lane < 16u) {
            float sv[16];
            float rowMax = NEG;
            [[unroll]] for (uint j = 0u; j < 16u; ++j) {
                uint key = kb + j;
                float x = (key < kvLen && key <= rowPos) ? sTile[lane * 16u + j] * p.scaleLog2 : NEG;
                sv[j] = x;
                rowMax = max(rowMax, x);
            }
            if (rowMax > m + RESCALE_LOG2) {
                corr = exp2(m - rowMax);
                m = rowMax;
            }
            float rowSum = 0.0;
            [[unroll]] for (uint j = 0u; j < 16u; ++j) {
                float e = sv[j] > 0.5 * NEG ? exp2(sv[j] - m) : 0.0;
                rowSum += e;
                ps[lane * LDP + j] = float16_t(e);
            }
            l = l * corr + rowSum;
            rowScale[lane] = corr;
        }
        bool rescale = subgroupAny(corr != 1.0) && !first;
        first = false;
        barrier();

        if (rescale) {
            [[unroll]] for (uint dt = 0u; dt < DT; ++dt) {
                coopMatStore(acc[dt], oStage, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
                barrier();
                for (uint e = lane; e < 256u; e += nLanes) oStage[e] *= rowScale[e >> 4];
                barrier();
                coopMatLoad(acc[dt], oStage, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
                barrier();
            }
        }

        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> pa;
        coopMatLoad(pa, ps, 0, LDP, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint dt = 0u; dt < DT; ++dt) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> vb;
            // B[j][d] = V[kb + j][d]: row j is key j.
            coopMatLoad(vb, vs, dt * 16u, LDH, gl_CooperativeMatrixLayoutRowMajor);
            acc[dt] = coopMatMulAdd(pa, vb, acc[dt]);
        }
    }

    barrier();
    if (lane < 16u) rowScale[lane] = 1.0 / max(l, 1.0e-30);
    barrier();
    [[unroll]] for (uint dt = 0u; dt < DT; ++dt) {
        coopMatStore(acc[dt], oStage, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint e = lane; e < 256u; e += nLanes) {
            uint rr = e >> 4;
            uint t = row0 + rr;
            if (t < p.T) o[(t * p.H + h) * HEAD_DIM + dt * 16u + (e & 15u)] = oStage[e] * rowScale[rr];
        }
        barrier();
    }
}
