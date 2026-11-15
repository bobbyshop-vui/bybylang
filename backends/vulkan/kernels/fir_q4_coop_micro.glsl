#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

// FIR custom micro-kernel for asymmetric Q4 prefill. This is compiled by the
// micro-VM JIT and dispatched as a normal FIR micro node. It does not call the
// native Q4/session GEMM path. One 128-thread workgroup computes 16 rows x 64
// columns, using four 16x16 cooperative matrices per K tile. The 16x64 tile
// supplies enough independent workgroups on T4 without a separate split-K
// scratch/reduction dispatch.
layout(local_size_x_id = 0) in;
layout(std430, set=0, binding=0) buffer MemF32Buf { float memF32[]; };
layout(std430, set=0, binding=1) readonly buffer MemU8Buf { uint memU8[]; };
layout(std430, set=0, binding=2) readonly buffer LutBuf { float lut[]; };
layout(std430, set=0, binding=3) readonly buffer InputBuf { float inF32[]; };

const uint Q4_M = @M@u;
const uint Q4_K = @K@u;
const uint Q4_N = @N@u;
const uint Q4_GROUP_SIZE = @GROUP_SIZE@u;
const uint Q4_N_GROUPS = @N_GROUPS@u;
const uint BM = 16u;
const uint BN = 64u;
const uint BK = 64u;
const uint LDS = 72u;
const uint N_TILES = (Q4_N + BN - 1u) / BN;
const uint SCALE_BASE = (Q4_N * (Q4_K / 2u) + 3u) / 4u;
const uint ZERO_BASE = SCALE_BASE + Q4_N * Q4_N_GROUPS;

shared float16_t aTile[BM * LDS];
shared float16_t bTile[BN * LDS];
shared float cTile[4u * 256u];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint subgroup = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint subgroupCount = gl_WorkGroupSize.x / gl_SubgroupSize;
    uint row0 = (gl_WorkGroupID.x / N_TILES) * BM;
    uint col0 = (gl_WorkGroupID.x % N_TILES) * BN;
    uint bCol = lid >> 1u;
    uint bHalf = lid & 1u;

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc0 =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc1 =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint base = 0u; base < Q4_K; base += BK) {
        // Stage one row tile of activations as FP16.
        for (uint i = lid; i < BM * (BK / 4u); i += gl_WorkGroupSize.x) {
            uint row = row0 + (i >> 4u);
            uint q = i & 15u;
            vec4 v = vec4(0.0);
            if (row < Q4_M) {
                uint x = row * Q4_K + base + q * 4u;
                v = vec4(inF32[x], inF32[x + 1u], inF32[x + 2u], inF32[x + 3u]);
            }
            uint o = (i >> 4u) * LDS + q * 4u;
            aTile[o] = float16_t(v.x);
            aTile[o + 1u] = float16_t(v.y);
            aTile[o + 2u] = float16_t(v.z);
            aTile[o + 3u] = float16_t(v.w);
        }

        // Dequantize one 64-column x 64-K Q4 tile once into shared FP16.
        if (lid < 2u * BN) {
            uint col = col0 + bCol;
            uint kb = base + bHalf * 32u;
            uint wordBase = (col * (Q4_K / 2u) + (kb >> 1u)) >> 2u;
            uint packed[4];
            for (uint j = 0u; j < 4u; ++j)
                packed[j] = col < Q4_N ? memU8[wordBase + j] : 0u;
            uint bOffset = bCol * LDS + bHalf * 32u;
            uint lastGroup = 0xffffffffu;
            float scale = 0.0;
            float zero = 0.0;
            for (uint j = 0u; j < 4u; ++j) {
                uint word = packed[j];
                for (uint b = 0u; b < 4u; ++b) {
                    uint k = kb + 8u * j + 2u * b;
                    uint group = col * Q4_N_GROUPS + k / Q4_GROUP_SIZE;
                    if (col < Q4_N && group != lastGroup) {
                        scale = lut[SCALE_BASE + group];
                        zero = lut[ZERO_BASE + group];
                        lastGroup = group;
                    }
                    uint byteValue = (word >> (8u * b)) & 255u;
                    bTile[bOffset + 8u * j + 2u * b] =
                        float16_t((float(byteValue & 15u) - zero) * scale);
                    bTile[bOffset + 8u * j + 2u * b + 1u] =
                        float16_t((float(byteValue >> 4u) - zero) * scale);
                }
            }
        }
        barrier();

        // A 128-thread workgroup has four subgroups on T4 and two on wave64
        // devices. Each subgroup computes one or two neighboring 16-column
        // fragments, keeping this custom micro-kernel valid for either width.
        for (uint kk = 0u; kk < BK; kk += 16u) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
            coopMatLoad(am, aTile, kk, LDS, gl_CooperativeMatrixLayoutRowMajor);
            for (uint fragment = 0u; fragment < 2u; ++fragment) {
                uint colFragment = subgroup + fragment * subgroupCount;
                if (colFragment < 4u) {
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bm;
                    coopMatLoad(bm, bTile, colFragment * 16u * LDS + kk, LDS,
                        gl_CooperativeMatrixLayoutColumnMajor);
                    if (fragment == 0u) acc0 = coopMatMulAdd(am, bm, acc0);
                    else acc1 = coopMatMulAdd(am, bm, acc1);
                }
            }
        }
        barrier();
    }

    for (uint fragment = 0u; fragment < 2u; ++fragment) {
        uint colFragment = subgroup + fragment * subgroupCount;
        if (colFragment < 4u) {
            if (fragment == 0u)
                coopMatStore(acc0, cTile, colFragment * 256u, 16,
                    gl_CooperativeMatrixLayoutRowMajor);
            else
                coopMatStore(acc1, cTile, colFragment * 256u, 16,
                    gl_CooperativeMatrixLayoutRowMajor);
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            for (uint e = lane; e < 256u; e += gl_SubgroupSize) {
                uint row = row0 + (e >> 4u);
                uint col = col0 + colFragment * 16u + (e & 15u);
                if (row < Q4_M && col < Q4_N)
                    memF32[row * Q4_N + col] = cTile[colFragment * 256u + e];
            }
            subgroupBarrier();
        }
    }
}
