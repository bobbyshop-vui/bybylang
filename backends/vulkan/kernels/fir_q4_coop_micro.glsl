#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

// FIR custom micro-kernel for asymmetric Q4 prefill. This is compiled by the
// micro-VM JIT and dispatched as a normal FIR micro node. It does not call the
// native Q4/session GEMM path. One 128-thread workgroup computes BM rows x 64
// columns (BM = q4CoopRowTile(M, N): 16, 32 or 64). Each K step dequantizes the
// 64-column Q4 weight tile once into shared FP16 and reuses it for all BM rows,
// so a 64-row prefill block reads and dequantizes every weight once instead of
// once per 16 rows. Each subgroup owns 16-column fragments and keeps one
// accumulator per 16-row tile.
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
const uint BM = @BM@u;
const uint BN = 64u;
const uint BK = 64u;
const uint LDS = 72u;
const uint ROW_TILES = BM / 16u;
const uint N_TILES = (Q4_N + BN - 1u) / BN;
const uint ROW_WORDS = Q4_K / 8u;          // packed 32-bit words per weight row
const uint SCALE_BASE = (Q4_N * (Q4_K / 2u) + 3u) / 4u;
const uint ZERO_BASE = SCALE_BASE + Q4_N * Q4_N_GROUPS;
// A packed word holds 8 consecutive K values; they share one quantization group
// whenever the group size is a multiple of 8.
const bool WORD_GROUP = (Q4_GROUP_SIZE % 8u) == 0u;

shared float16_t aTile[BM * LDS];
shared float16_t bTile[BN * LDS];
shared float cTile[4u * 256u];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint threads = gl_WorkGroupSize.x;
    uint subgroup = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint subgroupCount = gl_WorkGroupSize.x / gl_SubgroupSize;
    uint row0 = (gl_WorkGroupID.x / N_TILES) * BM;
    uint col0 = (gl_WorkGroupID.x % N_TILES) * BN;
    // Uniform per workgroup. A graph lifted with a shorter row tile launches
    // workgroups past the last row; they have nothing to do.
    if (row0 >= Q4_M) return;
    uint liveTiles = min(ROW_TILES, (Q4_M - row0 + 15u) / 16u);

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[2u * ROW_TILES];
    for (uint i = 0u; i < 2u * ROW_TILES; ++i)
        acc[i] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint base = 0u; base < Q4_K; base += BK) {
        // Stage the live row tiles of activations as FP16 (rows past M are zero).
        for (uint i = lid; i < liveTiles * 16u * (BK / 4u); i += threads) {
            uint r = i >> 4u;
            uint q = i & 15u;
            uint row = row0 + r;
            vec4 v = vec4(0.0);
            if (row < Q4_M) {
                uint x = row * Q4_K + base + q * 4u;
                v = vec4(inF32[x], inF32[x + 1u], inF32[x + 2u], inF32[x + 3u]);
            }
            uint o = r * LDS + q * 4u;
            aTile[o] = float16_t(v.x);
            aTile[o + 1u] = float16_t(v.y);
            aTile[o + 2u] = float16_t(v.z);
            aTile[o + 3u] = float16_t(v.w);
        }

        // Dequantize the 64-column x 64-K Q4 tile once: one packed word
        // (8 K values of one column) per step, neighbouring lanes on the same row.
        for (uint i = lid; i < BN * (BK / 8u); i += threads) {
            uint c = i >> 3u;
            uint w = i & 7u;
            uint col = col0 + c;
            uint k = base + w * 8u;
            uint o = c * LDS + w * 8u;
            if (col >= Q4_N) {
                for (uint j = 0u; j < 8u; ++j) bTile[o + j] = float16_t(0.0);
                continue;
            }
            uint word = memU8[col * ROW_WORDS + (k >> 3u)];
            uint group = col * Q4_N_GROUPS + k / Q4_GROUP_SIZE;
            float scale = lut[SCALE_BASE + group];
            float zero = lut[ZERO_BASE + group];
            for (uint b = 0u; b < 4u; ++b) {
                if (!WORD_GROUP && b > 0u) {
                    uint pairGroup = col * Q4_N_GROUPS + (k + 2u * b) / Q4_GROUP_SIZE;
                    if (pairGroup != group) {
                        group = pairGroup;
                        scale = lut[SCALE_BASE + group];
                        zero = lut[ZERO_BASE + group];
                    }
                }
                uint byteValue = (word >> (8u * b)) & 255u;
                bTile[o + 2u * b] = float16_t((float(byteValue & 15u) - zero) * scale);
                bTile[o + 2u * b + 1u] = float16_t((float(byteValue >> 4u) - zero) * scale);
            }
        }
        barrier();

        // Four subgroups on T4 own one 16-column fragment each; two wave64
        // subgroups own two. Every fragment is multiplied with each live row tile.
        for (uint kk = 0u; kk < BK; kk += 16u) {
            for (uint fragment = 0u; fragment < 2u; ++fragment) {
                uint colFragment = subgroup + fragment * subgroupCount;
                if (colFragment < 4u) {
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bm;
                    coopMatLoad(bm, bTile, colFragment * 16u * LDS + kk, LDS,
                        gl_CooperativeMatrixLayoutColumnMajor);
                    for (uint rt = 0u; rt < ROW_TILES; ++rt) {
                        if (rt < liveTiles) {
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
                            coopMatLoad(am, aTile, rt * 16u * LDS + kk, LDS,
                                gl_CooperativeMatrixLayoutRowMajor);
                            acc[fragment * ROW_TILES + rt] =
                                coopMatMulAdd(am, bm, acc[fragment * ROW_TILES + rt]);
                        }
                    }
                }
            }
        }
        barrier();
    }

    for (uint fragment = 0u; fragment < 2u; ++fragment) {
        uint colFragment = subgroup + fragment * subgroupCount;
        if (colFragment < 4u) {
            for (uint rt = 0u; rt < ROW_TILES; ++rt) {
                if (rt < liveTiles) {
                    coopMatStore(acc[fragment * ROW_TILES + rt], cTile, colFragment * 256u, 16,
                        gl_CooperativeMatrixLayoutRowMajor);
                    subgroupMemoryBarrierShared();
                    subgroupBarrier();
                    for (uint e = lane; e < 256u; e += gl_SubgroupSize) {
                        uint row = row0 + rt * 16u + (e >> 4u);
                        uint col = col0 + colFragment * 16u + (e & 15u);
                        if (row < Q4_M && col < Q4_N)
                            memF32[row * Q4_N + col] = cTile[colFragment * 256u + e];
                    }
                    subgroupBarrier();
                }
            }
        }
    }
}
