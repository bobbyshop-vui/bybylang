#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_control_flow_attributes : require

// FIR custom micro-kernel for FP4-KM prefill (Codestral Mamba-2 weights). It is
// compiled by the micro-VM JIT and dispatched as a normal FIR micro node; it does
// not call the native FP4 session GEMM. Its body follows the native tiled kernel
// (matmul_fp4km_tiled.glsl, format 1) and the Q4 micro-kernel's tiling
// (fir_q4_coop_micro.glsl): one 128-invocation workgroup computes BM rows x 64
// columns (BM = q4CoopRowTile(rows, N): 16, 32 or 64, chosen per dispatch). Each
// 64-deep K step decodes the 64-column weight tile once into shared FP16 (one
// invocation per column and 32-value group: one uvec4 + one scale byte) and
// reuses it for every live 16-row tile; accumulation is FP32. Each subgroup owns
// a 16-column fragment (FRAGS fragments when the workgroup has fewer than four
// subgroups) with one accumulator per row tile.
// Slots: memF32 = output [M, N]; memU8 and lut = the same FP4-KM blob: nibble
// bytes [N, K/2] (byte j of a 32-value group holds element j in the low nibble
// and element j+16 in the high one), scale bytes [N, K/32], padding to a word,
// then 272 floats at LUT_BASE: 16 FP4 values and 256 scale values
// (weight = lut[nibble] * lut[16 + scale byte]); input = x [M, K].
layout(local_size_x_id = 0) in;
layout(std430, set=0, binding=0) buffer MemF32Buf { float memF32[]; };
layout(std430, set=0, binding=1) readonly buffer MemU8Buf { uvec4 memU8[]; };
layout(std430, set=0, binding=2) readonly buffer LutBuf { float lut[]; };
layout(std430, set=0, binding=3) readonly buffer InputBuf { vec4 in4[]; };

const uint FP4_M = @M@u;
const uint FP4_K = @K@u;
const uint FP4_N = @N@u;
const uint FP4_N_GROUPS = @N_GROUPS@u;      // K / 32
const uint BM = @BM@u;
const uint FRAGS = @FRAGS@u;               // 16-column fragments per subgroup
const uint BN = 64u;
const uint BK = 64u;
const uint LDS = BK + 8u;                  // padded row (halves): 144 bytes, 16-byte aligned
const uint ROW_TILES = BM / 16u;
const uint N_TILES = (FP4_N + BN - 1u) / BN;
const uint K_VEC = FP4_K / 4u;
const uint SCALE_BASE = FP4_N * (FP4_K / 2u);                       // first scale byte
const uint LUT_BASE = (SCALE_BASE + FP4_N * FP4_N_GROUPS + 3u) / 4u; // first decode-table float

shared float16_t aTile[BM * LDS];
shared float16_t bTile[BN * LDS];
shared float cStage[4u * 256u];
shared float lutS[16];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint threads = gl_WorkGroupSize.x;
    uint sg = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint subgroupCount = gl_NumSubgroups;
    uint row0 = (gl_WorkGroupID.x / N_TILES) * BM;
    uint col0 = (gl_WorkGroupID.x % N_TILES) * BN;
    // Uniform per workgroup. A graph lifted with a shorter row tile launches
    // workgroups past the last row; they have nothing to do.
    if (row0 >= FP4_M) return;
    uint liveTiles = min(ROW_TILES, (FP4_M - row0 + 15u) / 16u);
    uint bCol = lid >> 1;
    uint bHalf = lid & 1u;
    uint gCol = col0 + bCol;

    if (lid < 16u) lutS[lid] = lut[LUT_BASE + lid];
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[FRAGS * ROW_TILES];
    [[unroll]] for (uint i = 0u; i < FRAGS * ROW_TILES; ++i)
        acc[i] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    barrier();

    for (uint base = 0u; base < FP4_K; base += BK) {
        // Activations: only live row tiles, 16 vec4 per row (rows past M are zero).
        for (uint i = lid; i < liveTiles * 16u * (BK / 4u); i += threads) {
            uint r = i >> 4;
            uint q = i & 15u;
            uint row = row0 + r;
            vec4 v = row < FP4_M ? in4[row * K_VEC + (base >> 2) + q] : vec4(0.0);
            uint o = r * LDS + q * 4u;
            aTile[o] = float16_t(v.x);
            aTile[o + 1u] = float16_t(v.y);
            aTile[o + 2u] = float16_t(v.z);
            aTile[o + 3u] = float16_t(v.w);
        }

        // Weights: invocation (column, half) decodes one 32-value group = one uvec4.
        if (lid < 2u * BN) {
            uint g = (base >> 5) + bHalf;
            uvec4 packed = uvec4(0u);
            float scale = 0.0;
            if (gCol < FP4_N) {
                packed = memU8[gCol * FP4_N_GROUPS + g];
                uint sb = SCALE_BASE + gCol * FP4_N_GROUPS + g;
                uint word = memU8[sb >> 4][(sb >> 2) & 3u];
                scale = lut[LUT_BASE + 16u + ((word >> ((sb & 3u) * 8u)) & 255u)];
            }
            uint ob = bCol * LDS + bHalf * 32u;
            [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                uint word = packed[j];
                [[unroll]] for (uint b = 0u; b < 4u; ++b) {
                    uint byteV = (word >> (8u * b)) & 255u;
                    bTile[ob + 4u * j + b] = float16_t(lutS[byteV & 15u] * scale);
                    bTile[ob + 16u + 4u * j + b] = float16_t(lutS[byteV >> 4] * scale);
                }
            }
        }
        barrier();

        [[unroll]] for (uint kk = 0u; kk < BK; kk += 16u) {
            [[unroll]] for (uint f = 0u; f < FRAGS; ++f) {
                uint colFragment = sg + f * subgroupCount;
                if (colFragment < 4u) {
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bm;
                    coopMatLoad(bm, bTile, colFragment * 16u * LDS + kk, LDS,
                        gl_CooperativeMatrixLayoutColumnMajor);
                    [[unroll]] for (uint r = 0u; r < ROW_TILES; ++r) {
                        if (r < liveTiles) {
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
                            coopMatLoad(am, aTile, r * 16u * LDS + kk, LDS,
                                gl_CooperativeMatrixLayoutRowMajor);
                            acc[f * ROW_TILES + r] = coopMatMulAdd(am, bm, acc[f * ROW_TILES + r]);
                        }
                    }
                }
            }
        }
        barrier();
    }

    // Each subgroup stages its own 16x16 tile, then writes the in-bounds elements.
    [[unroll]] for (uint f = 0u; f < FRAGS; ++f) {
        uint colFragment = sg + f * subgroupCount;
        if (colFragment < 4u) {
            [[unroll]] for (uint r = 0u; r < ROW_TILES; ++r) {
                if (r < liveTiles) {
                    coopMatStore(acc[f * ROW_TILES + r], cStage, colFragment * 256u, 16,
                        gl_CooperativeMatrixLayoutRowMajor);
                    subgroupMemoryBarrierShared();
                    subgroupBarrier();
                    for (uint e = lane; e < 256u; e += gl_SubgroupSize) {
                        uint row = row0 + r * 16u + (e >> 4);
                        uint col = col0 + colFragment * 16u + (e & 15u);
                        if (row < FP4_M && col < FP4_N)
                            memF32[row * FP4_N + col] = cStage[colFragment * 256u + e];
                    }
                    subgroupBarrier();
                }
            }
        }
    }
}
