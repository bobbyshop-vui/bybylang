#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_control_flow_attributes : require

// Prefill GEMM y[M,N] = x[M,K] * W[N,K]^T, FP16 MMA, FP32 accumulate.
// WEIGHT_FORMAT 1 = packed FP4-KM group32, 0 = FP32 rows (Linear++ sidecar).
// One workgroup = 4 subgroups of 32 = 64 output columns x 64 rows. Each 64-deep
// K step decodes the packed weights once into shared FP16 (one thread per
// column/group: one uvec4 + one scale) and reuses them for every active 16-row
// tile. matmul_coop.glsl instead re-decodes per 16x16 tile and per 16-deep K
// step, so a 17-token prompt decoded every weight twice. Row tiles at or past M
// skip both their activation loads and their MMAs.
// Host contract: local size 128, subgroup size 32, K % 64 == 0, x and y at
// buffer offset 0 with row strides K and N.
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Input { vec4 x4[]; };
#ifndef WEIGHT_FORMAT
#define WEIGHT_FORMAT 1
#endif
#if WEIGHT_FORMAT == 0
layout(std430, binding = 1) readonly buffer Weight { vec4 wf4[]; };
#else
layout(std430, binding = 1) readonly buffer Weight { uvec4 w4[]; };
layout(std430, binding = 1) readonly buffer WeightWords { uint w[]; };
#endif
layout(std430, binding = 2) writeonly buffer Output { float y[]; };
layout(std430, binding = 3) readonly buffer Lut { float lut[]; };
layout(push_constant) uniform Params { uint M, K, N, groupSize, groupsPerRow; } p;

const uint BM = 64;
const uint BN = 64;
const uint BK = 64;
const uint LDS = BK + 8;   // padded row (halves): 144 bytes, 16-byte aligned
const uint THREADS = 128;

shared float16_t aTile[BM * LDS];
shared float16_t bTile[BN * LDS];
shared float cStage[4 * 256];
shared float lutS[16];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint sg = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint row0 = gl_WorkGroupID.y * BM;
    uint col0 = gl_WorkGroupID.x * BN;
    uint rowTiles = min(4u, (p.M - row0 + 15u) / 16u);  // workgroup-uniform
    uint kVec = p.K / 4;
    uint groups = p.K / 32;
    uint scaleBase = p.N * (p.K / 2);
    uint bCol = lid >> 1;
    uint bHalf = lid & 1u;
    uint gCol = col0 + bCol;

#if WEIGHT_FORMAT != 0
    if (lid < 16u) lutS[lid] = lut[lid];
#endif
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[4];
    [[unroll]] for (uint r = 0; r < 4u; ++r)
        acc[r] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    barrier();

    for (uint base = 0; base < p.K; base += BK) {
        // Activations: only active row tiles, 16 vec4 per row.
        for (uint i = lid; i < rowTiles * 16u * (BK / 4u); i += THREADS) {
            uint r = i >> 4;
            uint q = i & 15u;
            uint row = row0 + r;
            vec4 v = row < p.M ? x4[row * kVec + (base >> 2) + q] : vec4(0.0);
            uint o = r * LDS + q * 4u;
            aTile[o] = float16_t(v.x);
            aTile[o + 1u] = float16_t(v.y);
            aTile[o + 2u] = float16_t(v.z);
            aTile[o + 3u] = float16_t(v.w);
        }
#if WEIGHT_FORMAT == 0
        // FP32 weights: 64 columns x 16 vec4, coalesced along K.
        for (uint i = lid; i < BN * (BK / 4u); i += THREADS) {
            uint c = i >> 4;
            uint q = i & 15u;
            uint col = col0 + c;
            vec4 v = col < p.N ? wf4[col * kVec + (base >> 2) + q] : vec4(0.0);
            uint o = c * LDS + q * 4u;
            bTile[o] = float16_t(v.x);
            bTile[o + 1u] = float16_t(v.y);
            bTile[o + 2u] = float16_t(v.z);
            bTile[o + 3u] = float16_t(v.w);
        }
#else
        // Weights: byte i of a 32-value group holds k=i (low) and k=16+i (high).
        uint g = (base >> 5) + bHalf;
        uvec4 packed = uvec4(0u);
        float scale = 0.0;
        if (gCol < p.N) {
            packed = w4[gCol * groups + g];
            uint sb = scaleBase + gCol * groups + g;
            scale = lut[16u + ((w[sb >> 2] >> ((sb & 3u) * 8u)) & 255u)];
        }
        uint ob = bCol * LDS + bHalf * 32u;
        [[unroll]] for (uint j = 0; j < 4u; ++j) {
            uint word = packed[j];
            [[unroll]] for (uint b = 0; b < 4u; ++b) {
                uint byteV = (word >> (8u * b)) & 255u;
                bTile[ob + 4u * j + b] = float16_t(lutS[byteV & 15u] * scale);
                bTile[ob + 16u + 4u * j + b] = float16_t(lutS[byteV >> 4] * scale);
            }
        }
#endif
        barrier();
        if (sg < 4u) {
            [[unroll]] for (uint kk = 0; kk < BK; kk += 16u) {
                coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bm;
                coopMatLoad(bm, bTile, sg * 16u * LDS + kk, LDS,
                    gl_CooperativeMatrixLayoutColumnMajor);
                [[unroll]] for (uint r = 0; r < 4u; ++r) {
                    if (r < rowTiles) {
                        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
                        coopMatLoad(am, aTile, r * 16u * LDS + kk, LDS,
                            gl_CooperativeMatrixLayoutRowMajor);
                        acc[r] = coopMatMulAdd(am, bm, acc[r]);
                    }
                }
            }
        }
        barrier();
    }

    // Each subgroup stages its own 16x16 tile, then writes in-bounds elements.
    if (sg < 4u) {
        [[unroll]] for (uint r = 0; r < 4u; ++r) {
            if (r < rowTiles) {
                coopMatStore(acc[r], cStage, sg * 256u, 16,
                    gl_CooperativeMatrixLayoutRowMajor);
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                for (uint e = lane; e < 256u; e += gl_SubgroupSize) {
                    uint row = row0 + r * 16u + (e >> 4);
                    uint col = col0 + sg * 16u + (e & 15u);
                    if (row < p.M && col < p.N) y[row * p.N + col] = cStage[sg * 256u + e];
                }
                subgroupBarrier();
            }
        }
    }
}
