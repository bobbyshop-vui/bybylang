// ============================================================
// matmul_q4_gemm_kernel - tiled prefill GEMM on asymmetric Q4 weights
// ------------------------------------------------------------
// C[M, N] = A[M, K] * W[N, K]^T, W packed like matmul_q4_batched: byte i of a
// row holds k = 2i (low nibble) and k = 2i + 1 (high), value =
// (q - zero[g]) * scale[g], g = n * groupsPerRow + k / groupSize.
//
// The batched/dynamic kernels are GEMV-shaped: the lanes of one threadgroup
// split K and reduce, and a weight byte serves at most 4-8 rows, so an
// 85-token prompt reads every weight 11-22 times and the ALUs wait on memory.
// Here a threadgroup owns a BM x BN output tile (64 x 128 by default): each
// 32-deep K step decodes the BN x 32 weight block ONCE into threadgroup
// memory (one 32-bit word = 8 weights per thread load) next to the BM x 32
// activation block, and each of the 128 threads accumulates a TM x TN
// register tile (8 x 8: 4 threadgroup loads per 64 FMAs). Weights are read
// once per BM rows and the arithmetic is plain FP32 FMA. The loops are fully
// unrolled: without that the accumulators live in private memory and the
// larger tiles run 5x slower on Intel Iris.
// Host contract: 128 threads, K % 32 == 0, groupSize 0 or a multiple of 32,
// wq bound 4-byte aligned with wqShift = the remaining 0-3 bytes.
// ============================================================

#define QG_BK 32

struct Q4GemmParams {
    uint M, K, N, groupSize, groupsPerRow, wqShift;
};

// BM x BN output tile, 128 threads, each a TM x TN register tile (TN % 4 == 0).
#define BB_Q4_GEMM(BM, BN, TM, TN, KNAME)                                                  \
kernel void KNAME(                                                                         \
    device const float* a        [[buffer(0)]],                                            \
    device const uchar* wq       [[buffer(1)]],                                            \
    device const float* scales   [[buffer(2)]],                                            \
    device const float* zeros    [[buffer(3)]],                                            \
    device float* c              [[buffer(4)]],                                            \
    constant Q4GemmParams& p     [[buffer(5)]],                                            \
    uint2 tg                     [[threadgroup_position_in_grid]],                         \
    uint tid                     [[thread_index_in_threadgroup]])                          \
{                                                                                          \
    const uint LDA = BM + 4;                                                               \
    const uint LDW = BN + 4;                                                               \
    threadgroup float As[QG_BK * (BM + 4)];   /* [k][row] */                               \
    threadgroup float Ws[QG_BK * (BN + 4)];   /* [k][col] */                               \
    const uint n0 = tg.x * BN;                                                             \
    const uint m0 = tg.y * BM;                                                             \
    const uint tx = tid % (BN / TN);                                                       \
    const uint ty = tid / (BN / TN);                                                       \
    const uint rowBytes = p.K / 2u;                                                        \
    float4 acc[TM][TN / 4];                                                                \
    _Pragma("unroll") for (uint i = 0; i < TM; ++i)                                        \
        _Pragma("unroll") for (uint j = 0; j < TN / 4; ++j) acc[i][j] = 0.0f;              \
    for (uint k0 = 0; k0 < p.K; k0 += QG_BK) {                                             \
        for (uint i = tid; i < BM * (QG_BK / 4u); i += 128u) {                             \
            uint r = i >> 3;                                                               \
            uint q = i & 7u;                                                               \
            uint m = m0 + r;                                                               \
            float4 v = m < p.M ? *((device const float4*)(a + m * p.K + k0) + q) : float4(0.0f); \
            uint o = (q * 4u) * LDA + r;                                                   \
            As[o] = v.x; As[o + LDA] = v.y; As[o + 2u * LDA] = v.z; As[o + 3u * LDA] = v.w; \
        }                                                                                  \
        for (uint i = tid; i < BN * 4u; i += 128u) {                                       \
            uint col = i >> 2;                                                             \
            uint wi = i & 3u;                                                              \
            uint n = n0 + col;                                                             \
            uint kk = k0 + wi * 8u;                                                        \
            float sc = 0.0f, zp = 0.0f;                                                    \
            uint packed = 0u;                                                              \
            if (n < p.N) {                                                                 \
                device const uint* word = (device const uint*)(wq + n * rowBytes + kk / 2u); \
                packed = p.wqShift == 0u ? word[0] :                                       \
                    (word[0] >> (8u * p.wqShift)) | (word[1] << (32u - 8u * p.wqShift));   \
                uint g = n * p.groupsPerRow + (p.groupSize > 0u ? kk / p.groupSize : 0u);  \
                sc = scales[g];                                                            \
                zp = zeros[g];                                                             \
            }                                                                              \
            _Pragma("unroll") for (uint t = 0; t < 8u; ++t)                                \
                Ws[(wi * 8u + t) * LDW + col] = (float((packed >> (4u * t)) & 15u) - zp) * sc; \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
        for (uint k = 0; k < QG_BK; ++k) {                                                 \
            float av[TM];                                                                  \
            _Pragma("unroll") for (uint i = 0; i < TM; ++i) av[i] = As[k * LDA + ty * TM + i]; \
            float4 wv[TN / 4];                                                             \
            _Pragma("unroll") for (uint j = 0; j < TN / 4; ++j)                            \
                wv[j] = *((threadgroup const float4*)(Ws + k * LDW + tx * TN) + j);        \
            _Pragma("unroll") for (uint i = 0; i < TM; ++i)                                \
                _Pragma("unroll") for (uint j = 0; j < TN / 4; ++j)                        \
                    acc[i][j] = fma(float4(av[i]), wv[j], acc[i][j]);                      \
        }                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                   \
    }                                                                                      \
    const uint row = m0 + ty * TM;                                                         \
    const uint col = n0 + tx * TN;                                                         \
    _Pragma("unroll") for (uint i = 0; i < TM; ++i) {                                      \
        uint m = row + i;                                                                  \
        if (m >= p.M) break;                                                               \
        _Pragma("unroll") for (uint j = 0; j < TN / 4; ++j) {                              \
            uint cc = col + 4u * j;                                                        \
            if (cc + 3u < p.N) {                                                           \
                device float* dst = c + m * p.N + cc;                                      \
                dst[0] = acc[i][j].x; dst[1] = acc[i][j].y;                                \
                dst[2] = acc[i][j].z; dst[3] = acc[i][j].w;                                \
            } else {                                                                       \
                for (uint e = 0; e < 4u; ++e)                                              \
                    if (cc + e < p.N) c[m * p.N + cc + e] = acc[i][j][e];                  \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
}

BB_Q4_GEMM(32, 64, 4, 4, matmul_q4_gemm_kernel)
BB_Q4_GEMM(64, 64, 8, 4, matmul_q4_gemm_64x64_kernel)
BB_Q4_GEMM(32, 128, 4, 8, matmul_q4_gemm_32x128_kernel)
BB_Q4_GEMM(64, 128, 8, 8, matmul_q4_gemm_64x128_kernel)
