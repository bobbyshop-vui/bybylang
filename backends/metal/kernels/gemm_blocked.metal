// Register-blocked FP32 GEMM: C[m,n] = A[m,k] * B, with B stored [k,n] (NN) or [n,k] (NT).
//
// One threadgroup computes a BM x BN tile of C with (BM/TM)*(BN/TN) threads; each thread keeps a
// TM x TN block of accumulators in registers, so every value read from threadgroup memory feeds
// TM*TN multiply-adds (the old kernels did one output per thread: one FMA per two shared reads).
// Tiles of A and B (BK deep) are staged in threadgroup memory with coalesced device loads.
// Variants differ only in BM, so short matrices (a prefill chunk of 8..32 rows) do not waste rows.
#define BB_GEMM_RB(NAME, BM, BN, TM, TN, BK, BT)                                              \
kernel void NAME(                                                                             \
    device const float* a [[buffer(0)]],                                                      \
    device const float* b [[buffer(1)]],                                                      \
    device float* c [[buffer(2)]],                                                            \
    device const int& m [[buffer(3)]],                                                        \
    device const int& k [[buffer(4)]],                                                        \
    device const int& n [[buffer(5)]],                                                        \
    uint2 tgid [[threadgroup_position_in_grid]],                                              \
    uint lid [[thread_index_in_threadgroup]]                                                  \
) {                                                                                           \
    constexpr int NTX = BN / TN;                                                              \
    constexpr int NTHREADS = (BM / TM) * NTX;                                                 \
    threadgroup float As[BK][BM + 1];                                                         \
    threadgroup float Bs[BK][BN + 1];                                                         \
    const int tx = int(lid) % NTX;                                                            \
    const int ty = int(lid) / NTX;                                                            \
    const int rowBase = int(tgid.y) * BM;                                                     \
    const int colBase = int(tgid.x) * BN;                                                     \
    float acc[TM][TN];                                                                        \
    for (int i = 0; i < TM; ++i)                                                              \
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;                                        \
    for (int k0 = 0; k0 < k; k0 += BK) {                                                      \
        for (int e = int(lid); e < BM * BK; e += NTHREADS) {                                  \
            const int mm = e / BK, kk = e % BK;                                               \
            const int row = rowBase + mm, kc = k0 + kk;                                       \
            As[kk][mm] = (row < m && kc < k) ? a[row * k + kc] : 0.0f;                        \
        }                                                                                     \
        for (int e = int(lid); e < BN * BK; e += NTHREADS) {                                  \
            if (BT) {                                                                         \
                const int nn = e / BK, kk = e % BK;                                           \
                const int col = colBase + nn, kc = k0 + kk;                                   \
                Bs[kk][nn] = (col < n && kc < k) ? b[col * k + kc] : 0.0f;                    \
            } else {                                                                          \
                const int kk = e / BN, nn = e % BN;                                           \
                const int col = colBase + nn, kc = k0 + kk;                                   \
                Bs[kk][nn] = (col < n && kc < k) ? b[kc * n + col] : 0.0f;                    \
            }                                                                                 \
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
        for (int kk = 0; kk < BK; ++kk) {                                                     \
            float ra[TM], rb[TN];                                                             \
            for (int i = 0; i < TM; ++i) ra[i] = As[kk][ty * TM + i];                         \
            for (int j = 0; j < TN; ++j) rb[j] = Bs[kk][tx * TN + j];                         \
            for (int i = 0; i < TM; ++i)                                                      \
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(ra[i], rb[j], acc[i][j]);        \
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
    }                                                                                         \
    for (int i = 0; i < TM; ++i) {                                                            \
        const int row = rowBase + ty * TM + i;                                                \
        if (row >= m) continue;                                                               \
        for (int j = 0; j < TN; ++j) {                                                        \
            const int col = colBase + tx * TN + j;                                            \
            if (col < n) c[row * n + col] = acc[i][j];                                        \
        }                                                                                     \
    }                                                                                         \
}

BB_GEMM_RB(gemm_rb_nn_64, 64, 64, 4, 4, 16, 0)
BB_GEMM_RB(gemm_rb_nn_32, 32, 64, 2, 4, 16, 0)
BB_GEMM_RB(gemm_rb_nn_16, 16, 64, 1, 4, 16, 0)
BB_GEMM_RB(gemm_rb_nt_64, 64, 64, 4, 4, 16, 1)
BB_GEMM_RB(gemm_rb_nt_32, 32, 64, 2, 4, 16, 1)
BB_GEMM_RB(gemm_rb_nt_16, 16, 64, 1, 4, 16, 1)
