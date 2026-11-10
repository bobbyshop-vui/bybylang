// FP4-KM GEMM for prefill: C[m,n] = A[m,k] * W^T, W packed FP4 (see fp4_gemv_vm.nim).
//
// Weight layout, column n of W (K values):
//   packed[n * K/2 + g*16 + j]  : group g (32 values) uses 16 bytes; value j (0..31) is byte j&15,
//                                 low nibble for j < 16, high nibble for j >= 16 (E2M1 index)
//   packed[N*K/2 + n*(K/32) + g]: E4M3 group scale byte
// lut[0..15] = E2M1 values, lut[16 + scaleByte] = E4M3 scale already multiplied by the global scale.
//
// Register-blocked kernel, portable to every Metal GPU: a threadgroup of 256 threads computes a
// BM x 64 tile of C; the weight tile (64 columns x one 32-value group) is decoded once into
// threadgroup memory and reused by every row, so the packed weights are read once per BM rows
// instead of once per 8 rows.
#define BB_FP4_RB(NAME, BM, TM)                                                              \
kernel void NAME(                                                                            \
    device const float* a [[buffer(0)]],                                                     \
    device const uchar* w [[buffer(1)]],                                                     \
    device const float* lut [[buffer(2)]],                                                   \
    device float* c [[buffer(3)]],                                                           \
    device const int& m [[buffer(4)]],                                                       \
    device const int& k [[buffer(5)]],                                                       \
    device const int& n [[buffer(6)]],                                                       \
    uint2 tgid [[threadgroup_position_in_grid]],                                             \
    uint lid [[thread_index_in_threadgroup]]                                                 \
) {                                                                                          \
    constexpr int BN = 64, TN = 4, BK = 32, NTX = BN / TN, NTHREADS = (BM / TM) * NTX;       \
    threadgroup float As[BK][BM + 1];                                                        \
    threadgroup float Bs[BK][BN + 1];                                                        \
    threadgroup float Lut[272];                                                              \
    for (int i = int(lid); i < 272; i += NTHREADS) Lut[i] = lut[i];                          \
    const int tx = int(lid) % NTX, ty = int(lid) / NTX;                                      \
    const int rowBase = int(tgid.y) * BM, colBase = int(tgid.x) * BN;                        \
    const ulong halfK = ulong(k) / 2, scaleBase = ulong(n) * halfK, groups = ulong(k) / 32;  \
    float acc[TM][TN];                                                                       \
    for (int i = 0; i < TM; ++i)                                                             \
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;                                       \
    for (int k0 = 0; k0 < k; k0 += BK) {                                                     \
        const ulong g = ulong(k0) / 32;                                                      \
        for (int e = int(lid); e < BM * BK; e += NTHREADS) {                                 \
            const int mm = e / BK, kk = e % BK, row = rowBase + mm;                          \
            As[kk][mm] = row < m ? a[ulong(row) * ulong(k) + ulong(k0 + kk)] : 0.0f;         \
        }                                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
        /* One packed byte holds values j (low nibble) and j+16 (high nibble) of the group, so a   \
           thread decodes both from a single byte read and a single scale lookup. */             \
        for (int e = int(lid); e < BN * 16; e += NTHREADS) {                                     \
            const int nn = e / 16, j = e % 16, col = colBase + nn;                               \
            float lo = 0.0f, hi = 0.0f;                                                          \
            if (col < n) {                                                                       \
                const uchar byte = w[ulong(col) * halfK + g * 16 + ulong(j)];                    \
                const float sc = Lut[16 + w[scaleBase + ulong(col) * groups + g]];               \
                lo = Lut[byte & 15] * sc;                                                        \
                hi = Lut[byte >> 4] * sc;                                                        \
            }                                                                                    \
            Bs[j][nn] = lo;                                                                      \
            Bs[j + 16][nn] = hi;                                                                 \
        }                                                                                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
        for (int kk = 0; kk < BK; ++kk) {                                                    \
            float ra[TM], rb[TN];                                                            \
            for (int i = 0; i < TM; ++i) ra[i] = As[kk][ty * TM + i];                        \
            for (int j = 0; j < TN; ++j) rb[j] = Bs[kk][tx * TN + j];                        \
            for (int i = 0; i < TM; ++i)                                                     \
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(ra[i], rb[j], acc[i][j]);       \
        }                                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                     \
    }                                                                                        \
    for (int i = 0; i < TM; ++i) {                                                           \
        const int row = rowBase + ty * TM + i;                                               \
        if (row >= m) continue;                                                              \
        for (int j = 0; j < TN; ++j) {                                                       \
            const int col = colBase + tx * TN + j;                                           \
            if (col < n) c[ulong(row) * ulong(n) + ulong(col)] = acc[i][j];                  \
        }                                                                                    \
    }                                                                                        \
}

BB_FP4_RB(gemm_fp4km_rb_64, 64, 4)
BB_FP4_RB(gemm_fp4km_rb_32, 32, 2)
BB_FP4_RB(gemm_fp4km_rb_16, 16, 1)
