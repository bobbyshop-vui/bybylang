// ============================================================
// matmul_q4_batched_kernel — REAL batch M for prefill (M=T tokens > 1)
// ------------------------------------------------------------
// ORIGINAL BUG, found by reading matmul_q4_kernel_ndst2_paired (the M=1
// decode kernel in use, NOT touched): dispatch grid Y = m (one threadgroup per
// row of M), so EVERY row re-read ALL of its Q4 weights from scratch,
// independently of the other rows - with M=92 (prefill) the whole
// weight matrix was read 92 TIMES instead of once. That is why prefill
// (944ms/token) was almost as slow as decode (1121ms/token) although prefill should be
// tens of times faster (the weight reads are amortised over many tokens).
//
// THIS KERNEL: reads each weight byte EXACTLY ONCE and reuses it for MBLOCK=4
// consecutive rows (dispatch grid Y = ceil(M/MBLOCK) instead of Y=M) - about 4x
// less memory traffic for the weights. MBLOCK=4 (no larger) keeps register
// pressure safe - the M=1 decode kernel itself picked NDST=2
// instead of NDST=8 BECAUSE NDST=8 spilled registers on Intel Iris 640
// (see the comment history of matmul_q4_kernel_ndst2_paired) - this kernel adds
// the MBLOCK dimension, so NDST stays at 2 (the lowest) to keep the register total
// within the range that kernel proved does not spill.
//
// SAME MATH as the original kernel (sc[rr]*dot[rr] - zp[rr]*sc[rr]*sumA[mm]),
// except that dot/sum/sumA have an extra mm dimension (0..<MBLOCK) and the weight byte
// wq[...] is read once per k2 and REUSED for every mm of the inner loop.
//
// ONLY FOR M>1 (prefill) - decode (M=1) still uses
// matmul_q4_kernel_ndst2_paired unchanged.
// ============================================================

#define BB_MATMUL_Q4_BATCHED(NDST_VAL, MBLOCK_VAL, KFUNC)                    \
kernel void KFUNC(                                                          \
    device const float* a          [[buffer(0)]],                          \
    device const uchar* wq         [[buffer(1)]],                          \
    device const float* scales     [[buffer(2)]],                          \
    device const float* zeros      [[buffer(3)]],                          \
    device float* c                [[buffer(4)]],                          \
    device const int& M            [[buffer(5)]],                          \
    device const int& K            [[buffer(6)]],                          \
    device const int& N            [[buffer(7)]],                          \
    device const int& groupSize    [[buffer(8)]],                          \
    device const int& nGroupsPerRow [[buffer(9)]],                         \
    device const int& simdWidth    [[buffer(10)]],                         \
    threadgroup float* partial [[threadgroup(0)]],                         \
    uint3 tgpig [[threadgroup_position_in_grid]],                          \
    uint3 tiisgv [[thread_position_in_threadgroup]]                        \
) {                                                                          \
    uint tiisg = tiisgv.x;                                                  \
    const int NDST = NDST_VAL;                                              \
    const int MBLOCK = MBLOCK_VAL;                                          \
    int m0 = int(tgpig.y) * MBLOCK;                                         \
    if (m0 >= M) return;                                                    \
                                                                              \
    int rowBlock = int(tgpig.x);                                            \
    int n0 = rowBlock * NDST;                                               \
    int bytesPerRow = (K + 1) / 2;                                          \
                                                                              \
    float sum[MBLOCK_VAL][NDST_VAL];                                        \
    _Pragma("unroll")                                                       \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) sum[mm][rr] = 0.0f;          \
    }                                                                        \
                                                                              \
    int prevGr = -1;                                                        \
    float sc[NDST_VAL];                                                     \
    float zp[NDST_VAL];                                                     \
    float dot[MBLOCK_VAL][NDST_VAL];                                        \
    _Pragma("unroll")                                                       \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) dot[mm][rr] = 0.0f;          \
    }                                                                        \
    float sumA[MBLOCK_VAL];                                                 \
    _Pragma("unroll")                                                       \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) sumA[mm] = 0.0f;                \
                                                                              \
    for (int k2 = int(tiisg) * 2; k2 < K; k2 += int(simdWidth) * 2) {       \
        bool hasK1 = (k2 + 1) < K;                                          \
        int gr = groupSize > 0 ? (k2 / groupSize) : 0;                      \
                                                                              \
        if (gr != prevGr) {                                                 \
            if (prevGr != -1) {                                             \
                _Pragma("unroll")                                           \
                for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                  \
                    _Pragma("unroll")                                       \
                    for (int rr = 0; rr < NDST_VAL; ++rr) {                \
                        sum[mm][rr] += sc[rr] * dot[mm][rr] - zp[rr] * sc[rr] * sumA[mm]; \
                        dot[mm][rr] = 0.0f;                                  \
                    }                                                        \
                }                                                            \
                _Pragma("unroll")                                           \
                for (int mm = 0; mm < MBLOCK_VAL; ++mm) sumA[mm] = 0.0f;    \
            }                                                               \
            prevGr = gr;                                                    \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                int n = n0 + rr;                                            \
                if (n < N) {                                                \
                    sc[rr] = scales[n * nGroupsPerRow + gr];                \
                    zp[rr] = zeros[n * nGroupsPerRow + gr];                 \
                } else {                                                    \
                    sc[rr] = 0.0f;                                          \
                    zp[rr] = 0.0f;                                          \
                }                                                           \
            }                                                               \
        }                                                                   \
                                                                              \
        int byteIdx = k2 >> 1;                                              \
        /* KEY POINT: each weight byte is read EXACTLY ONCE per k2 and REUSED for all \
           MBLOCK rows below - the ONLY memory-traffic difference                 \
           from the original decode kernel (which re-reads this byte            \
           for EACH row because each row is an independent threadgroup).           \
           NEW (sparsity): if sc[rr]==0 (this block was marked by Nim as   \
           pruned in uploadResident - see nimformer.nim), the weight byte      \
           for that column rr is NOT read at all - this saves REAL read bandwidth, \
           not only arithmetic (without the skip the result would still  \
           be 0 because sc=0, but skipping is what avoids reading    \
           that byte from GPU memory - the real speed gain of        \
           sparsity, not just computing 0 faster). */                    \
        uchar wbyte[NDST_VAL];                                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) {                             \
            int n = n0 + rr;                                                \
            wbyte[rr] = (n < N && sc[rr] != 0.0f) ? wq[n * bytesPerRow + byteIdx] : uchar(0); \
        }                                                                    \
                                                                              \
        _Pragma("unroll")                                                   \
        for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                          \
            int m = m0 + mm;                                                \
            if (m >= M) continue;                                          \
            float aVal0 = a[m * K + k2];                                    \
            float aVal1 = hasK1 ? a[m * K + k2 + 1] : 0.0f;                 \
            sumA[mm] += aVal0 + (hasK1 ? aVal1 : 0.0f);                     \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                int n = n0 + rr;                                            \
                if (n >= N) continue;                                      \
                uchar nib0 = wbyte[rr] & 0x0Fu;                             \
                uchar nib1 = (wbyte[rr] >> 4) & 0x0Fu;                      \
                dot[mm][rr] += aVal0 * float(nib0);                         \
                if (hasK1) dot[mm][rr] += aVal1 * float(nib1);              \
            }                                                                \
        }                                                                    \
    }                                                                        \
    if (prevGr != -1) {                                                     \
        _Pragma("unroll")                                                   \
        for (int mm = 0; mm < MBLOCK_VAL; ++mm) {                          \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                sum[mm][rr] += sc[rr] * dot[mm][rr] - zp[rr] * sc[rr] * sumA[mm]; \
            }                                                                \
        }                                                                    \
    }                                                                         \
                                                                              \
    for (int mm = 0; mm < MBLOCK_VAL; ++mm) \
        for (int rr = 0; rr < NDST_VAL; ++rr) \
            partial[(mm * NDST_VAL + rr) * simdWidth + tiisg] = sum[mm][rr]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint stride = uint(simdWidth) / 2; stride > 0; stride /= 2) { \
        if (tiisg < stride) { \
            for (int i = 0; i < MBLOCK_VAL * NDST_VAL; ++i) \
                partial[i * simdWidth + tiisg] += partial[i * simdWidth + tiisg + stride]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (tiisg == 0) { \
        for (int mm = 0; mm < MBLOCK_VAL; ++mm) \
            for (int rr = 0; rr < NDST_VAL; ++rr) \
                if (m0 + mm < M && n0 + rr < N) \
                    c[(m0 + mm) * N + n0 + rr] = partial[(mm * NDST_VAL + rr) * simdWidth]; \
    } \
}

// MBLOCK=4, NDST=2 - extra registers (dot[4][2]+sum[4][2]+sumA[4]=20)
// vs the original NDST=2 (dot[2]+sum[2]+sumA=5) - a moderate increase; higher NDST
// settings are NOT used (known to spill in the decode kernel).
BB_MATMUL_Q4_BATCHED(2, 4, matmul_q4_batched_m4_kernel)
