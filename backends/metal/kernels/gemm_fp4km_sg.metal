// FP4-KM GEMM with simdgroup_matrix (Apple GPUs). Same data layout and math as gemm_fp4km.metal.
// Compiled only on Apple GPUs from a separate library, so an Intel/AMD GPU never sees it.
//
// 128 threads = 4 simdgroups in a 2x2 grid; the threadgroup computes a 64 x 64 tile of C, each
// simdgroup a 32 x 32 block held in sixteen 8x8 accumulators. Per 32-value group the A tile and
// the decoded weight tile go to threadgroup memory, then every simdgroup multiplies 8-deep slices.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void gemm_fp4km_sg(
    device const float* a [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* lut [[buffer(2)]],
    device float* c [[buffer(3)]],
    device const int& m [[buffer(4)]],
    device const int& k [[buffer(5)]],
    device const int& n [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr int BM = 64, BN = 64, BK = 32, NTHREADS = 128, ALD = BK + 4, BLD = BN + 4;
    threadgroup float As[BM * ALD];
    threadgroup float Bs[BK * BLD];
    threadgroup float Lut[272];
    threadgroup float Stage[4 * 64];
    for (int i = int(lid); i < 272; i += NTHREADS) Lut[i] = lut[i];
    const int rowBase = int(tgid.y) * BM, colBase = int(tgid.x) * BN;
    const int sgRow = int(sg) / 2 * 32, sgCol = int(sg) % 2 * 32;
    const ulong halfK = ulong(k) / 2, scaleBase = ulong(n) * halfK, groups = ulong(k) / 32;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k0 = 0; k0 < k; k0 += BK) {
        const ulong g = ulong(k0) / 32;
        for (int e = int(lid); e < BM * BK; e += NTHREADS) {
            const int mm = e / BK, kk = e % BK, row = rowBase + mm;
            As[mm * ALD + kk] = row < m ? a[ulong(row) * ulong(k) + ulong(k0 + kk)] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // also publishes Lut on the first pass
        for (int e = int(lid); e < BN * BK; e += NTHREADS) {
            const int nn = e / BK, kk = e % BK, col = colBase + nn;
            float v = 0.0f;
            if (col < n) {
                const uchar byte = w[ulong(col) * halfK + g * 16 + ulong(kk & 15)];
                const uchar sc = w[scaleBase + ulong(col) * groups + g];
                v = Lut[(byte >> ((kk >> 4) * 4)) & 15] * Lut[16 + sc];
            }
            Bs[kk * BLD + nn] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int ks = 0; ks < BK; ks += 8) {
            simdgroup_float8x8 fa[4], fb[4];
            for (int i = 0; i < 4; ++i)
                simdgroup_load(fa[i], &As[(sgRow + i * 8) * ALD + ks], ALD);
            for (int j = 0; j < 4; ++j)
                simdgroup_load(fb[j], &Bs[ks * BLD + sgCol + j * 8], BLD);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int row0 = rowBase + sgRow + i * 8, col0 = colBase + sgCol + j * 8;
            if (row0 >= m || col0 >= n) continue;
            if (row0 + 8 <= m && col0 + 8 <= n) {
                simdgroup_store(acc[i][j], c + ulong(row0) * ulong(n) + ulong(col0), ulong(n));
            } else {
                threadgroup float* stage = Stage + sg * 64;
                simdgroup_store(acc[i][j], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = int(lane); e < 64; e += 32) {
                    const int r = row0 + e / 8, cc = col0 + e % 8;
                    if (r < m && cc < n) c[ulong(r) * ulong(n) + ulong(cc)] = stage[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
