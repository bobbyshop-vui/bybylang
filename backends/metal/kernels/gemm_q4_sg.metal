// Grouped asymmetric Q4 GEMM for prefill with simdgroup_matrix (Apple GPUs).
// Same weight layout and math as matmul_q4_dynamic:
//   out[m][n] = sum_k a[m][k] * (q[n][k] - zero[n][g]) * scale[n][g]
// Compiled only on an Apple GPU from its own library (see q4_dynamic.inc), so an
// Intel/AMD GPU never sees it. Tile layout follows MLX's quantized qmm_t: a
// 32 x 32 output tile per threadgroup, K walked in steps of 32, the activation tile
// and the dequantised weight tile staged in threadgroup memory, four simdgroups
// (2 x 2) each multiplying a 16 x 16 block with 8 x 8 simdgroup matrices.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul_q4_mm_sg(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr int BM = 32, BN = 32, BK = 32, LD = BK + 4;
    threadgroup float Xs[BM * LD];
    threadgroup float Ws[BN * LD];
    const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
    const int sm = int(sgid) / 2, sn = int(sgid) % 2;
    // This thread fills row `lr` of both tiles, 8 consecutive k from `lc`.
    const int lr = int(lid) / 4, lc = (int(lid) % 4) * 8;
    const uint rowBytes = uint(K) / 2;

    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int gk = k0 + lc;
        const bool kOk = gk < K; // K is a multiple of 8, so the 8 values are all in or all out
        {
            const int gm = m0 + lr;
            float4 v0 = float4(0), v1 = float4(0);
            if (kOk && gm < M) {
                device const float4* src = (device const float4*)(a + gm * K + gk);
                v0 = src[0]; v1 = src[1];
            }
            threadgroup float4* dst = (threadgroup float4*)(Xs + lr * LD + lc);
            dst[0] = v0; dst[1] = v1;
        }
        {
            const int n = n0 + lr;
            float w[8];
            for (int t = 0; t < 8; ++t) w[t] = 0;
            if (kOk && n < N) {
                device const uint* word = (device const uint*)(wq + uint(n) * rowBytes + uint(gk) / 2);
                const uint p = wqShift == 0 ? word[0] :
                    (word[0] >> (8 * wqShift)) | (word[1] << (32 - 8 * wqShift));
                const int gi = n * groupsPerRow + (groupSize > 0 ? gk / groupSize : 0);
                const float sc = scales[gi], zp = zeros[gi];
                for (int t = 0; t < 8; ++t)
                    w[t] = (float((p >> (4 * t)) & 15u) - zp) * sc;
            }
            for (int t = 0; t < 8; ++t) Ws[lr * LD + lc + t] = w[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK / 8; ++kk) {
            simdgroup_float8x8 af[2], bf[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(af[i], Xs + (sm * 16 + i * 8) * LD + kk * 8, LD);
            // B[k][n] = Ws[n][k]: load the weight slice transposed.
            for (int j = 0; j < 2; ++j)
                simdgroup_load(bf[j], Ws + (sn * 16 + j * 8) * LD + kk * 8, LD, ulong2(0, 0), true);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m0 + BM <= M && n0 + BN <= N) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(acc[i][j],
                    out + (m0 + sm * 16 + i * 8) * N + n0 + sn * 16 + j * 8, ulong(N));
    } else {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(acc[i][j], Xs + (sm * 16 + i * 8) * LD + sn * 16 + j * 8, LD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(lid); idx < BM * BN; idx += 128) {
            const int r = idx / BN, c = idx % BN;
            if (m0 + r < M && n0 + c < N) out[(m0 + r) * N + n0 + c] = Xs[r * LD + c];
        }
    }
}

// Same GEMM with a configurable tile: BN output columns per threadgroup (32 or 64, i.e. 128 or
// 256 threads) and BK k-values per step (32 or 64). Larger BK halves the barriers and reads 32
// contiguous weight bytes per row; BN = 64 puts 8 simdgroups on one A tile. Picked by the tuner.
constant uint mmBN [[function_constant(130)]];
constant uint mmBK [[function_constant(131)]];

kernel void matmul_q4_mm_sg2(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    const int BM = 32, BN = int(mmBN), BK = int(mmBK), LD = BK + 4;
    const int nThreads = BN * 4;
    threadgroup float Xs[32 * 68];
    threadgroup float Ws[64 * 68];
    const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
    const int sgPerRow = BN / 16;
    const int sm = int(sgid) / sgPerRow, sn = int(sgid) % sgPerRow;
    const uint rowBytes = uint(K) / 2;

    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int chunksPerRow = BK / 8;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int idx = int(lid); idx < BM * chunksPerRow; idx += nThreads) {
            const int r = idx / chunksPerRow, c = (idx % chunksPerRow) * 8;
            const int gk = k0 + c, gm = m0 + r;
            float4 v0 = float4(0), v1 = float4(0);
            if (gk < K && gm < M) {
                device const float4* src = (device const float4*)(a + gm * K + gk);
                v0 = src[0]; v1 = src[1];
            }
            threadgroup float4* dst = (threadgroup float4*)(Xs + r * LD + c);
            dst[0] = v0; dst[1] = v1;
        }
        for (int idx = int(lid); idx < BN * chunksPerRow; idx += nThreads) {
            const int r = idx / chunksPerRow, c = (idx % chunksPerRow) * 8;
            const int gk = k0 + c, n = n0 + r;
            float w[8];
            for (int t = 0; t < 8; ++t) w[t] = 0;
            if (gk < K && n < N) {
                device const uint* word = (device const uint*)(wq + uint(n) * rowBytes + uint(gk) / 2);
                const uint p = wqShift == 0 ? word[0] :
                    (word[0] >> (8 * wqShift)) | (word[1] << (32 - 8 * wqShift));
                const int gi = n * groupsPerRow + (groupSize > 0 ? gk / groupSize : 0);
                const float sc = scales[gi], zp = zeros[gi];
                for (int t = 0; t < 8; ++t)
                    w[t] = (float((p >> (4 * t)) & 15u) - zp) * sc;
            }
            for (int t = 0; t < 8; ++t) Ws[r * LD + c + t] = w[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK / 8; ++kk) {
            simdgroup_float8x8 af[2], bf[2];
            for (int i = 0; i < 2; ++i)
                simdgroup_load(af[i], Xs + (sm * 16 + i * 8) * LD + kk * 8, LD);
            for (int j = 0; j < 2; ++j)
                simdgroup_load(bf[j], Ws + (sn * 16 + j * 8) * LD + kk * 8, LD, ulong2(0, 0), true);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m0 + BM <= M && n0 + BN <= N) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(acc[i][j],
                    out + (m0 + sm * 16 + i * 8) * N + n0 + sn * 16 + j * 8, ulong(N));
    } else {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                simdgroup_store(acc[i][j], Ws + (sm * 16 + i * 8) * BN + sn * 16 + j * 8, ulong(BN));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(lid); idx < BM * BN; idx += nThreads) {
            const int r = idx / BN, c = idx % BN;
            if (m0 + r < M && n0 + c < N) out[(m0 + r) * N + n0 + c] = Ws[r * BN + c];
        }
    }
}

// 64 x 64 output tile per threadgroup for long prompts: four simdgroups (2 x 2) each own a
// 32 x 32 block (4 x 4 simdgroup matrices), so every k step of 8 does 16 multiplies for 8
// fragment loads, twice the ratio of the 16 x 16 blocks above. Same math, fp32 tiles.
kernel void matmul_q4_mm_sg3(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]) {
    constexpr int BM = 64, BN = 64, BK = 32, LD = BK + 4;
    threadgroup float Ts[2 * 64 * LD];
    threadgroup float* Xs = Ts;
    threadgroup float* Ws = Ts + 64 * LD;
    const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
    const int sm = int(sgid) / 2, sn = int(sgid) % 2;
    const uint rowBytes = uint(K) / 2;

    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int t = 0; t < 2; ++t) {
            const int idx = int(lid) + t * 128;
            const int r = idx / 4, c = (idx % 4) * 8;
            const int gk = k0 + c;
            {
                const int gm = m0 + r;
                float4 v0 = float4(0), v1 = float4(0);
                if (gk < K && gm < M) {
                    device const float4* src = (device const float4*)(a + gm * K + gk);
                    v0 = src[0]; v1 = src[1];
                }
                threadgroup float4* dst = (threadgroup float4*)(Xs + r * LD + c);
                dst[0] = v0; dst[1] = v1;
            }
            {
                const int n = n0 + r;
                float w[8];
                for (int u = 0; u < 8; ++u) w[u] = 0;
                if (gk < K && n < N) {
                    device const uint* word = (device const uint*)(wq + uint(n) * rowBytes + uint(gk) / 2);
                    const uint p = wqShift == 0 ? word[0] :
                        (word[0] >> (8 * wqShift)) | (word[1] << (32 - 8 * wqShift));
                    const int gi = n * groupsPerRow + (groupSize > 0 ? gk / groupSize : 0);
                    const float sc = scales[gi], zp = zeros[gi];
                    for (int u = 0; u < 8; ++u)
                        w[u] = (float((p >> (4 * u)) & 15u) - zp) * sc;
                }
                for (int u = 0; u < 8; ++u) Ws[r * LD + c + u] = w[u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK / 8; ++kk) {
            simdgroup_float8x8 af[4], bf[4];
            for (int i = 0; i < 4; ++i)
                simdgroup_load(af[i], Xs + (sm * 32 + i * 8) * LD + kk * 8, LD);
            for (int j = 0; j < 4; ++j)
                simdgroup_load(bf[j], Ws + (sn * 32 + j * 8) * LD + kk * 8, LD, ulong2(0, 0), true);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m0 + BM <= M && n0 + BN <= N) {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                simdgroup_store(acc[i][j],
                    out + (m0 + sm * 32 + i * 8) * N + n0 + sn * 32 + j * 8, ulong(N));
    } else {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                simdgroup_store(acc[i][j], Ts + (sm * 32 + i * 8) * BN + sn * 32 + j * 8, ulong(BN));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = int(lid); idx < BM * BN; idx += 128) {
            const int r = idx / BN, c = idx % BN;
            if (m0 + r < M && n0 + c < N) out[(m0 + r) * N + n0 + c] = Ts[r * BN + c];
        }
    }
}
