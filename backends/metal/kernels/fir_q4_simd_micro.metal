// FIR custom micro-kernel for asymmetric Q4 prefill on Apple GPUs: the Metal lowering of
// OP_Q4_COOP_GEMM (Vulkan: fir_q4_coop_micro.glsl). It is compiled by the micro-VM JIT and
// dispatched as a normal FIR micro node; it does not call the native Q4/session GEMM path.
// MSL simdgroup_matrix needs an Apple GPU (family 7+, MSL 2.3); Intel/AMD Macs do not have it.
// One 128-thread threadgroup (four 32-lane simdgroups) computes BM rows x 64 columns (BM =
// q4CoopRowTile(rows, N): 16, 32 or 64, chosen per dispatch, the same grid as Vulkan). Each
// 32-deep K step dequantizes the 64-column Q4 weight tile once into threadgroup FP32 (one
// thread per column half: two words = 16 K values) and reuses it for every live 8-row
// fragment. Simdgroup s owns columns 16s .. 16s+15 (two 8x8 fragments) with one FP32
// accumulator per row fragment, so tiles and products stay FP32 like the CPU reference.
// Slots: memF32 = output [M, N]; memU8 = packed nibbles [N, K/2] followed by the FP32 scales
// and zero points [N, N_GROUPS] (read through lut); inF32 = x [M, K].
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint Q4_M = @M@u;
constant constexpr uint Q4_K = @K@u;
constant constexpr uint Q4_N = @N@u;
constant constexpr uint Q4_GROUP_SIZE = @GROUP_SIZE@u;
constant constexpr uint Q4_N_GROUPS = @N_GROUPS@u;
constant constexpr uint BM = @BM@u;
constant constexpr uint BN = 64u;
constant constexpr uint BK = 32u;
constant constexpr uint THREADS = 128u;
constant constexpr uint ALD = BK + 4u;              // padded A row (floats)
constant constexpr uint BLD = BN + 4u;              // padded B row (floats)
constant constexpr uint ROW_FRAGS = BM / 8u;
constant constexpr uint N_TILES = (Q4_N + BN - 1u) / BN;
constant constexpr uint ROW_BYTES = Q4_K / 2u;
constant constexpr uint SCALE_BASE = (Q4_N * ROW_BYTES + 3u) / 4u;
constant constexpr uint ZERO_BASE = SCALE_BASE + Q4_N * Q4_N_GROUPS;
// The 16 K values one thread decodes share one quantization group whenever the group size
// is a multiple of 16 (K steps start on multiples of 16).
constant constexpr bool UNIT_GROUP = (Q4_GROUP_SIZE % 16u) == 0u;

kernel void @KERNEL@(
    device float*       memF32 [[buffer(1)]],
    device const uchar* memU8  [[buffer(2)]],
    device const float* lut    [[buffer(3)]],
    device const float* inF32  [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float aTile[BM * ALD];
    threadgroup float bTile[BK * BLD];
    threadgroup float cStage[4u * 64u];

    const uint row0 = (tgid / N_TILES) * BM;
    const uint col0 = (tgid % N_TILES) * BN;
    // Uniform per threadgroup. A graph lifted with a shorter row tile launches
    // threadgroups past the last row; they have nothing to do.
    if (row0 >= Q4_M) return;
    const uint liveFrags = min(ROW_FRAGS, (Q4_M - row0 + 7u) / 8u);
    const uint liveRows = liveFrags * 8u;
    const uint bCol = lid >> 1;
    const uint bHalf = lid & 1u;
    const uint gCol = col0 + bCol;
    const uint sgCol = sg * 16u;

    simdgroup_float8x8 acc[2 * ROW_FRAGS];
    for (uint i = 0u; i < 2u * ROW_FRAGS; ++i)
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint base = 0u; base < Q4_K; base += BK) {
        // Activations: only live row fragments, 8 float4 per row (rows past M are zero).
        for (uint i = lid; i < liveRows * (BK / 4u); i += THREADS) {
            const uint r = i >> 3;
            const uint q = i & 7u;
            const uint row = row0 + r;
            const float4 v = row < Q4_M
                ? float4(*((device const packed_float4*)(inF32 + row * Q4_K + base + q * 4u)))
                : float4(0.0f);
            const uint o = r * ALD + q * 4u;
            aTile[o] = v.x;
            aTile[o + 1u] = v.y;
            aTile[o + 2u] = v.z;
            aTile[o + 3u] = v.w;
        }

        // Weights: thread (column, half) decodes two words = 16 K values into column bCol.
        {
            const uint kb = base + bHalf * 16u;
            uint words[2] = {0u, 0u};
            float sc = 0.0f;
            float zr = 0.0f;
            const uint gBase = gCol * Q4_N_GROUPS;
            if (gCol < Q4_N) {
                device const uint* src = (device const uint*)(memU8 + gCol * ROW_BYTES + (kb >> 1));
                words[0] = src[0];
                words[1] = src[1];
                if (UNIT_GROUP) {
                    sc = lut[SCALE_BASE + gBase + kb / Q4_GROUP_SIZE];
                    zr = lut[ZERO_BASE + gBase + kb / Q4_GROUP_SIZE];
                }
            }
            const uint kOff = bHalf * 16u;
            for (uint j = 0u; j < 2u; ++j) {
                const uint word = words[j];
                for (uint b = 0u; b < 4u; ++b) {
                    const uint k = 8u * j + 2u * b;
                    if (!UNIT_GROUP && gCol < Q4_N) {
                        const uint g = gBase + (kb + k) / Q4_GROUP_SIZE;
                        sc = lut[SCALE_BASE + g];
                        zr = lut[ZERO_BASE + g];
                    }
                    const uint byteV = (word >> (8u * b)) & 255u;
                    bTile[(kOff + k) * BLD + bCol] = (float(byteV & 15u) - zr) * sc;
                    bTile[(kOff + k + 1u) * BLD + bCol] = (float(byteV >> 4) - zr) * sc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 b0, b1;
            simdgroup_load(b0, bTile + kk * BLD + sgCol, BLD);
            simdgroup_load(b1, bTile + kk * BLD + sgCol + 8u, BLD);
            for (uint r = 0u; r < ROW_FRAGS; ++r) {
                if (r < liveFrags) {
                    simdgroup_float8x8 am;
                    simdgroup_load(am, aTile + r * 8u * ALD + kk, ALD);
                    simdgroup_multiply_accumulate(acc[2u * r], am, b0, acc[2u * r]);
                    simdgroup_multiply_accumulate(acc[2u * r + 1u], am, b1, acc[2u * r + 1u]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Full 8x8 fragments go straight to memory; edge fragments are staged per simdgroup.
    for (uint r = 0u; r < ROW_FRAGS; ++r) {
        if (r >= liveFrags) continue;
        for (uint f = 0u; f < 2u; ++f) {
            const uint row = row0 + r * 8u;
            const uint col = col0 + sgCol + f * 8u;
            if (col >= Q4_N) continue;
            if (row + 8u <= Q4_M && col + 8u <= Q4_N) {
                simdgroup_store(acc[2u * r + f], memF32 + row * Q4_N + col, ulong(Q4_N));
            } else {
                threadgroup float* stage = cStage + sg * 64u;
                simdgroup_store(acc[2u * r + f], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) {
                    const uint rr = row + (e >> 3);
                    const uint cc = col + (e & 7u);
                    if (rr < Q4_M && cc < Q4_N) memF32[rr * Q4_N + cc] = stage[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
