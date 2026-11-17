// FIR custom micro-kernel for FP4-KM prefill (Codestral Mamba-2 weights) on Apple GPUs: the
// Metal lowering of OP_FP4_COOP_GEMM (Vulkan: fir_fp4_coop_micro.glsl). It is compiled by the
// micro-VM JIT and dispatched as a normal FIR micro node; it does not call the native FP4
// session GEMM. MSL simdgroup_matrix needs an Apple GPU (family 7+, MSL 2.3).
// Same tiling as fir_q4_simd_micro.metal: one 128-thread threadgroup (four 32-lane
// simdgroups) computes BM rows x 64 columns (BM = q4CoopRowTile(rows, N): 16, 32 or 64,
// chosen per dispatch). Each 32-deep K step is exactly one FP4 group: thread (column, half)
// reads the group's 16 bytes and decodes the low nibbles (elements 0..15) or the high
// nibbles (elements 16..31) into threadgroup FP32. Accumulation is FP32.
// Slots: memF32 = output [M, N]; memU8 and lut = the same FP4-KM blob: nibble bytes [N, K/2]
// (byte j of a 32-value group holds element j in the low nibble and element j+16 in the high
// one), scale bytes [N, K/32], padding to a word, then 272 floats at LUT_BASE: 16 FP4 values
// and 256 scale values (weight = lut[nibble] * lut[16 + scale byte]); inF32 = x [M, K].
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint FP4_M = @M@u;
constant constexpr uint FP4_K = @K@u;
constant constexpr uint FP4_N = @N@u;
constant constexpr uint FP4_N_GROUPS = @N_GROUPS@u;   // K / 32
constant constexpr uint BM = @BM@u;
constant constexpr uint BN = 64u;
constant constexpr uint BK = 32u;
constant constexpr uint THREADS = 128u;
constant constexpr uint ALD = BK + 4u;
constant constexpr uint BLD = BN + 4u;
constant constexpr uint ROW_FRAGS = BM / 8u;
constant constexpr uint N_TILES = (FP4_N + BN - 1u) / BN;
constant constexpr uint ROW_BYTES = FP4_K / 2u;
constant constexpr uint SCALE_BASE = FP4_N * ROW_BYTES;                          // first scale byte
constant constexpr uint LUT_BASE = (SCALE_BASE + FP4_N * FP4_N_GROUPS + 3u) / 4u; // first decode-table float

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
    threadgroup float lutS[16];

    const uint row0 = (tgid / N_TILES) * BM;
    const uint col0 = (tgid % N_TILES) * BN;
    // Uniform per threadgroup. A graph lifted with a shorter row tile launches
    // threadgroups past the last row; they have nothing to do.
    if (row0 >= FP4_M) return;
    const uint liveFrags = min(ROW_FRAGS, (FP4_M - row0 + 7u) / 8u);
    const uint liveRows = liveFrags * 8u;
    const uint bCol = lid >> 1;
    const uint bHalf = lid & 1u;
    const uint gCol = col0 + bCol;
    const uint sgCol = sg * 16u;

    if (lid < 16u) lutS[lid] = lut[LUT_BASE + lid];
    simdgroup_float8x8 acc[2 * ROW_FRAGS];
    for (uint i = 0u; i < 2u * ROW_FRAGS; ++i)
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0u; base < FP4_K; base += BK) {
        // Activations: only live row fragments, 8 float4 per row (rows past M are zero).
        for (uint i = lid; i < liveRows * (BK / 4u); i += THREADS) {
            const uint r = i >> 3;
            const uint q = i & 7u;
            const uint row = row0 + r;
            const float4 v = row < FP4_M
                ? float4(*((device const packed_float4*)(inF32 + row * FP4_K + base + q * 4u)))
                : float4(0.0f);
            const uint o = r * ALD + q * 4u;
            aTile[o] = v.x;
            aTile[o + 1u] = v.y;
            aTile[o + 2u] = v.z;
            aTile[o + 3u] = v.w;
        }

        // Weights: thread (column, half) decodes 16 values of group g = base / 32.
        {
            const uint g = base >> 5;
            uint words[4] = {0u, 0u, 0u, 0u};
            float scale = 0.0f;
            if (gCol < FP4_N) {
                device const uint* src = (device const uint*)(memU8 + gCol * ROW_BYTES + g * 16u);
                words[0] = src[0];
                words[1] = src[1];
                words[2] = src[2];
                words[3] = src[3];
                scale = lut[LUT_BASE + 16u + uint(memU8[SCALE_BASE + gCol * FP4_N_GROUPS + g])];
            }
            const uint shift = bHalf * 4u;
            const uint kOff = bHalf * 16u;
            for (uint j = 0u; j < 4u; ++j) {
                const uint word = words[j];
                for (uint b = 0u; b < 4u; ++b) {
                    const uint nibble = (word >> (8u * b + shift)) & 15u;
                    bTile[(kOff + 4u * j + b) * BLD + bCol] = lutS[nibble] * scale;
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
            if (col >= FP4_N) continue;
            if (row + 8u <= FP4_M && col + 8u <= FP4_N) {
                simdgroup_store(acc[2u * r + f], memF32 + row * FP4_N + col, ulong(FP4_N));
            } else {
                threadgroup float* stage = cStage + sg * 64u;
                simdgroup_store(acc[2u * r + f], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) {
                    const uint rr = row + (e >> 3);
                    const uint cc = col + (e & 7u);
                    if (rr < FP4_M && cc < FP4_N) memF32[rr * FP4_N + cc] = stage[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}
