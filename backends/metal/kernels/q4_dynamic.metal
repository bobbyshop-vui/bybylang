// Q4 asymmetric counterpart of the wide FP4 inference kernels. The packed
// bytes/scales/zeros retain their existing resident allocations and layout.
constant uint q4_dynamic_tg [[function_constant(100)]];
constant uint q4_dynamic_rows [[function_constant(101)]];
constant uint q4_dynamic_batch [[function_constant(102)]];

kernel void matmul_q4_dynamic(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]], // wq bound 0-3 bytes below the weights
    threadgroup float* partial [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint n0 = gid.x * q4_dynamic_rows;
    const uint m0 = gid.y * q4_dynamic_batch;
    float accum[8]; // maximum rows*batch among the compiled variants
    for (uint i = 0; i < q4_dynamic_rows * q4_dynamic_batch; ++i) accum[i] = 0;
    // K is a multiple of eight. A lane owns one aligned word, including the
    // last partially populated workgroup for shapes such as K=8096.
    for (uint k = lane * 8; k < uint(K); k += q4_dynamic_tg * 8) {
        const uint group = groupSize > 0 ? k / uint(groupSize) : 0;
        for (uint r = 0; r < q4_dynamic_rows; ++r) {
            uint n = n0 + r;
            if (n >= uint(N)) continue;
            // Rows hold K/2 bytes (a multiple of 4), so every word shares
            // the same 0-3 byte shift from an unaligned mapping.
            device const uint* word = (device const uint*)(wq + n * uint(K / 2) + k / 2);
            const uint packed = wqShift == 0 ? word[0] :
                (word[0] >> (8 * wqShift)) | (word[1] << (32 - 8 * wqShift));
            const uint g = n * uint(groupsPerRow) + group;
            const float scale = scales[g], zero = zeros[g];
            for (uint t = 0; t < 8; ++t) {
                const float weight = (float((packed >> (4 * t)) & 15) - zero) * scale;
                for (uint b = 0; b < q4_dynamic_batch; ++b) {
                    uint m = m0 + b;
                    if (m < uint(M))
                        accum[r * q4_dynamic_batch + b] = fma(weight,
                            a[m * uint(K) + k + t], accum[r * q4_dynamic_batch + b]);
                }
            }
        }
    }
    for (uint i = 0; i < q4_dynamic_rows * q4_dynamic_batch; ++i)
        partial[i * q4_dynamic_tg + lane] = accum[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = q4_dynamic_tg / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            for (uint i = 0; i < q4_dynamic_rows * q4_dynamic_batch; ++i)
                partial[i * q4_dynamic_tg + lane] += partial[i * q4_dynamic_tg + lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint r = 0; r < q4_dynamic_rows; ++r)
            for (uint b = 0; b < q4_dynamic_batch; ++b)
                if (n0 + r < uint(N) && m0 + b < uint(M))
                    out[(m0 + b) * uint(N) + n0 + r] = partial[(r * q4_dynamic_batch + b) * q4_dynamic_tg];
    }
}

// Decode (M=1) Q4 GEMV with one simdgroup per `q4s_rows` output rows. The lanes
// of a simdgroup split K (8 weights = one packed word per lane and row), the
// activation is loaded once per step and shared by every row, and the partial
// sums are combined with simd_sum, so there is no threadgroup memory and no
// barrier. dot(q, a) is accumulated per group and corrected once:
//   sum += scale * (dot(q, a) - zero * sum(a)).
constant uint q4s_rows [[function_constant(110)]];
constant uint q4s_sg [[function_constant(111)]];

kernel void matmul_q4_simd(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdW [[threads_per_simdgroup]]) {
    const uint row0 = (tg * q4s_sg + sgid) * q4s_rows;
    if (row0 >= uint(N)) return;
    float sum[8];
    for (uint r = 0; r < q4s_rows; ++r) sum[r] = 0;
    const uint rowBytes = uint(K) / 2;
    const uint lastRow = uint(N) - 1;
    for (uint k = lane * 8; k < uint(K); k += simdW * 8) {
        const float4 a0 = *(device const float4*)(a + k);
        const float4 a1 = *(device const float4*)(a + k + 4);
        const float sumA = (a0.x + a0.y) + (a0.z + a0.w) + ((a1.x + a1.y) + (a1.z + a1.w));
        const uint g = groupSize > 0 ? k / uint(groupSize) : 0;
        for (uint r = 0; r < q4s_rows; ++r) {
            const uint n = min(row0 + r, lastRow);
            device const uint* word = (device const uint*)(wq + n * rowBytes + k / 2);
            const uint p = wqShift == 0 ? word[0] :
                (word[0] >> (8 * wqShift)) | (word[1] << (32 - 8 * wqShift));
            const float d = fma(a0.x, float(p & 15u), fma(a0.y, float((p >> 4) & 15u),
                fma(a0.z, float((p >> 8) & 15u), fma(a0.w, float((p >> 12) & 15u),
                fma(a1.x, float((p >> 16) & 15u), fma(a1.y, float((p >> 20) & 15u),
                fma(a1.z, float((p >> 24) & 15u), a1.w * float(p >> 28))))))));
            const uint gi = n * uint(groupsPerRow) + g;
            sum[r] = fma(scales[gi], d - zeros[gi] * sumA, sum[r]);
        }
    }
    for (uint r = 0; r < q4s_rows; ++r) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < uint(N)) out[row0 + r] = s;
    }
}

// Fused decode GEMV (M=1), MLX-style. One simdgroup computes 4 output rows.
//   mode 0: up to 3 independent matrices (Q/K/V) share the input vector; the
//           grid covers N0+N1+N2 rows, each N must be a multiple of 4.
//   mode 1: gate/up pair; out0[n] = silu(gate[n]) * up[n].
// HAS_RES adds res[n] to the result (residual epilogue for o_proj / down_proj).
constant uint q4f_mode [[function_constant(120)]];
constant uint q4f_res [[function_constant(121)]];

static inline float q4f_row(device const uchar* wq, device const float* sc,
        device const float* zr, uint n, uint rowBytes, uint gpr, uint k, uint g,
        uint shift, float4 a0, float4 a1, float sumA, float acc) {
    device const uint* word = (device const uint*)(wq + n * rowBytes + k / 2);
    const uint p = shift == 0 ? word[0] :
        (word[0] >> (8 * shift)) | (word[1] << (32 - 8 * shift));
    const float d = fma(a0.x, float(p & 15u), fma(a0.y, float((p >> 4) & 15u),
        fma(a0.z, float((p >> 8) & 15u), fma(a0.w, float((p >> 12) & 15u),
        fma(a1.x, float((p >> 16) & 15u), fma(a1.y, float((p >> 20) & 15u),
        fma(a1.z, float((p >> 24) & 15u), a1.w * float(p >> 28))))))));
    const uint gi = n * gpr + g;
    return fma(sc[gi], d - zr[gi] * sumA, acc);
}

kernel void matmul_q4_fused(
    device const float* a [[buffer(0)]],
    device const uchar* wq0 [[buffer(1)]], device const uchar* wq1 [[buffer(2)]],
    device const uchar* wq2 [[buffer(3)]],
    device const float* sc0 [[buffer(4)]], device const float* sc1 [[buffer(5)]],
    device const float* sc2 [[buffer(6)]],
    device const float* zr0 [[buffer(7)]], device const float* zr1 [[buffer(8)]],
    device const float* zr2 [[buffer(9)]],
    device float* out0 [[buffer(10)]], device float* out1 [[buffer(11)]],
    device float* out2 [[buffer(12)]],
    device const float* res [[buffer(13)]],
    constant int& K [[buffer(14)]], constant int& groupSize [[buffer(15)]],
    constant int& groupsPerRow [[buffer(16)]],
    constant uint4& nn [[buffer(17)]],      // N0, N1, N2, unused
    constant uint4& sh [[buffer(18)]],      // wqShift of the three matrices
    uint tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdW [[threads_per_simdgroup]]) {
    const uint row0 = (tg * 2 + sgid) * 4;
    device const uchar* wqA = wq0; device const float* scA = sc0; device const float* zrA = zr0;
    device float* outA = out0; uint shA = sh.x; uint base = 0;
    if (q4f_mode == 0) {
        if (row0 >= nn.x + nn.y + nn.z) return;
        if (row0 >= nn.x + nn.y) {
            wqA = wq2; scA = sc2; zrA = zr2; outA = out2; shA = sh.z; base = nn.x + nn.y;
        } else if (row0 >= nn.x) {
            wqA = wq1; scA = sc1; zrA = zr1; outA = out1; shA = sh.y; base = nn.x;
        }
    } else if (row0 >= nn.x) {
        return;
    }
    const uint local0 = row0 - base;
    float sumA_[4] = {0, 0, 0, 0};
    float sumB_[4] = {0, 0, 0, 0};
    const uint rowBytes = uint(K) / 2;
    const uint gpr = uint(groupsPerRow);
    for (uint k = lane * 8; k < uint(K); k += simdW * 8) {
        const float4 a0 = *(device const float4*)(a + k);
        const float4 a1 = *(device const float4*)(a + k + 4);
        const float sumA = (a0.x + a0.y) + (a0.z + a0.w) + ((a1.x + a1.y) + (a1.z + a1.w));
        const uint g = groupSize > 0 ? k / uint(groupSize) : 0;
        for (uint r = 0; r < 4; ++r) {
            const uint n = local0 + r;
            sumA_[r] = q4f_row(wqA, scA, zrA, n, rowBytes, gpr, k, g, shA, a0, a1, sumA, sumA_[r]);
            if (q4f_mode == 1)
                sumB_[r] = q4f_row(wq1, sc1, zr1, n, rowBytes, gpr, k, g, sh.y, a0, a1, sumA, sumB_[r]);
        }
    }
    for (uint r = 0; r < 4; ++r) {
        float v = simd_sum(sumA_[r]);
        if (q4f_mode == 1) {
            const float u = simd_sum(sumB_[r]);
            v = v / (1.0f + exp(-v)) * u;
        }
        if (q4f_res != 0) v += res[local0 + r];
        if (lane == 0) outA[local0 + r] = v;
    }
}
