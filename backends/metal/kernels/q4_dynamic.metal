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
            const uint packed = *((device const uint*)(wq + n * uint(K / 2) + k / 2));
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
