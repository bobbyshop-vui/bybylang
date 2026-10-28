#version 450
// WG_SIZE, OUT_ROWS and BATCH_ROWS are supplied when compiling each candidate.
// Bindings and packed Q4 layout match matmul_q4.comp; no weight repacking.
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Input { float a[]; };
layout(std430, binding = 1) readonly buffer Weight { uint w[]; };
layout(std430, binding = 2) readonly buffer Scale { float scales[]; };
layout(std430, binding = 3) readonly buffer Zero { float zeros[]; };
layout(std430, binding = 4) writeonly buffer Output { float outData[]; };
layout(push_constant) uniform Params { uint M, K, N, groupSize, groupsPerRow; } p;
shared float partial[OUT_ROWS * BATCH_ROWS * WG_SIZE];

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint n0 = gl_WorkGroupID.x * OUT_ROWS;
    uint m0 = gl_WorkGroupID.y * BATCH_ROWS;
    float accum[OUT_ROWS * BATCH_ROWS];
    for (uint i = 0; i < OUT_ROWS * BATCH_ROWS; ++i) accum[i] = 0;
    for (uint k = lane * 8; k < p.K; k += WG_SIZE * 8) {
        uint group = p.groupSize > 0 ? k / p.groupSize : 0;
        for (uint r = 0; r < OUT_ROWS; ++r) {
            uint n = n0 + r;
            if (n >= p.N) continue;
            uint packed = w[(n * (p.K / 2) + k / 2) / 4];
            uint g = n * p.groupsPerRow + group;
            float scale = scales[g], zero = zeros[g];
            for (uint t = 0; t < 8; ++t) {
                float weight = (float((packed >> (4 * t)) & 15) - zero) * scale;
                for (uint b = 0; b < BATCH_ROWS; ++b) {
                    uint m = m0 + b;
                    if (m < p.M)
                        accum[r * BATCH_ROWS + b] = fma(weight,
                            a[m * p.K + k + t], accum[r * BATCH_ROWS + b]);
                }
            }
        }
    }
    for (uint i = 0; i < OUT_ROWS * BATCH_ROWS; ++i)
        partial[i * WG_SIZE + lane] = accum[i];
    barrier();
    for (uint stride = WG_SIZE / 2; stride > 0; stride /= 2) {
        if (lane < stride)
            for (uint i = 0; i < OUT_ROWS * BATCH_ROWS; ++i)
                partial[i * WG_SIZE + lane] += partial[i * WG_SIZE + lane + stride];
        barrier();
    }
    if (lane == 0)
        for (uint r = 0; r < OUT_ROWS; ++r)
            for (uint b = 0; b < BATCH_ROWS; ++b)
                if (n0 + r < p.N && m0 + b < p.M)
                    outData[(m0 + b) * p.N + n0 + r] = partial[(r * BATCH_ROWS + b) * WG_SIZE];
}
