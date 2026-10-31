#version 450
// Q4 GEMV/skinny GEMM for decode: one subgroup per ROWS output rows.
// WG_SIZE, ROWS, VEC (1: uint, 4: uvec4 weight loads), SG_HINT (the subgroup
// size the host assumed for the grid) and SHARED_REDUCE are supplied when
// compiling each candidate; the autotuner checks parity with matmul_q4.comp.
// A workgroup owns ROWS * WG_SIZE / SG_HINT rows and its subgroups stride
// over them, so a driver that picks another subgroup width stays correct.
//
// Consecutive lanes read consecutive words of one row (128 B per subgroup and
// row per step on a 32-wide subgroup), so the weights stream at full DRAM
// width. Each word holds 8 nibbles of one quant group, which lets the lane
// apply the group's scale/zero once:
//   sum_t a_t * (q_t - z) * s = s * (dot(a, q) - z * sum(a))
// The partial sums meet in subgroupAdd, without shared memory or barriers.
// SHARED_REDUCE=1 (drivers without subgroup arithmetic) uses fixed groups of
// SG_HINT invocations and a shared-memory sum instead of subgroup builtins.
// Bindings and packed layout match matmul_q4.comp; wqOff must be a multiple
// of 4*VEC bytes, K of 8*VEC and the group size of 8*VEC (host checks).
#if SHARED_REDUCE
#define LANE (gl_LocalInvocationID.x % SG_HINT)
#define SG_ID (gl_LocalInvocationID.x / SG_HINT)
#define SG_COUNT (WG_SIZE / SG_HINT)
#define SG_SIZE SG_HINT
#else
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#define LANE gl_SubgroupInvocationID
#define SG_ID gl_SubgroupID
#define SG_COUNT gl_NumSubgroups
#define SG_SIZE gl_SubgroupSize
#endif
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Input { vec4 a4[]; };
#if VEC == 4
layout(std430, binding = 1) readonly buffer Weight { uvec4 w[]; };
#else
layout(std430, binding = 1) readonly buffer Weight { uint w[]; };
#endif
layout(std430, binding = 2) readonly buffer Scale { float scales[]; };
layout(std430, binding = 3) readonly buffer Zero { float zeros[]; };
layout(std430, binding = 4) writeonly buffer Output { float outData[]; };
layout(push_constant) uniform Params { uint M, K, N, groupSize, groupsPerRow, wqOff; } p;

#define ROWS_PER_GROUP (ROWS * WG_SIZE / SG_HINT)
#if SHARED_REDUCE
shared float red[WG_SIZE];
#endif

float dotWord(uint packed, vec4 lo, vec4 hi) {
    vec4 q0 = vec4(packed & 15u, (packed >> 4) & 15u, (packed >> 8) & 15u, (packed >> 12) & 15u);
    vec4 q1 = vec4((packed >> 16) & 15u, (packed >> 20) & 15u, (packed >> 24) & 15u, packed >> 28);
    return dot(lo, q0) + dot(hi, q1);
}

void main() {
    const uint lane = LANE;
    const uint m = gl_WorkGroupID.y;   // uniform per workgroup
    if (m >= p.M) return;
    const uint unitsPerRow = p.K / (8u * VEC);
    const uint wBase = p.wqOff / (4u * VEC);
    const uint aBase = m * (p.K / 4u);
    // SHARED_REDUCE: SG_COUNT * ROWS == ROWS_PER_GROUP, one trip for all.
    for (uint chunk = SG_ID * ROWS; chunk < ROWS_PER_GROUP; chunk += SG_COUNT * ROWS) {
        const uint row0 = gl_WorkGroupID.x * ROWS_PER_GROUP + chunk;
        float acc[ROWS];
        for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0;
        for (uint j = lane; j < unitsPerRow; j += SG_SIZE) {
            const uint k = j * 8u * VEC;
            const uint g = p.groupSize > 0 ? k / p.groupSize : 0u;
#if VEC == 4
            vec4 a[8];
            for (uint t = 0; t < 8u; ++t) a[t] = a4[aBase + k / 4u + t];
            const float asum = dot(a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7], vec4(1.0));
#else
            const vec4 lo = a4[aBase + k / 4u], hi = a4[aBase + k / 4u + 1u];
            const float asum = dot(lo + hi, vec4(1.0));
#endif
            for (uint r = 0; r < ROWS; ++r) {
                const uint n = row0 + r;
                if (n >= p.N) break;
#if VEC == 4
                const uvec4 q = w[wBase + n * unitsPerRow + j];
                const float d = dotWord(q.x, a[0], a[1]) + dotWord(q.y, a[2], a[3]) +
                                dotWord(q.z, a[4], a[5]) + dotWord(q.w, a[6], a[7]);
#else
                const float d = dotWord(w[wBase + n * unitsPerRow + j], lo, hi);
#endif
                const uint gi = n * p.groupsPerRow + g;
                acc[r] += scales[gi] * (d - zeros[gi] * asum);
            }
        }
        for (uint r = 0; r < ROWS; ++r) {
#if SHARED_REDUCE
            red[gl_LocalInvocationID.x] = acc[r];
            barrier();
            float s = 0.0;
            if (lane == 0u)
                for (uint i = 0; i < SG_HINT; ++i) s += red[gl_LocalInvocationID.x + i];
            barrier();
#else
            const float s = subgroupAdd(acc[r]);
#endif
            if (lane == 0u && row0 + r < p.N && chunk + r < ROWS_PER_GROUP)
                outData[m * p.N + row0 + r] = s;
        }
    }
}
