#version 450
#extension GL_EXT_control_flow_attributes : require
// FIR custom micro-kernel for asymmetric Q4 decode GEMV (y[m, n] = x[m, :] . W[n, :]).
// It is compiled by the micro-VM JIT and dispatched as a normal FIR micro node; it
// does not call the native Q4/session API. The algorithm matches the native
// matmul_q4_sg.glsl decode kernel:
// - one group of 32 invocations owns ROWS output rows; consecutive lanes read
//   consecutive words of one weight row (VEC=4: one uvec4 = 32 nibbles), so the
//   weights stream at full DRAM width;
// - a load unit lies inside one quantization group, so the scale/zero are applied
//   once per unit: sum_t a_t * (q_t - z) * s = s * (dot(a, q) - z * sum(a));
// - partial sums meet in subgroupAdd (SHARED_REDUCE=0, devices whose compute
//   subgroups are guaranteed 32 wide) or in a shared-memory sum (SHARED_REDUCE=1).
// A workgroup of WG invocations owns ROWS_PER_GROUP = ROWS * WG / 32 rows of one
// activation row m: workgroup w computes m = w / ROW_GROUPS.
// Slots: memF32 = output [M, N]; memU8 = packed nibbles [N, K/2] followed by the
// FP32 scales and zero points [N, N_GROUPS] (read through lut); input = x [M, K].
#define VEC @VEC@
#define ROWS @ROWS@
#define WG @WG@
#define SHARED_REDUCE @SHARED_REDUCE@
#if SHARED_REDUCE
#define LANE (gl_LocalInvocationID.x % 32u)
#define SG_ID (gl_LocalInvocationID.x / 32u)
#define SG_COUNT (WG / 32u)
#define SG_SIZE 32u
#else
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#define LANE gl_SubgroupInvocationID
#define SG_ID gl_SubgroupID
#define SG_COUNT gl_NumSubgroups
#define SG_SIZE gl_SubgroupSize
#endif
layout(local_size_x_id = 0) in;
layout(std430, set=0, binding=0) buffer MemF32Buf { float memF32[]; };
#if VEC == 4
layout(std430, set=0, binding=1) readonly buffer MemU8Buf { uvec4 memU8[]; };
#else
layout(std430, set=0, binding=1) readonly buffer MemU8Buf { uint memU8[]; };
#endif
layout(std430, set=0, binding=2) readonly buffer LutBuf { float lut[]; };
layout(std430, set=0, binding=3) readonly buffer InputBuf { vec4 in4[]; };

const uint Q4_M = @M@u;
const uint Q4_K = @K@u;
const uint Q4_N = @N@u;
const uint Q4_GROUP_SIZE = @GROUP_SIZE@u;
const uint Q4_N_GROUPS = @N_GROUPS@u;
const uint UNITS = Q4_K / (8u * VEC);                 // load units per weight row
const uint SCALE_BASE = (Q4_N * (Q4_K / 2u) + 3u) / 4u;
const uint ZERO_BASE = SCALE_BASE + Q4_N * Q4_N_GROUPS;
const uint ROWS_PER_GROUP = ROWS * WG / 32u;
const uint ROW_GROUPS = (Q4_N + ROWS_PER_GROUP - 1u) / ROWS_PER_GROUP;

#if SHARED_REDUCE
shared float red[WG];
#endif

float dotWord(uint packed, vec4 lo, vec4 hi) {
    vec4 q0 = vec4(packed & 15u, (packed >> 4) & 15u, (packed >> 8) & 15u, (packed >> 12) & 15u);
    vec4 q1 = vec4((packed >> 16) & 15u, (packed >> 20) & 15u, (packed >> 24) & 15u, packed >> 28);
    return dot(lo, q0) + dot(hi, q1);
}

void main() {
    const uint lane = LANE;
    const uint m = gl_WorkGroupID.x / ROW_GROUPS;   // uniform per workgroup
    if (m >= Q4_M) return;
    const uint rowBase = (gl_WorkGroupID.x % ROW_GROUPS) * ROWS_PER_GROUP;
    const uint aBase = m * (Q4_K / 4u);
    // SHARED_REDUCE: SG_COUNT * ROWS == ROWS_PER_GROUP, one trip for every invocation.
    for (uint chunk = SG_ID * ROWS; chunk < ROWS_PER_GROUP; chunk += SG_COUNT * ROWS) {
        const uint row0 = rowBase + chunk;
        // Every loop over ROWS or the 8 activation vectors is unrolled, so acc[] and a[]
        // keep constant indices and stay in registers.
        float acc[ROWS];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) acc[r] = 0.0;
        for (uint j = lane; j < UNITS; j += SG_SIZE) {
            const uint k = j * 8u * VEC;
            const uint g = k / Q4_GROUP_SIZE;
#if VEC == 4
            vec4 a[8];
            [[unroll]] for (uint t = 0u; t < 8u; ++t) a[t] = in4[aBase + k / 4u + t];
            const float asum = dot(a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7], vec4(1.0));
#else
            const vec4 lo = in4[aBase + k / 4u];
            const vec4 hi = in4[aBase + k / 4u + 1u];
            const float asum = dot(lo + hi, vec4(1.0));
#endif
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                const uint n = row0 + r;
                if (n >= Q4_N) continue;
#if VEC == 4
                const uvec4 q = memU8[n * UNITS + j];
                const float d = dotWord(q.x, a[0], a[1]) + dotWord(q.y, a[2], a[3]) +
                                dotWord(q.z, a[4], a[5]) + dotWord(q.w, a[6], a[7]);
#else
                const float d = dotWord(memU8[n * UNITS + j], lo, hi);
#endif
                const uint gi = n * Q4_N_GROUPS + g;
                acc[r] += lut[SCALE_BASE + gi] * (d - lut[ZERO_BASE + gi] * asum);
            }
        }
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
#if SHARED_REDUCE
            red[gl_LocalInvocationID.x] = acc[r];
            barrier();
            float s = 0.0;
            if (lane == 0u)
                for (uint i = 0u; i < 32u; ++i) s += red[gl_LocalInvocationID.x + i];
            barrier();
#else
            const float s = subgroupAdd(acc[r]);
#endif
            if (lane == 0u && row0 + r < Q4_N && chunk + r < ROWS_PER_GROUP)
                memF32[m * Q4_N + row0 + r] = s;
        }
    }
}
