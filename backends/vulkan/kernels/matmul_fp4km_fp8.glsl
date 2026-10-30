#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_float_e4m3 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

// FP4-KM weights stay packed in device memory. Decode only each live tile,
// convert activations and decoded weights to E4M3, and accumulate in FP32.
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Input { float x[]; };
layout(std430, binding = 1) readonly buffer Weight { uint w[]; };
layout(std430, binding = 2) writeonly buffer Output { float y[]; };
layout(std430, binding = 3) readonly buffer LutScale { float lut[]; };
layout(std430, binding = 4) readonly buffer Zero { float zero[]; };
layout(push_constant) uniform Params { uint M, K, N, groupSize, groupsPerRow; } p;
shared floate4m3_t aTile[16 * 16];
shared floate4m3_t bTile[16 * 16];
shared float cTile[16 * 16];

uint loadByte(uint i) { return (w[i >> 2] >> ((i & 3) * 8)) & 255; }
float weightAt(uint col, uint k) {
    if (col >= p.N || k >= p.K) return 0.0;
    uint g = k / 32;
    uint b = loadByte(col * (p.K / 2) + g * 16 + (k & 15));
    uint q = (b >> ((k & 31) >= 16 ? 4 : 0)) & 15;
    uint s = loadByte(p.N * (p.K / 2) + col * (p.K / 32) + g);
    return lut[q] * lut[16 + s];
}

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint row0 = gl_WorkGroupID.y * 16;
    uint col0 = gl_WorkGroupID.x * 16;
    coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
    coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sum =
        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    for (uint base = 0; base < p.K; base += 16) {
        for (uint i = lid; i < 256; i += gl_WorkGroupSize.x) {
            uint row = row0 + i / 16;
            uint k = base + i % 16;
            aTile[i] = floate4m3_t(row < p.M && k < p.K ? x[row * p.K + k] : 0.0);
            bTile[i] = floate4m3_t(weightAt(col0 + i / 16, k));
        }
        barrier();
        coopMatLoad(a, aTile, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(b, bTile, 0, 16, gl_CooperativeMatrixLayoutColumnMajor);
        sum = coopMatMulAdd(a, b, sum);
        barrier();
    }
    coopMatStore(sum, cTile, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
    barrier();
    for (uint i = lid; i < 256; i += gl_WorkGroupSize.x) {
        uint row = row0 + i / 16, col = col0 + i % 16;
        if (row < p.M && col < p.N) y[row * p.N + col] = cTile[i];
    }
}
