#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require
// Tiled asymmetric Q4 GEMM matching Metal matmul_q4_gemm.metal.
// BM/BN/TM/TN are injected at JIT compile time; 128 invocations per group.
#define BK 32
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer Input  { float a[]; };
layout(std430, binding = 1) readonly buffer Weight { uint8_t wq[]; };
layout(std430, binding = 2) readonly buffer Scale  { float scales[]; };
layout(std430, binding = 3) readonly buffer Zero   { float zeros[]; };
layout(std430, binding = 4) writeonly buffer Output { float c[]; };
layout(push_constant) uniform Params {
    uint M, K, N, groupSize, groupsPerRow, wqOff;
} p;

shared float As[BK * (BM + 4)];
shared float Ws[BK * (BN + 4)];

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint LDA = BM + 4;
    const uint LDW = BN + 4;
    const uint n0 = gl_WorkGroupID.x * BN;
    const uint m0 = gl_WorkGroupID.y * BM;
    const uint tx = tid % (BN / TN);
    const uint ty = tid / (BN / TN);
    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0;

    for (uint k0 = 0; k0 < p.K; k0 += BK) {
        // Stage A as [k][m], four contiguous K values per invocation.
        for (uint i = tid; i < BM * (BK / 4); i += 128) {
            uint r = i >> 3;
            uint q = i & 7;
            uint m = m0 + r;
            uint k = k0 + q * 4;
            uint o = (q * 4) * LDA + r;
            for (uint e = 0; e < 4; ++e)
                As[o + e * LDA] = m < p.M ? a[m * p.K + k + e] : 0.0;
        }
        // Decode each Q4 tile once into shared memory for reuse by BM rows.
        for (uint i = tid; i < BN * 4; i += 128) {
            uint col = i >> 2;
            uint wi = i & 3;
            uint n = n0 + col;
            uint kk = k0 + wi * 8;
            uint packed = 0;
            float sc = 0.0, zp = 0.0;
            if (n < p.N) {
                uint byte = p.wqOff + n * (p.K / 2) + kk / 2;
                packed = uint(wq[byte]) | (uint(wq[byte + 1]) << 8) |
                         (uint(wq[byte + 2]) << 16) | (uint(wq[byte + 3]) << 24);
                uint g = n * p.groupsPerRow + (p.groupSize > 0 ? kk / p.groupSize : 0);
                sc = scales[g];
                zp = zeros[g];
            }
            for (uint t = 0; t < 8; ++t)
                Ws[(wi * 8 + t) * LDW + col] = (float((packed >> (4 * t)) & 15) - zp) * sc;
        }
        barrier();
        for (uint k = 0; k < BK; ++k) {
            float av[TM];
            for (uint i = 0; i < TM; ++i) av[i] = As[k * LDA + ty * TM + i];
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j)
                    acc[i][j] = fma(av[i], Ws[k * LDW + tx * TN + j], acc[i][j]);
        }
        barrier();
    }

    uint row = m0 + ty * TM;
    uint col = n0 + tx * TN;
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            if (row + i < p.M && col + j < p.N)
                c[(row + i) * p.N + col + j] = acc[i][j];
}
