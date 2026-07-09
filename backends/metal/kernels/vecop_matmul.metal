#include <metal_stdlib>
using namespace metal;

kernel void vecop_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] + b[id];
}

kernel void vecop_sub(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] - b[id];
}

kernel void vecop_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] * b[id];
}

kernel void vecop_div(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    c[id] = a[id] / b[id];
}

kernel void matmul_naive(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          constant int& M [[buffer(3)]],
                          constant int& K [[buffer(4)]],
                          constant int& N [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= (uint)N || gid.y >= (uint)M) return;
    float sum = 0.0;
    for (int p = 0; p < K; p++) {
        sum += a[gid.y * (uint)K + p] * b[(uint)p * (uint)N + gid.x];
    }
    c[gid.y * (uint)N + gid.x] = sum;
}
