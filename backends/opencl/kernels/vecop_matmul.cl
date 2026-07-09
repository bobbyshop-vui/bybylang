__kernel void vecop_add(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}

__kernel void vecop_sub(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] - b[i];
}

__kernel void vecop_mul(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] * b[i];
}

__kernel void vecop_div(__global const float* a, __global const float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] / b[i];
}

__kernel void matmul_naive(__global const float* a, __global const float* b, __global float* c,
                            const int M, const int K, const int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int p = 0; p < K; p++) {
        sum += a[row * K + p] * b[p * N + col];
    }
    c[row * N + col] = sum;
}
