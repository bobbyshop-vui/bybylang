// metal_shim.h - C interface exposed to Nim for the Metal compute backend (macOS only).
#ifndef METAL_SHIM_H
#define METAL_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Trả về 1 nếu máy có GPU hỗ trợ Metal, 0 nếu không.
int metal_available(void);

// kernel_src: nguồn Metal Shading Language (đọc từ kernels/vecop_matmul.metal
// lúc Nim compile-time qua staticRead, KHÔNG hardcode trong file .m nữa).
// op: 0=add, 1=sub, 2=mul, 3=div. Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n);

// C(m x n) = A(m x k) * B(k x n), row-major. Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n);

#ifdef __cplusplus
}
#endif

#endif
