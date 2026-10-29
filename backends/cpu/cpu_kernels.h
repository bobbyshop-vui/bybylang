/* Nimformers CPU backend — runtime-dispatched SIMD kernels.
 *
 * The exported nimf_cpu_* functions select the widest safe ISA for the work
 * size.  Small elementwise jobs prefer SSE4/NEON to avoid the AVX frequency
 * licence/offset penalty on Intel; large jobs use AVX2 (or AVX-512 on capable
 * CPUs).  Matmul uses cache-blocked loops sized from L1/L2/L3.
 */
#ifndef NIMFORMERS_CPU_KERNELS_H
#define NIMFORMERS_CPU_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#  define NIMF_CPU_HIDDEN __attribute__((visibility("hidden")))
#else
#  define NIMF_CPU_HIDDEN
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NimfCpuInfo {
    int has_sse4;
    int has_avx;
    int has_avx2;
    int has_fma;
    int has_avx512;
    int has_neon;
    int isa;             /* 0=scalar 1=sse4 2=avx 3=avx2 4=avx512 */
    int small_threshold; /* use SSE at/below this element count */
    int avx512_threshold;
    size_t l1_bytes;
    size_t l2_bytes;
    size_t l3_bytes;
} NimfCpuInfo;

NIMF_CPU_HIDDEN void nimf_cpu_init(void);
NIMF_CPU_HIDDEN NimfCpuInfo* nimf_cpu_info(void);
NIMF_CPU_HIDDEN const char* nimf_cpu_backend_isa_name(int isa);
NIMF_CPU_HIDDEN int nimf_cpu_current_isa(void);
NIMF_CPU_HIDDEN void nimf_cpu_backend_block_sizes(int* mc, int* nc, int* kc);

/* elementwise (n = element count) */
NIMF_CPU_HIDDEN void nimf_cpu_add(float* a, float* b, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_sub(float* a, float* b, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_mul(float* a, float* b, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_div(float* a, float* b, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_relu(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_sqrt(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_exp(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_log(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_sigmoid(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_tanh(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_neg(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_abs(float* x, float* out, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_pow(float* x, float* out, size_t n, float p);
NIMF_CPU_HIDDEN void nimf_cpu_add_scalar(float* x, float* out, size_t n, float s);
NIMF_CPU_HIDDEN void nimf_cpu_mul_scalar(float* x, float* out, size_t n, float s);
NIMF_CPU_HIDDEN void nimf_cpu_apflu(float* x, float* out, size_t n, float alpha, float beta);
NIMF_CPU_HIDDEN void nimf_cpu_apflu_backward(float* x, float* dy, float* out,
                             size_t n, float alpha, float beta);
/* act: 0 none, 1 relu, 2 sigmoid, 3 tanh */
NIMF_CPU_HIDDEN void nimf_cpu_fused_add_act(float* a, float* b, float* out,
                            size_t n, int act);
NIMF_CPU_HIDDEN void nimf_cpu_copy(float* dst, float* src, size_t n);
NIMF_CPU_HIDDEN void nimf_cpu_scale(float* x, size_t n, float s);

/* rows x cols row-major */
NIMF_CPU_HIDDEN void nimf_cpu_softmax(float* x, float* out, int rows, int cols);
NIMF_CPU_HIDDEN void nimf_cpu_layernorm(float* x, float* gamma, float* beta,
                        float* out, int rows, int cols, float eps);
NIMF_CPU_HIDDEN void nimf_cpu_rmsnorm(float* x, float* gamma, float* out,
                      int rows, int cols, float eps);
NIMF_CPU_HIDDEN void nimf_cpu_layernorm_backward(float* dy, float* x,
                                 float* gamma, float* beta,
                                 int rows, int cols, float eps,
                                 float* dx, float* dgamma, float* dbeta);

/* C[M,N] = A[M,K] @ B[K,N] / C[M,N] = A[M,K] @ B[N,K]^T */
NIMF_CPU_HIDDEN void nimf_cpu_matmul(float* a, float* b, float* c,
                     int M, int K, int N);
NIMF_CPU_HIDDEN void nimf_cpu_matmul_nt(float* a, float* b, float* c,
                        int M, int K, int N);
NIMF_CPU_HIDDEN void nimf_cpu_matmul_q4(float* a, uint8_t* wq,
                        float* scales, float* zeros, float* c,
                        int M, int K, int N, int group_size, int groups_per_row);

/* Fused causal attention; scores receives B*H*S*S probabilities (upper = 0). */
NIMF_CPU_HIDDEN void nimf_cpu_attention(float* q, float* k, float* v,
                        float* mask, int mask_len, float* out,
                        float* scores, int B, int H, int S, int D, float scale);

/* data movement */
NIMF_CPU_HIDDEN void nimf_cpu_embedding(float* table, int32_t* indices, float* out,
                        int num_idx, int vocab, int dim);
NIMF_CPU_HIDDEN void nimf_cpu_gather(float* src, int32_t* idx, float* out, int n);
NIMF_CPU_HIDDEN void nimf_cpu_compare(float* a, float* b, uint8_t* mask, int n, int op);
NIMF_CPU_HIDDEN void nimf_cpu_dropout_mask(float* out, int n, float p, uint32_t seed);
NIMF_CPU_HIDDEN void nimf_cpu_argmax(float* x, float* out_val, int32_t* out_idx,
                     int outer, int dim_size, int inner);
NIMF_CPU_HIDDEN void nimf_cpu_permute(float* src, float* out, int32_t* new_strides,
                      int32_t* old_strides, int32_t* dims_map,
                      int ndim, int n);
NIMF_CPU_HIDDEN void nimf_cpu_split_slice(float* src, float* out, int dim_total,
                          int this_size, int offset, int inner, int n);
NIMF_CPU_HIDDEN void nimf_cpu_concat(float* flat_src, float* out, int32_t* src_base,
                     int32_t* src_dim_size, int32_t* src_dim_offset,
                     int num_tensors, int total_dim, int inner, int n);
NIMF_CPU_HIDDEN void nimf_cpu_rope(float* x, float* cos_t, float* sin_t,
                   int T, int H, int D);

/* CPU micro-VM.  in_f32 may be NULL to read from mem.  Returns 0 on success. */
NIMF_CPU_HIDDEN int nimf_cpu_vm_run(uint32_t* code, int n_words, float* mem, int mem_len,
                    float* in_f32, int in_len, int n_threads,
                    int tg_floats, int tg_size, char* err, int err_len);

#ifdef __cplusplus
}
#endif
#endif
