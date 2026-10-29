/* Nimformers CPU backend dispatcher and scalar kernels. */
#include "cpu_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#define NIMF_CPU_X86 1
#endif
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)
#define NIMF_CPU_ARM 1
#endif

/* Tunables used by the implementation template (matmul blocking). */
static int nimf_cpu_block_mc = 32;
static int nimf_cpu_block_nc = 256;
static int nimf_cpu_block_kc = 256;

static NimfCpuInfo g_info;
static int g_inited = 0;
static int g_forced_isa = -1;

/* Scalar variants. */
#define CPU_SUFFIX _base
#define CPU_ATTR
#include "cpu_kernels_impl.inc"

#if defined(NIMF_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
#define CPU_SUFFIX _sse
#define CPU_ATTR __attribute__((target("sse4.1")))
#include "cpu_kernels_impl.inc"

#define CPU_SUFFIX _avx
#define CPU_ATTR __attribute__((target("avx")))
#include "cpu_kernels_impl.inc"

#define CPU_SUFFIX _avx2
#define CPU_ATTR __attribute__((target("avx2,fma")))
#include "cpu_kernels_impl.inc"

#define CPU_SUFFIX _avx512
#define CPU_ATTR __attribute__((target("avx512f")))
#include "cpu_kernels_impl.inc"
#endif

#if defined(NIMF_CPU_ARM)
#define CPU_SUFFIX _neon
#define CPU_ATTR
#include "cpu_kernels_impl.inc"
#endif

/* ------------------------------------------------------------------ */
/* Function tables                                                     */
/* ------------------------------------------------------------------ */

typedef void (*BinFn)(const float*, const float*, float*, size_t);
typedef void (*UnFn)(const float*, float*, size_t);
typedef void (*PowFn)(const float*, float*, size_t, float);
typedef void (*ScalarFn)(const float*, float*, size_t, float);
typedef void (*ApfluFn)(const float*, float*, size_t, float, float);
typedef void (*ApfluBwdFn)(const float*, const float*, float*, size_t, float, float);
typedef void (*FusedFn)(const float*, const float*, float*, size_t, int);
typedef void (*CopyFn)(float*, const float*, size_t);
typedef void (*ScaleFn)(float*, size_t, float);
typedef void (*SoftmaxFn)(const float*, float*, int, int);
typedef void (*NormFn)(const float*, const float*, const float*, float*, int, int, float);
typedef void (*RmsFn)(const float*, const float*, float*, int, int, float);
typedef void (*NormBwdFn)(const float*, const float*, const float*, const float*,
                          int, int, float, float*, float*, float*);
typedef void (*MatmulFn)(const float*, const float*, float*, int, int, int);
typedef void (*MatmulNTFn)(const float*, const float*, float*, int, int, int);
typedef void (*MatmulQ4Fn)(const float*, const uint8_t*, const float*, const float*,
                           float*, int, int, int, int, int);
typedef void (*AttentionFn)(const float*, const float*, const float*, const float*,
                            int, float*, float*, int, int, int, int, float);
typedef void (*EmbeddingFn)(const float*, const int32_t*, float*, int, int, int);
typedef void (*GatherFn)(const float*, const int32_t*, float*, int);
typedef void (*CompareFn)(const float*, const float*, uint8_t*, int, int);
typedef void (*DropoutFn)(float*, int, float, uint32_t);
typedef void (*ArgmaxFn)(const float*, float*, int32_t*, int, int, int);
typedef void (*PermuteFn)(const float*, float*, const int32_t*, const int32_t*,
                          const int32_t*, int, int);
typedef void (*SplitFn)(const float*, float*, int, int, int, int, int);
typedef void (*ConcatFn)(const float*, float*, const int32_t*, const int32_t*,
                         const int32_t*, int, int, int, int);
typedef void (*RopeFn)(float*, const float*, const float*, int, int, int);

static BinFn g_add[5], g_sub[5], g_mul[5], g_div[5];
static UnFn g_relu[5], g_sqrt[5], g_exp[5], g_log[5], g_sigmoid[5], g_tanh[5], g_neg[5], g_abs[5];
static PowFn g_pow[5];
static ScalarFn g_add_scalar[5], g_mul_scalar[5];
static ApfluFn g_apflu[5];
static ApfluBwdFn g_apflu_bwd[5];
static FusedFn g_fused[5];
static CopyFn g_copy[5];
static ScaleFn g_scale[5];
static SoftmaxFn g_softmax[5];
static NormFn g_layernorm[5];
static RmsFn g_rmsnorm[5];
static NormBwdFn g_layernorm_bwd[5];
static MatmulFn g_matmul[5], g_matmul_nt[5];
static MatmulQ4Fn g_matmul_q4[5];
static AttentionFn g_attention[5];
static EmbeddingFn g_embedding[5];
static GatherFn g_gather[5];
static CompareFn g_compare[5];
static DropoutFn g_dropout[5];
static ArgmaxFn g_argmax[5];
static PermuteFn g_permute[5];
static SplitFn g_split[5];
static ConcatFn g_concat[5];
static RopeFn g_rope[5];

#define SET1(table, name) (table)[0] = nimf_cpu_##name##_base
#if defined(NIMF_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
#define SETX(table, name) do { \
    (table)[1] = nimf_cpu_##name##_sse; \
    (table)[2] = nimf_cpu_##name##_avx; \
    (table)[3] = nimf_cpu_##name##_avx2; \
    (table)[4] = nimf_cpu_##name##_avx512; \
} while (0)
#elif defined(NIMF_CPU_ARM)
#define SETX(table, name) do { (table)[1] = nimf_cpu_##name##_neon; } while (0)
#else
#define SETX(table, name) do { } while (0)
#endif

static int clamp_isa(int isa) {
    if (isa <= 0) return 0;
    if (isa >= 1 && !(g_info.has_sse4 || g_info.has_neon)) isa = 0;
    if (isa >= 2 && !g_info.has_avx) isa = g_info.has_neon ? 1 : 0;
    if (isa >= 3 && !g_info.has_avx2) isa = g_info.has_avx ? 2 : (g_info.has_neon ? 1 : 0);
    if (isa >= 4 && !g_info.has_avx512) isa = g_info.has_avx2 ? 3 : (g_info.has_avx ? 2 : (g_info.has_neon ? 1 : 0));
    return isa;
}

static int pick_isa(size_t n, size_t work) {
    if (g_forced_isa >= 0) return clamp_isa(g_forced_isa);
    if (n == 0) return 0;
    int small = g_info.small_threshold > 0 ? g_info.small_threshold : 4096;
    int avx512_min = g_info.avx512_threshold > 0 ? g_info.avx512_threshold : 131072;
#if defined(NIMF_CPU_ARM)
    (void)work; (void)avx512_min;
    if (g_info.has_neon) return 1;
    return 0;
#else
    if (n <= (size_t)small) return g_info.has_sse4 ? 1 : 0;
    if (g_info.has_avx512 && (work >= (size_t)avx512_min || n >= (size_t)avx512_min)) return 4;
    if (g_info.has_avx2) return 3;
    if (g_info.has_avx) return 2;
    if (g_info.has_sse4) return 1;
    return 0;
#endif
}

static int pick_matmul(size_t work) {
    if (g_forced_isa >= 0) return clamp_isa(g_forced_isa);
#if defined(NIMF_CPU_ARM)
    (void)work;
    return g_info.has_neon ? 1 : 0;
#else
    int small = g_info.small_threshold > 0 ? g_info.small_threshold : 4096;
    if (work <= (size_t)small * 32) return g_info.has_sse4 ? 1 : 0;
    if (g_info.has_avx512 && work >= (size_t)g_info.avx512_threshold * 64) return 4;
    if (g_info.has_avx2) return 3;
    if (g_info.has_avx) return 2;
    if (g_info.has_sse4) return 1;
    return 0;
#endif
}

static void init_tables(void) {
    SET1(g_add, add); SETX(g_add, add);
    SET1(g_sub, sub); SETX(g_sub, sub);
    SET1(g_mul, mul); SETX(g_mul, mul);
    SET1(g_div, div); SETX(g_div, div);
    SET1(g_relu, relu); SETX(g_relu, relu);
    SET1(g_sqrt, sqrt); SETX(g_sqrt, sqrt);
    SET1(g_exp, exp); SETX(g_exp, exp);
    SET1(g_log, log); SETX(g_log, log);
    SET1(g_sigmoid, sigmoid); SETX(g_sigmoid, sigmoid);
    SET1(g_tanh, tanh); SETX(g_tanh, tanh);
    SET1(g_neg, neg); SETX(g_neg, neg);
    SET1(g_abs, abs); SETX(g_abs, abs);
    SET1(g_pow, pow); SETX(g_pow, pow);
    SET1(g_add_scalar, add_scalar); SETX(g_add_scalar, add_scalar);
    SET1(g_mul_scalar, mul_scalar); SETX(g_mul_scalar, mul_scalar);
    SET1(g_apflu, apflu); SETX(g_apflu, apflu);
    SET1(g_apflu_bwd, apflu_backward); SETX(g_apflu_bwd, apflu_backward);
    SET1(g_fused, fused_add_act); SETX(g_fused, fused_add_act);
    SET1(g_copy, copy); SETX(g_copy, copy);
    SET1(g_scale, scale); SETX(g_scale, scale);
    SET1(g_softmax, softmax); SETX(g_softmax, softmax);
    SET1(g_layernorm, layernorm); SETX(g_layernorm, layernorm);
    SET1(g_rmsnorm, rmsnorm); SETX(g_rmsnorm, rmsnorm);
    SET1(g_layernorm_bwd, layernorm_backward); SETX(g_layernorm_bwd, layernorm_backward);
    SET1(g_matmul, matmul); SETX(g_matmul, matmul);
    SET1(g_matmul_nt, matmul_nt); SETX(g_matmul_nt, matmul_nt);
    SET1(g_matmul_q4, matmul_q4); SETX(g_matmul_q4, matmul_q4);
    SET1(g_attention, attention); SETX(g_attention, attention);
    SET1(g_embedding, embedding); SETX(g_embedding, embedding);
    SET1(g_gather, gather); SETX(g_gather, gather);
    SET1(g_compare, compare); SETX(g_compare, compare);
    SET1(g_dropout, dropout_mask); SETX(g_dropout, dropout_mask);
    SET1(g_argmax, argmax); SETX(g_argmax, argmax);
    SET1(g_permute, permute); SETX(g_permute, permute);
    SET1(g_split, split_slice); SETX(g_split, split_slice);
    SET1(g_concat, concat); SETX(g_concat, concat);
    SET1(g_rope, rope); SETX(g_rope, rope);
}

static size_t env_size(const char* name, size_t fallback) {
    const char* s = getenv(name);
    if (!s || !*s) return fallback;
    long v = strtol(s, NULL, 10);
    return v > 0 ? (size_t)v : fallback;
}

static int env_int(const char* name, int fallback) {
    const char* s = getenv(name);
    if (!s || !*s) return fallback;
    return (int)strtol(s, NULL, 10);
}

static size_t cache_bytes_apple(const char* name, size_t fallback) {
#if defined(__APPLE__)
    uint64_t v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, NULL, 0) == 0 && v > 0) return (size_t)v;
#else
    (void)name;
#endif
    return fallback;
}

#if defined(__linux__)
static size_t cache_bytes_linux(int level, size_t fallback) {
    long v = 0;
    switch (level) {
    case 1: v = sysconf(_SC_LEVEL1_DCACHE_SIZE); break;
    case 2: v = sysconf(_SC_LEVEL2_CACHE_SIZE); break;
    case 3: v = sysconf(_SC_LEVEL3_CACHE_SIZE); break;
    default: break;
    }
    return v > 0 ? (size_t)v : fallback;
}
#endif

void nimf_cpu_init(void) {
    if (g_inited) return;
    memset(&g_info, 0, sizeof(g_info));
    g_info.l1_bytes = 32 * 1024;
    g_info.l2_bytes = 256 * 1024;
    g_info.l3_bytes = 8 * 1024 * 1024;
#if defined(__APPLE__)
    g_info.l1_bytes = cache_bytes_apple("hw.l1dcachesize", g_info.l1_bytes);
    g_info.l2_bytes = cache_bytes_apple("hw.l2cachesize", g_info.l2_bytes);
    g_info.l3_bytes = cache_bytes_apple("hw.l3cachesize", g_info.l3_bytes);
#elif defined(__linux__)
    g_info.l1_bytes = cache_bytes_linux(1, g_info.l1_bytes);
    g_info.l2_bytes = cache_bytes_linux(2, g_info.l2_bytes);
    g_info.l3_bytes = cache_bytes_linux(3, g_info.l3_bytes);
#endif
#if defined(NIMF_CPU_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    g_info.has_sse4 = __builtin_cpu_supports("sse4.1") ? 1 : 0;
    g_info.has_avx = __builtin_cpu_supports("avx") ? 1 : 0;
    g_info.has_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
    g_info.has_fma = __builtin_cpu_supports("fma") ? 1 : 0;
    g_info.has_avx512 = __builtin_cpu_supports("avx512f") ? 1 : 0;
#endif
#if defined(NIMF_CPU_ARM)
    g_info.has_neon = 1;
#endif
    if (g_info.has_avx512) g_info.isa = 4;
    else if (g_info.has_avx2) g_info.isa = 3;
    else if (g_info.has_avx) g_info.isa = 2;
    else if (g_info.has_sse4 || g_info.has_neon) g_info.isa = 1;
    else g_info.isa = 0;
    g_info.small_threshold = env_int("NIMFORMERS_CPU_SMALL", 4096);
    g_info.avx512_threshold = env_int("NIMFORMERS_CPU_AVX512_MIN", 131072);

    const char* force = getenv("NIMFORMERS_CPU_ISA");
    if (force && *force) {
        if (!strcmp(force, "scalar")) g_forced_isa = 0;
        else if (!strcmp(force, "sse") || !strcmp(force, "sse4")) g_forced_isa = 1;
        else if (!strcmp(force, "avx")) g_forced_isa = 2;
        else if (!strcmp(force, "avx2")) g_forced_isa = 3;
        else if (!strcmp(force, "avx512")) g_forced_isa = 4;
    } else {
        g_forced_isa = -1;
    }

    /* Cache-aware GEMM blocking: keep the B panel in L3 and the C block in L2. */
    int nc, kc, mc;
    if (g_info.l3_bytes >= 32u * 1024u * 1024u) nc = 1024;
    else if (g_info.l3_bytes >= 16u * 1024u * 1024u) nc = 512;
    else if (g_info.l3_bytes >= 8u * 1024u * 1024u) nc = 384;
    else nc = 256;
    if (g_info.l2_bytes >= 1024u * 1024u) kc = 512;
    else if (g_info.l2_bytes >= 512u * 1024u) kc = 256;
    else kc = 128;
    mc = 32;
    /* Ensure the C tile (mc*nc*4) stays comfortably inside L2. */
    while (mc > 4 && (size_t)mc * (size_t)nc * sizeof(float) > g_info.l2_bytes / 2) mc /= 2;
    nimf_cpu_block_nc = (int)env_size("NIMFORMERS_CPU_BLOCK_NC", (size_t)nc);
    nimf_cpu_block_kc = (int)env_size("NIMFORMERS_CPU_BLOCK_KC", (size_t)kc);
    nimf_cpu_block_mc = (int)env_size("NIMFORMERS_CPU_BLOCK_MC", (size_t)mc);
    init_tables();
    g_inited = 1;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void nimf_cpu_ctor(void) { nimf_cpu_init(); }
#endif

NimfCpuInfo* nimf_cpu_info(void) { nimf_cpu_init(); return &g_info; }
int nimf_cpu_current_isa(void) { nimf_cpu_init(); return g_info.isa; }

void nimf_cpu_backend_block_sizes(int* mc, int* nc, int* kc) {
    nimf_cpu_init();
    if (mc) *mc = nimf_cpu_block_mc;
    if (nc) *nc = nimf_cpu_block_nc;
    if (kc) *kc = nimf_cpu_block_kc;
}

const char* nimf_cpu_backend_isa_name(int isa) {
    switch (isa) {
    case 0: return "scalar";
    case 1: return "sse4";
    case 2: return "avx";
    case 3: return "avx2";
    case 4: return "avx512";
    default: return "unknown";
    }
}

/* ------------------------------------------------------------------ */
/* Public dispatchers                                                  */
/* ------------------------------------------------------------------ */

#define WRAP_BIN(name) \
void nimf_cpu_##name(float* a, float* b, float* out, size_t n) { \
    nimf_cpu_init(); g_##name[pick_isa(n, n)](a, b, out, n); }
#define WRAP_UN(name) \
void nimf_cpu_##name(float* x, float* out, size_t n) { \
    nimf_cpu_init(); g_##name[pick_isa(n, n)](x, out, n); }

WRAP_BIN(add) WRAP_BIN(sub) WRAP_BIN(mul) WRAP_BIN(div)
WRAP_UN(relu) WRAP_UN(sqrt) WRAP_UN(exp) WRAP_UN(log)
WRAP_UN(sigmoid) WRAP_UN(tanh) WRAP_UN(neg) WRAP_UN(abs)

void nimf_cpu_pow(float* x, float* out, size_t n, float p) {
    nimf_cpu_init(); g_pow[pick_isa(n, n)](x, out, n, p);
}
void nimf_cpu_add_scalar(float* x, float* out, size_t n, float s) {
    nimf_cpu_init(); g_add_scalar[pick_isa(n, n)](x, out, n, s);
}
void nimf_cpu_mul_scalar(float* x, float* out, size_t n, float s) {
    nimf_cpu_init(); g_mul_scalar[pick_isa(n, n)](x, out, n, s);
}
void nimf_cpu_apflu(float* x, float* out, size_t n, float alpha, float beta) {
    nimf_cpu_init(); g_apflu[pick_isa(n, n)](x, out, n, alpha, beta);
}
void nimf_cpu_apflu_backward(float* x, float* dy, float* out,
                             size_t n, float alpha, float beta) {
    nimf_cpu_init(); g_apflu_bwd[pick_isa(n, n)](x, dy, out, n, alpha, beta);
}
void nimf_cpu_fused_add_act(float* a, float* b, float* out, size_t n, int act) {
    nimf_cpu_init(); g_fused[pick_isa(n, n)](a, b, out, n, act);
}
void nimf_cpu_copy(float* dst, float* src, size_t n) {
    nimf_cpu_init(); g_copy[pick_isa(n, n)](dst, src, n);
}
void nimf_cpu_scale(float* x, size_t n, float s) {
    nimf_cpu_init(); g_scale[pick_isa(n, n)](x, n, s);
}
void nimf_cpu_softmax(float* x, float* out, int rows, int cols) {
    nimf_cpu_init();
    size_t work = (size_t)(rows > 0 ? rows : 0) * (size_t)(cols > 0 ? cols : 0);
    g_softmax[pick_isa(work, work)](x, out, rows, cols);
}
void nimf_cpu_layernorm(float* x, float* gamma, float* beta,
                        float* out, int rows, int cols, float eps) {
    nimf_cpu_init();
    size_t work = (size_t)(rows > 0 ? rows : 0) * (size_t)(cols > 0 ? cols : 0);
    g_layernorm[pick_isa(work, work)](x, gamma, beta, out, rows, cols, eps);
}
void nimf_cpu_rmsnorm(float* x, float* gamma, float* out,
                      int rows, int cols, float eps) {
    nimf_cpu_init();
    size_t work = (size_t)(rows > 0 ? rows : 0) * (size_t)(cols > 0 ? cols : 0);
    g_rmsnorm[pick_isa(work, work)](x, gamma, out, rows, cols, eps);
}
void nimf_cpu_layernorm_backward(float* dy, float* x,
                                 float* gamma, float* beta,
                                 int rows, int cols, float eps,
                                 float* dx, float* dgamma, float* dbeta) {
    nimf_cpu_init();
    size_t work = (size_t)(rows > 0 ? rows : 0) * (size_t)(cols > 0 ? cols : 0);
    g_layernorm_bwd[pick_isa(work, work)](dy, x, gamma, beta, rows, cols, eps,
                                          dx, dgamma, dbeta);
}
void nimf_cpu_matmul(float* a, float* b, float* c, int M, int K, int N) {
    nimf_cpu_init();
    size_t work = (size_t)(M > 0 ? M : 0) * (size_t)(K > 0 ? K : 0) * (size_t)(N > 0 ? N : 0);
    g_matmul[pick_matmul(work)](a, b, c, M, K, N);
}
void nimf_cpu_matmul_nt(float* a, float* b, float* c, int M, int K, int N) {
    nimf_cpu_init();
    size_t work = (size_t)(M > 0 ? M : 0) * (size_t)(K > 0 ? K : 0) * (size_t)(N > 0 ? N : 0);
    g_matmul_nt[pick_matmul(work)](a, b, c, M, K, N);
}
void nimf_cpu_matmul_q4(float* a, uint8_t* wq, float* scales,
                        float* zeros, float* c, int M, int K, int N,
                        int group_size, int groups_per_row) {
    nimf_cpu_init();
    size_t work = (size_t)(M > 0 ? M : 0) * (size_t)(K > 0 ? K : 0) * (size_t)(N > 0 ? N : 0);
    g_matmul_q4[pick_matmul(work)](a, wq, scales, zeros, c, M, K, N,
                                   group_size, groups_per_row);
}
void nimf_cpu_attention(float* q, float* k, float* v,
                        float* mask, int mask_len, float* out, float* scores,
                        int B, int H, int S, int D, float scale) {
    nimf_cpu_init();
    size_t work = (size_t)(B > 0 ? B : 0) * (size_t)(H > 0 ? H : 0) *
                  (size_t)(S > 0 ? S : 0) * (size_t)(S > 0 ? S : 0) *
                  (size_t)(D > 0 ? D : 0);
    g_attention[pick_isa(work, work)](q, k, v, mask, mask_len, out, scores,
                                      B, H, S, D, scale);
}
void nimf_cpu_embedding(float* table, int32_t* indices, float* out,
                        int num_idx, int vocab, int dim) {
    nimf_cpu_init(); g_embedding[pick_isa((size_t)(num_idx > 0 ? num_idx : 0),
        (size_t)(num_idx > 0 ? num_idx : 0) * (size_t)(dim > 0 ? dim : 0))]
        (table, indices, out, num_idx, vocab, dim);
}
void nimf_cpu_gather(float* src, int32_t* idx, float* out, int n) {
    nimf_cpu_init(); g_gather[pick_isa((size_t)(n > 0 ? n : 0), (size_t)(n > 0 ? n : 0))](src, idx, out, n);
}
void nimf_cpu_compare(float* a, float* b, uint8_t* mask, int n, int op) {
    nimf_cpu_init(); g_compare[pick_isa((size_t)(n > 0 ? n : 0), (size_t)(n > 0 ? n : 0))](a, b, mask, n, op);
}
void nimf_cpu_dropout_mask(float* out, int n, float p, uint32_t seed) {
    nimf_cpu_init(); g_dropout[pick_isa((size_t)(n > 0 ? n : 0), (size_t)(n > 0 ? n : 0))](out, n, p, seed);
}
void nimf_cpu_argmax(float* x, float* out_val, int32_t* out_idx,
                     int outer, int dim_size, int inner) {
    nimf_cpu_init();
    size_t work = (size_t)(outer > 0 ? outer : 0) * (size_t)(inner > 0 ? inner : 0);
    g_argmax[pick_isa(work, work)](x, out_val, out_idx, outer, dim_size, inner);
}
void nimf_cpu_permute(float* src, float* out, int32_t* new_strides,
                      int32_t* old_strides, int32_t* dims_map,
                      int ndim, int n) {
    nimf_cpu_init(); g_permute[pick_isa((size_t)(n > 0 ? n : 0), (size_t)(n > 0 ? n : 0))]
        (src, out, new_strides, old_strides, dims_map, ndim, n);
}
void nimf_cpu_split_slice(float* src, float* out, int dim_total,
                          int this_size, int offset, int inner, int n) {
    nimf_cpu_init(); g_split[pick_isa((size_t)(n > 0 ? n : 0), (size_t)(n > 0 ? n : 0))]
        (src, out, dim_total, this_size, offset, inner, n);
}
void nimf_cpu_concat(float* flat_src, float* out, int32_t* src_base,
                     int32_t* src_dim_size, int32_t* src_dim_offset,
                     int num_tensors, int total_dim, int inner, int n) {
    nimf_cpu_init(); g_concat[pick_isa((size_t)(n > 0 ? n : 0), (size_t)(n > 0 ? n : 0))]
        (flat_src, out, src_base, src_dim_size, src_dim_offset, num_tensors,
         total_dim, inner, n);
}
void nimf_cpu_rope(float* x, float* cos_t, float* sin_t,
                   int T, int H, int D) {
    nimf_cpu_init();
    size_t work = (size_t)(T > 0 ? T : 0) * (size_t)(H > 0 ? H : 0) * (size_t)(D > 0 ? D : 0);
    g_rope[pick_isa(work, work)](x, cos_t, sin_t, T, H, D);
}
