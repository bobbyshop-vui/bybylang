/* CPU kernel implementation template.
 *
 * Included once per ISA with CPU_SUFFIX and CPU_ATTR defined.  Plain loops
 * are intentional: with -O3 and a target attribute the compiler emits the
 * corresponding vector ISA (SSE4/AVX/AVX2+FMA/AVX-512/NEON).
 */
#ifndef CPU_SUFFIX
#error "CPU_SUFFIX must be defined before including cpu_kernels_impl.inc"
#endif
#ifndef CPU_ATTR
#define CPU_ATTR
#endif

#define NIMF_CAT2_(a, b) a##b
#define NIMF_CAT2(a, b) NIMF_CAT2_(a, b)
#define CPU_FN(name) NIMF_CAT2(nimf_cpu_##name, CPU_SUFFIX)

#ifndef NIMF_RESTRICT
#if defined(__cplusplus)
#define NIMF_RESTRICT
#else
#define NIMF_RESTRICT restrict
#endif
#endif

static CPU_ATTR void CPU_FN(add)(const float* NIMF_RESTRICT a,
                          const float* NIMF_RESTRICT b,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
static CPU_ATTR void CPU_FN(sub)(const float* NIMF_RESTRICT a,
                          const float* NIMF_RESTRICT b,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}
static CPU_ATTR void CPU_FN(mul)(const float* NIMF_RESTRICT a,
                          const float* NIMF_RESTRICT b,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
static CPU_ATTR void CPU_FN(div)(const float* NIMF_RESTRICT a,
                          const float* NIMF_RESTRICT b,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}
static CPU_ATTR void CPU_FN(relu)(const float* NIMF_RESTRICT x,
                           float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
static CPU_ATTR void CPU_FN(sqrt)(const float* NIMF_RESTRICT x,
                           float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = sqrtf(x[i]);
}
static CPU_ATTR void CPU_FN(exp)(const float* NIMF_RESTRICT x,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = expf(x[i]);
}
static CPU_ATTR void CPU_FN(log)(const float* NIMF_RESTRICT x,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = logf(x[i]);
}
static CPU_ATTR void CPU_FN(sigmoid)(const float* NIMF_RESTRICT x,
                              float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + expf(-x[i]));
}
static CPU_ATTR void CPU_FN(tanh)(const float* NIMF_RESTRICT x,
                           float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = tanhf(x[i]);
}
static CPU_ATTR void CPU_FN(neg)(const float* NIMF_RESTRICT x,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = -x[i];
}
static CPU_ATTR void CPU_FN(abs)(const float* NIMF_RESTRICT x,
                          float* NIMF_RESTRICT out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = fabsf(x[i]);
}
static CPU_ATTR void CPU_FN(pow)(const float* NIMF_RESTRICT x,
                          float* NIMF_RESTRICT out, size_t n, float p) {
    for (size_t i = 0; i < n; ++i) out[i] = powf(x[i], p);
}
static CPU_ATTR void CPU_FN(add_scalar)(const float* NIMF_RESTRICT x,
                                 float* NIMF_RESTRICT out, size_t n, float s) {
    for (size_t i = 0; i < n; ++i) out[i] = x[i] + s;
}
static CPU_ATTR void CPU_FN(mul_scalar)(const float* NIMF_RESTRICT x,
                                 float* NIMF_RESTRICT out, size_t n, float s) {
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * s;
}
static CPU_ATTR void CPU_FN(apflu)(const float* NIMF_RESTRICT x,
                            float* NIMF_RESTRICT out, size_t n,
                            float alpha, float beta) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = v > 0.0f ? v * (1.0f + alpha * v) : beta * v * expf(v);
    }
}
static CPU_ATTR void CPU_FN(apflu_backward)(const float* NIMF_RESTRICT x,
                                     const float* NIMF_RESTRICT dy,
                                     float* NIMF_RESTRICT out, size_t n,
                                     float alpha, float beta) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = v > 0.0f
            ? dy[i] * (1.0f + 2.0f * alpha * v)
            : dy[i] * beta * expf(v) * (1.0f + v);
    }
}
static CPU_ATTR void CPU_FN(fused_add_act)(const float* NIMF_RESTRICT a,
                                    const float* NIMF_RESTRICT b,
                                    float* NIMF_RESTRICT out, size_t n,
                                    int act) {
    for (size_t i = 0; i < n; ++i) {
        float s = a[i] + b[i];
        switch (act) {
        case 1: s = s > 0.0f ? s : 0.0f; break;
        case 2: s = 1.0f / (1.0f + expf(-s)); break;
        case 3: s = tanhf(s); break;
        default: break;
        }
        out[i] = s;
    }
}
static CPU_ATTR void CPU_FN(copy)(float* NIMF_RESTRICT dst,
                           const float* NIMF_RESTRICT src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}
static CPU_ATTR void CPU_FN(scale)(float* NIMF_RESTRICT x, size_t n, float s) {
    for (size_t i = 0; i < n; ++i) x[i] *= s;
}

static CPU_ATTR void CPU_FN(softmax)(const float* NIMF_RESTRICT x,
                              float* NIMF_RESTRICT out, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * (size_t)cols;
        float* orow = out + (size_t)r * (size_t)cols;
        float mx = xr[0];
        for (int c = 1; c < cols; ++c) if (xr[c] > mx) mx = xr[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(xr[c] - mx);
            orow[c] = e;
            sum += e;
        }
        float inv = sum != 0.0f ? 1.0f / sum : 0.0f;
        for (int c = 0; c < cols; ++c) orow[c] *= inv;
    }
}

static CPU_ATTR void CPU_FN(layernorm)(const float* NIMF_RESTRICT x,
                                const float* NIMF_RESTRICT gamma,
                                const float* NIMF_RESTRICT beta,
                                float* NIMF_RESTRICT out,
                                int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * (size_t)cols;
        float* orow = out + (size_t)r * (size_t)cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= (float)cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float inv = 1.0f / sqrtf(var + eps);
        for (int c = 0; c < cols; ++c)
            orow[c] = (xr[c] - mean) * inv * gamma[c] + beta[c];
    }
}

static CPU_ATTR void CPU_FN(rmsnorm)(const float* NIMF_RESTRICT x,
                              const float* NIMF_RESTRICT gamma,
                              float* NIMF_RESTRICT out,
                              int rows, int cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * (size_t)cols;
        float* orow = out + (size_t)r * (size_t)cols;
        float ss = 0.0f;
        for (int c = 0; c < cols; ++c) ss += xr[c] * xr[c];
        ss /= (float)cols;
        float inv = 1.0f / sqrtf(ss + eps);
        for (int c = 0; c < cols; ++c) orow[c] = xr[c] * inv * gamma[c];
    }
}

static CPU_ATTR void CPU_FN(layernorm_backward)(const float* NIMF_RESTRICT dy,
                                         const float* NIMF_RESTRICT x,
                                         const float* NIMF_RESTRICT gamma,
                                         const float* NIMF_RESTRICT beta,
                                         int rows, int cols, float eps,
                                         float* NIMF_RESTRICT dx,
                                         float* NIMF_RESTRICT dgamma,
                                         float* NIMF_RESTRICT dbeta) {
    (void)beta;
    if (rows <= 0 || cols <= 0) return;
    for (int c = 0; c < cols; ++c) { dgamma[c] = 0.0f; dbeta[c] = 0.0f; }
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t)r * (size_t)cols;
        const float* dr = dy + (size_t)r * (size_t)cols;
        float* dxr = dx + (size_t)r * (size_t)cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= (float)cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = xr[c] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float inv = 1.0f / sqrtf(var + eps);
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float norm = (xr[c] - mean) * inv;
            dgamma[c] += dr[c] * norm;
            dbeta[c] += dr[c];
            float g = dr[c] * gamma[c] * inv;
            sum1 += g;
            sum2 += g * (xr[c] - mean);
        }
        sum2 *= -inv * inv / (float)cols;
        for (int c = 0; c < cols; ++c) {
            float term1 = dr[c] * gamma[c] * inv;
            float term2 = sum1 / (float)cols;
            float term3 = (xr[c] - mean) * sum2;
            dxr[c] = term1 - term2 + term3;
        }
    }
}

static CPU_ATTR void CPU_FN(matmul)(const float* NIMF_RESTRICT a,
                             const float* NIMF_RESTRICT b,
                             float* NIMF_RESTRICT c,
                             int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0) return;
    int mc = nimf_cpu_block_mc;
    int nc = nimf_cpu_block_nc;
    int kc = nimf_cpu_block_kc;
    if (mc <= 0) mc = 32;
    if (nc <= 0) nc = 256;
    if (kc <= 0) kc = 256;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) c[(size_t)i * N + j] = 0.0f;
    for (int jc = 0; jc < N; jc += nc) {
        int jn = (jc + nc <= N) ? nc : N - jc;
        for (int kc0 = 0; kc0 < K; kc0 += kc) {
            int kn = (kc0 + kc <= K) ? kc : K - kc0;
            for (int ic = 0; ic < M; ic += mc) {
                int im = (ic + mc <= M) ? mc : M - ic;
                for (int i = 0; i < im; ++i) {
                    const float* arow = a + (size_t)(ic + i) * K + kc0;
                    float* crow = c + (size_t)(ic + i) * N + jc;
                    for (int k = 0; k < kn; ++k) {
                        float av = arow[k];
                        const float* brow = b + (size_t)(kc0 + k) * N + jc;
                        for (int j = 0; j < jn; ++j) crow[j] += av * brow[j];
                    }
                }
            }
        }
    }
}

static CPU_ATTR void CPU_FN(matmul_nt)(const float* NIMF_RESTRICT a,
                                const float* NIMF_RESTRICT b,
                                float* NIMF_RESTRICT c,
                                int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0) return;
    for (int i = 0; i < M; ++i) {
        const float* arow = a + (size_t)i * K;
        for (int j = 0; j < N; ++j) {
            const float* brow = b + (size_t)j * K;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int k = 0;
            for (; k + 3 < K; k += 4) {
                s0 += arow[k]     * brow[k];
                s1 += arow[k + 1] * brow[k + 1];
                s2 += arow[k + 2] * brow[k + 2];
                s3 += arow[k + 3] * brow[k + 3];
            }
            float s = (s0 + s1) + (s2 + s3);
            for (; k < K; ++k) s += arow[k] * brow[k];
            c[(size_t)i * N + j] = s;
        }
    }
}

static CPU_ATTR void CPU_FN(matmul_q4)(const float* NIMF_RESTRICT a,
                                const uint8_t* NIMF_RESTRICT wq,
                                const float* NIMF_RESTRICT scales,
                                const float* NIMF_RESTRICT zeros,
                                float* NIMF_RESTRICT c,
                                int M, int K, int N,
                                int group_size, int groups_per_row) {
    if (M <= 0 || K <= 0 || N <= 0) return;
    int bytes_per_row = (K + 1) / 2;
    for (int m = 0; m < M; ++m) {
        const float* arow = a + (size_t)m * K;
        for (int n = 0; n < N; ++n) {
            const uint8_t* wrow = wq + (size_t)n * bytes_per_row;
            int rg = n * groups_per_row;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int g = group_size > 0 ? (k / group_size) : 0;
                float sc = scales[rg + g];
                float zp = zeros[rg + g];
                uint8_t byte = wrow[k >> 1];
                uint8_t nib = (k & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
                sum += arow[k] * ((float)nib - zp) * sc;
            }
            c[(size_t)m * N + n] = sum;
        }
    }
}

static CPU_ATTR void CPU_FN(attention)(const float* NIMF_RESTRICT q,
                                const float* NIMF_RESTRICT k,
                                const float* NIMF_RESTRICT v,
                                const float* NIMF_RESTRICT mask, int mask_len,
                                float* NIMF_RESTRICT out,
                                float* NIMF_RESTRICT scores,
                                int B, int H, int S, int D, float scale) {
    (void)mask; (void)mask_len;
    if (B <= 0 || H <= 0 || S <= 0 || D <= 0) return;
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            size_t base = ((size_t)b * H + h) * S * D;
            size_t sbase = ((size_t)b * H + h) * S * S;
            const float* qh = q + base;
            const float* kh = k + base;
            const float* vh = v + base;
            float* oh = out + base;
            float* sh = scores + sbase;
            for (int i = 0; i < S; ++i) {
                float mx = -INFINITY;
                for (int j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d)
                        dot += qh[(size_t)i * D + d] * kh[(size_t)j * D + d];
                    float s = dot * scale;
                    sh[(size_t)i * S + j] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float e = expf(sh[(size_t)i * S + j] - mx);
                    sh[(size_t)i * S + j] = e;
                    sum += e;
                }
                float inv = sum != 0.0f ? 1.0f / sum : 0.0f;
                for (int j = 0; j <= i; ++j) sh[(size_t)i * S + j] *= inv;
                for (int j = i + 1; j < S; ++j) sh[(size_t)i * S + j] = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; ++j)
                        acc += sh[(size_t)i * S + j] * vh[(size_t)j * D + d];
                    oh[(size_t)i * D + d] = acc;
                }
            }
        }
    }
}

static CPU_ATTR void CPU_FN(embedding)(const float* NIMF_RESTRICT table,
                                const int32_t* NIMF_RESTRICT indices,
                                float* NIMF_RESTRICT out,
                                int num_idx, int vocab, int dim) {
    for (int i = 0; i < num_idx; ++i) {
        int idx = indices[i];
        float* dst = out + (size_t)i * dim;
        if (idx >= 0 && idx < vocab) {
            const float* src = table + (size_t)idx * dim;
            for (int d = 0; d < dim; ++d) dst[d] = src[d];
        } else {
            for (int d = 0; d < dim; ++d) dst[d] = 0.0f;
        }
    }
}

static CPU_ATTR void CPU_FN(gather)(const float* NIMF_RESTRICT src,
                             const int32_t* NIMF_RESTRICT idx,
                             float* NIMF_RESTRICT out, int n) {
    for (int i = 0; i < n; ++i) out[i] = src[(int)idx[i]];
}

static CPU_ATTR void CPU_FN(compare)(const float* NIMF_RESTRICT a,
                              const float* NIMF_RESTRICT b,
                              uint8_t* NIMF_RESTRICT mask, int n, int op) {
    for (int i = 0; i < n; ++i) {
        int r = 0;
        switch (op) {
        case 0: r = a[i] > b[i]; break;
        case 1: r = a[i] < b[i]; break;
        case 2: r = a[i] == b[i]; break;
        case 3: r = a[i] >= b[i]; break;
        default: r = a[i] <= b[i]; break;
        }
        mask[i] = (uint8_t)(r ? 1 : 0);
    }
}

static CPU_ATTR void CPU_FN(dropout_mask)(float* NIMF_RESTRICT out, int n,
                                   float p, uint32_t seed) {
    float scale = p < 1.0f ? 1.0f / (1.0f - p) : 0.0f;
    for (int i = 0; i < n; ++i) {
        uint32_t state = seed ^ ((uint32_t)i * 2654435761u + 1u);
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        float u = (float)(state >> 8) * (1.0f / 16777216.0f);
        out[i] = u < p ? 0.0f : scale;
    }
}

static CPU_ATTR void CPU_FN(argmax)(const float* NIMF_RESTRICT x,
                             float* NIMF_RESTRICT out_val,
                             int32_t* NIMF_RESTRICT out_idx,
                             int outer, int dim_size, int inner) {
    for (int o = 0; o < outer; ++o) {
        for (int i = 0; i < inner; ++i) {
            float best = x[(size_t)(o * dim_size) * inner + i];
            int best_i = 0;
            for (int d = 1; d < dim_size; ++d) {
                float v = x[(size_t)(o * dim_size + d) * inner + i];
                if (v > best) { best = v; best_i = d; }
            }
            out_val[(size_t)o * inner + i] = best;
            out_idx[(size_t)o * inner + i] = (int32_t)best_i;
        }
    }
}

static CPU_ATTR void CPU_FN(permute)(const float* NIMF_RESTRICT src,
                              float* NIMF_RESTRICT out,
                              const int32_t* NIMF_RESTRICT new_strides,
                              const int32_t* NIMF_RESTRICT old_strides,
                              const int32_t* NIMF_RESTRICT dims_map,
                              int ndim, int n) {
    for (int linear = 0; linear < n; ++linear) {
        int rem = linear;
        int old_linear = 0;
        for (int i = 0; i < ndim; ++i) {
            int idx = rem / new_strides[i];
            rem = rem % new_strides[i];
            old_linear += idx * old_strides[dims_map[i]];
        }
        out[linear] = src[old_linear];
    }
}

static CPU_ATTR void CPU_FN(split_slice)(const float* NIMF_RESTRICT src,
                                  float* NIMF_RESTRICT out,
                                  int dim_total, int this_size, int offset,
                                  int inner, int n) {
    for (int linear = 0; linear < n; ++linear) {
        int i = linear % inner;
        int rem = linear / inner;
        int d = rem % this_size;
        int o = rem / this_size;
        out[linear] = src[(size_t)((o * dim_total + offset + d) * inner + i)];
    }
}

static CPU_ATTR void CPU_FN(concat)(const float* NIMF_RESTRICT flat_src,
                             float* NIMF_RESTRICT out,
                             const int32_t* NIMF_RESTRICT src_base,
                             const int32_t* NIMF_RESTRICT src_dim_size,
                             const int32_t* NIMF_RESTRICT src_dim_offset,
                             int num_tensors, int total_dim, int inner, int n) {
    for (int linear = 0; linear < n; ++linear) {
        int i = linear % inner;
        int rem = linear / inner;
        int dg = rem % total_dim;
        int o = rem / total_dim;
        int ti = 0;
        for (int k = 0; k < num_tensors; ++k) {
            if (dg < src_dim_offset[k] + src_dim_size[k]) { ti = k; break; }
        }
        int dl = dg - src_dim_offset[ti];
        out[linear] = flat_src[(size_t)src_base[ti] +
            (size_t)(o * src_dim_size[ti] + dl) * inner + i];
    }
}

static CPU_ATTR void CPU_FN(rope)(float* NIMF_RESTRICT x,
                           const float* NIMF_RESTRICT cos_t,
                           const float* NIMF_RESTRICT sin_t,
                           int T, int H, int D) {
    int half = D / 2;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            float* base = x + ((size_t)t * H + h) * D;
            for (int i = 0; i < half; ++i) {
                float c = cos_t[(size_t)t * half + i];
                float s = sin_t[(size_t)t * half + i];
                float x1 = base[i];
                float x2 = base[i + half];
                base[i] = x1 * c - x2 * s;
                base[i + half] = x2 * c + x1 * s;
            }
        }
    }
}

#undef CPU_FN
#undef CPU_ATTR
#undef CPU_SUFFIX
