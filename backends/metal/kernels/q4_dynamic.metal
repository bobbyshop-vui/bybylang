// Q4 asymmetric counterpart of the wide FP4 inference kernels. The packed
// bytes/scales/zeros retain their existing resident allocations and layout.
constant uint q4_dynamic_tg [[function_constant(100)]];
constant uint q4_dynamic_rows [[function_constant(101)]];
constant uint q4_dynamic_batch [[function_constant(102)]];

kernel void matmul_q4_dynamic(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]], // wq bound 0-3 bytes below the weights
    threadgroup float* partial [[threadgroup(0)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint n0 = gid.x * q4_dynamic_rows;
    const uint m0 = gid.y * q4_dynamic_batch;
    float accum[8]; // maximum rows*batch among the compiled variants
    for (uint i = 0; i < q4_dynamic_rows * q4_dynamic_batch; ++i) accum[i] = 0;
    // K is a multiple of eight. A lane owns one aligned word, including the
    // last partially populated workgroup for shapes such as K=8096.
    for (uint k = lane * 8; k < uint(K); k += q4_dynamic_tg * 8) {
        const uint group = groupSize > 0 ? k / uint(groupSize) : 0;
        for (uint r = 0; r < q4_dynamic_rows; ++r) {
            uint n = n0 + r;
            if (n >= uint(N)) continue;
            // Rows hold K/2 bytes (a multiple of 4), so every word shares
            // the same 0-3 byte shift from an unaligned mapping.
            device const uint* word = (device const uint*)(wq + n * uint(K / 2) + k / 2);
            const uint packed = wqShift == 0 ? word[0] :
                (word[0] >> (8 * wqShift)) | (word[1] << (32 - 8 * wqShift));
            const uint g = n * uint(groupsPerRow) + group;
            const float scale = scales[g], zero = zeros[g];
            for (uint t = 0; t < 8; ++t) {
                const float weight = (float((packed >> (4 * t)) & 15) - zero) * scale;
                for (uint b = 0; b < q4_dynamic_batch; ++b) {
                    uint m = m0 + b;
                    if (m < uint(M))
                        accum[r * q4_dynamic_batch + b] = fma(weight,
                            a[m * uint(K) + k + t], accum[r * q4_dynamic_batch + b]);
                }
            }
        }
    }
    for (uint i = 0; i < q4_dynamic_rows * q4_dynamic_batch; ++i)
        partial[i * q4_dynamic_tg + lane] = accum[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = q4_dynamic_tg / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            for (uint i = 0; i < q4_dynamic_rows * q4_dynamic_batch; ++i)
                partial[i * q4_dynamic_tg + lane] += partial[i * q4_dynamic_tg + lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint r = 0; r < q4_dynamic_rows; ++r)
            for (uint b = 0; b < q4_dynamic_batch; ++b)
                if (n0 + r < uint(N) && m0 + b < uint(M))
                    out[(m0 + b) * uint(N) + n0 + r] = partial[(r * q4_dynamic_batch + b) * q4_dynamic_tg];
    }
}

// Shared pieces of the decode (M=1) Q4 GEMV kernels below. Three function
// constants pick how a thread reads and multiplies its weights; the host times
// the combinations on the device and keeps the fastest.
//   packs: 4-byte packs (8 weights each) a thread takes per step: 1, 2 or 4.
//   load:  0 = two aligned words shifted together (the weights sit 0-3 bytes
//              off a word boundary in a zero-copy mapping),
//          1 = one unaligned 4-byte load, 2 = four byte loads.
//   pre:   1 = the activations are pre-divided by 16^t so a nibble is only
//              masked, never shifted, before the multiply.
constant uint q4v_packs_c [[function_constant(112)]];
constant uint q4v_load_c [[function_constant(113)]];
constant uint q4v_pre_c [[function_constant(114)]];
constant uint q4v_packs = is_function_constant_defined(q4v_packs_c) ? q4v_packs_c : 1;
constant uint q4v_load = is_function_constant_defined(q4v_load_c) ? q4v_load_c : 0;
constant uint q4v_pre = is_function_constant_defined(q4v_pre_c) ? q4v_pre_c : 0;

struct Q4Act {
    float4 v[8];   // 8 activations per pack
    float sum;     // their plain sum, for the zero-point correction
};

// Adds the 8 activations of pack `pk` to act (pre-scaled when q4v_pre is set).
static inline void q4v_put(thread Q4Act& act, uint pk, float4 x0, float4 x1) {
    act.sum += (x0.x + x0.y) + (x0.z + x0.w) + ((x1.x + x1.y) + (x1.z + x1.w));
    if (q4v_pre != 0) {
        act.v[2 * pk] = x0 * float4(1.0f, 0.0625f, 0.00390625f, 0.000244140625f);
        act.v[2 * pk + 1] = x1 * float4(1.52587890625e-05f, 9.5367431640625e-07f,
            5.9604644775390625e-08f, 3.725290298461914e-09f);
    } else {
        act.v[2 * pk] = x0;
        act.v[2 * pk + 1] = x1;
    }
}

static inline Q4Act q4v_act(device const float* a) {
    Q4Act act;
    act.sum = 0.0f;
    for (uint pk = 0; pk < q4v_packs; ++pk)
        q4v_put(act, pk, *(device const float4*)(a + 8 * pk), *(device const float4*)(a + 8 * pk + 4));
    return act;
}

// Same for a vector that still has to go through RMSNorm: the activations are
// multiplied by gamma here and the sum of squares of the raw values goes to ssq,
// so the norm becomes one scale of the finished dot products (x*g*inv . w = inv * (x*g . w)).
static inline Q4Act q4v_act_norm(device const float* a, device const float* g, thread float& ssq) {
    Q4Act act;
    act.sum = 0.0f;
    for (uint pk = 0; pk < q4v_packs; ++pk) {
        const float4 x0 = *(device const float4*)(a + 8 * pk);
        const float4 x1 = *(device const float4*)(a + 8 * pk + 4);
        ssq += dot(x0, x0) + dot(x1, x1);
        q4v_put(act, pk, x0 * *(device const float4*)(g + 8 * pk),
                x1 * *(device const float4*)(g + 8 * pk + 4));
    }
    return act;
}

// The 4 bytes (8 weights) at `base + shift`; `base` is word aligned.
static inline uint q4v_word(device const uchar* base, uint shift) {
    if (q4v_load == 1)
        return as_type<uint>(uchar4(*(device const packed_uchar4*)(base + shift)));
    if (q4v_load == 2) {
        device const uchar* b = base + shift;
        return uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24);
    }
    device const uint* word = (device const uint*)base;
    return shift == 0 ? word[0] : (word[0] >> (8 * shift)) | (word[1] << (32 - 8 * shift));
}

static inline float q4v_dot(uint p, float4 a0, float4 a1) {
    if (q4v_pre != 0)
        return fma(a0.x, float(p & 0xFu), fma(a0.y, float(p & 0xF0u),
            fma(a0.z, float(p & 0xF00u), fma(a0.w, float(p & 0xF000u),
            fma(a1.x, float(p & 0xF0000u), fma(a1.y, float(p & 0xF00000u),
            fma(a1.z, float(p & 0xF000000u), a1.w * float(p & 0xF0000000u))))))));
    return fma(a0.x, float(p & 15u), fma(a0.y, float((p >> 4) & 15u),
        fma(a0.z, float((p >> 8) & 15u), fma(a0.w, float((p >> 12) & 15u),
        fma(a1.x, float((p >> 16) & 15u), fma(a1.y, float((p >> 20) & 15u),
        fma(a1.z, float((p >> 24) & 15u), a1.w * float(p >> 28))))))));
}

// dot(weights of one row at this step, activations), before scale and zero.
static inline float q4v_rowdot(device const uchar* base, uint shift, Q4Act act) {
    float d = 0.0f;
    for (uint pk = 0; pk < q4v_packs; ++pk)
        d += q4v_dot(q4v_word(base + 4 * pk, shift), act.v[2 * pk], act.v[2 * pk + 1]);
    return d;
}

// Decode (M=1) Q4 GEMV with one simdgroup per `q4s_rows` output rows. The lanes
// of a simdgroup split K, the activation is loaded once per step and shared by
// every row, and the partial sums are combined with simd_sum, so there is no
// threadgroup memory and no barrier. dot(q, a) is corrected once per step:
//   sum += scale * (dot(q, a) - zero * sum(a)).
constant uint q4s_rows [[function_constant(110)]];
constant uint q4s_sg [[function_constant(111)]];

kernel void matmul_q4_simd(
    device const float* a [[buffer(0)]],
    device const uchar* wq [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* zeros [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant int& M [[buffer(5)]], constant int& K [[buffer(6)]],
    constant int& N [[buffer(7)]], constant int& groupSize [[buffer(8)]],
    constant int& groupsPerRow [[buffer(9)]],
    constant uint& wqShift [[buffer(10)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdW [[threads_per_simdgroup]]) {
    const uint row0 = (tg * q4s_sg + sgid) * q4s_rows;
    if (row0 >= uint(N)) return;
    float sum[8];
    for (uint r = 0; r < q4s_rows; ++r) sum[r] = 0;
    const uint rowBytes = uint(K) / 2;
    const uint lastRow = uint(N) - 1;
    const uint vals = 8 * q4v_packs;
    const uint gs = uint(max(groupSize, 1));
    const bool gPow2 = (gs & (gs - 1)) == 0;
    const uint gShift = 31 - clz(gs);
    for (uint k = lane * vals; k < uint(K); k += simdW * vals) {
        const Q4Act act = q4v_act(a + k);
        const uint g = groupSize > 0 ? (gPow2 ? k >> gShift : k / gs) : 0;
        for (uint r = 0; r < q4s_rows; ++r) {
            const uint n = min(row0 + r, lastRow);
            const float d = q4v_rowdot(wq + n * rowBytes + k / 2, wqShift, act);
            const uint gi = n * uint(groupsPerRow) + g;
            sum[r] = fma(scales[gi], d - zeros[gi] * act.sum, sum[r]);
        }
    }
    for (uint r = 0; r < q4s_rows; ++r) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < uint(N)) out[row0 + r] = s;
    }
}

// Fused decode GEMV (M=1), MLX-style. One simdgroup computes 4 output rows.
//   mode 0: up to 3 independent matrices (Q/K/V) share the input vector; the
//           grid covers N0+N1+N2 rows, each N must be a multiple of 4.
//   mode 1: gate/up pair; out0[n] = silu(gate[n]) * up[n].
// HAS_RES adds res[n] to the result (residual epilogue for o_proj / down_proj).
constant uint q4f_mode [[function_constant(120)]];
constant uint q4f_res [[function_constant(121)]];
constant uint q4f_half [[function_constant(124)]];   // 1: scales/zeros are fp16
constant uint q4f_norm [[function_constant(123)]];   // 1: `a` is the raw vector, RMSNorm folded in

// One step of row n: its packs at column k, scaled by their group, added to acc.
static inline float q4f_row(device const uchar* wq, device const uchar* sc,
        device const uchar* zr, uint n, uint rowBytes, uint gpr, uint k, uint g,
        uint shift, Q4Act act, float acc) {
    const float d = q4v_rowdot(wq + n * rowBytes + k / 2, shift, act);
    const uint gi = n * gpr + g;
    float s, z;
    if (q4f_half != 0) {   // `sc` holds the (scale, zero) fp16 pair of the group: one 4-byte load
        const half2 sz = ((device const half2*)sc)[gi];
        s = float(sz.x); z = float(sz.y);
    } else {
        s = ((device const float*)sc)[gi]; z = ((device const float*)zr)[gi];
    }
    return fma(s, d - z * act.sum, acc);
}

kernel void matmul_q4_fused(
    device const float* a [[buffer(0)]],
    device const uchar* wq0 [[buffer(1)]], device const uchar* wq1 [[buffer(2)]],
    device const uchar* wq2 [[buffer(3)]],
    device const uchar* sc0 [[buffer(4)]], device const uchar* sc1 [[buffer(5)]],
    device const uchar* sc2 [[buffer(6)]],
    device const uchar* zr0 [[buffer(7)]], device const uchar* zr1 [[buffer(8)]],
    device const uchar* zr2 [[buffer(9)]],
    device float* out0 [[buffer(10)]], device float* out1 [[buffer(11)]],
    device float* out2 [[buffer(12)]],
    device const float* res [[buffer(13)]],
    constant int& K [[buffer(14)]], constant int& groupSize [[buffer(15)]],
    constant int& groupsPerRow [[buffer(16)]],
    constant uint4& nn [[buffer(17)]],      // N0, N1, N2, unused
    constant uint4& sh [[buffer(18)]],      // wqShift of the three matrices
    device const float* gamma [[buffer(19)]],   // RMSNorm weight (q4f_norm)
    constant float& eps [[buffer(20)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdW [[threads_per_simdgroup]]) {
    const uint row0 = (tg * 2 + sgid) * 4;
    device const uchar* wqA = wq0; device const uchar* scA = sc0; device const uchar* zrA = zr0;
    device float* outA = out0; uint shA = sh.x; uint base = 0;
    if (q4f_mode == 0) {
        if (row0 >= nn.x + nn.y + nn.z) return;
        if (row0 >= nn.x + nn.y) {
            wqA = wq2; scA = sc2; zrA = zr2; outA = out2; shA = sh.z; base = nn.x + nn.y;
        } else if (row0 >= nn.x) {
            wqA = wq1; scA = sc1; zrA = zr1; outA = out1; shA = sh.y; base = nn.x;
        }
    } else if (row0 >= nn.x) {
        return;
    }
    const uint local0 = row0 - base;
    float sumA_[4] = {0, 0, 0, 0};
    float sumB_[4] = {0, 0, 0, 0};
    const uint rowBytes = uint(K) / 2;
    const uint gpr = uint(groupsPerRow);
    const uint vals = 8 * q4v_packs;   // activations per thread per step
    const uint gs = uint(max(groupSize, 1));
    const bool gPow2 = (gs & (gs - 1)) == 0;
    const uint gShift = 31 - clz(gs);
    float ssq = 0.0f;
    for (uint k = lane * vals; k < uint(K); k += simdW * vals) {
        const Q4Act act = q4f_norm != 0 ? q4v_act_norm(a + k, gamma + k, ssq) : q4v_act(a + k);
        const uint g = groupSize > 0 ? (gPow2 ? k >> gShift : k / gs) : 0;
        for (uint r = 0; r < 4; ++r) {
            const uint n = local0 + r;
            sumA_[r] = q4f_row(wqA, scA, zrA, n, rowBytes, gpr, k, g, shA, act, sumA_[r]);
            if (q4f_mode == 1)
                sumB_[r] = q4f_row(wq1, sc1, zr1, n, rowBytes, gpr, k, g, sh.y, act, sumB_[r]);
        }
    }
    const float inv = q4f_norm != 0 ? rsqrt(simd_sum(ssq) / float(K) + eps) : 1.0f;
    for (uint r = 0; r < 4; ++r) {
        float v = simd_sum(sumA_[r]) * inv;
        if (q4f_mode == 1) {
            const float u = simd_sum(sumB_[r]) * inv;
            v = v / (1.0f + exp(-v)) * u;
        }
        if (q4f_res != 0) v += res[local0 + r];
        if (lane == 0) outA[local0 + r] = v;
    }
}

// Appends one decode token's K and V (HKV heads x D) into the combined KV
// caches laid out as [head][maxSeq][D].
kernel void kv_append_kernel(
    device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
    device float* kAll [[buffer(2)]], device float* vAll [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& maxSeq [[buffer(5)]],
    constant uint& pos [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    const uint h = i / D, d = i % D;
    const uint dst = h * maxSeq * D + pos * D + d;
    kAll[dst] = k[i];
    vAll[dst] = v[i];
}

// Interleaves the fp32 scale and zero of every group into one half2 (scale, zero).
kernel void pack_scale_zero_f16_kernel(device const float* scales [[buffer(0)]],
                                       device const float* zeros [[buffer(1)]],
                                       device half2* out [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
    out[i] = half2(half(scales[i]), half(zeros[i]));
}

// RoPE(Q) + Q scale + RoPE(K) + KV append in one dispatch (decode, one token).
// Threads [0, H*half) rotate Q in place and scale it, the next HKV*half rotate K straight into
// the cache row, the last HKV*D copy V into the cache row.
kernel void rope_kv_kernel(
    device float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
    device const float* csTab [[buffer(3)]], device const float* snTab [[buffer(4)]],   // [maxSeq][D/2]
    device float* kAll [[buffer(5)]], device float* vAll [[buffer(6)]],
    constant uint4& dims [[buffer(7)]],     // H, HKV, D, maxSeq
    constant uint& pos [[buffer(8)]], constant float& scale [[buffer(9)]],
    uint t [[thread_position_in_grid]]) {
    const uint H = dims.x, HKV = dims.y, D = dims.z, maxSeq = dims.w, half_ = D / 2;
    device const float* cs = csTab + pos * half_;
    device const float* sn = snTab + pos * half_;
    if (t < H * half_) {
        const uint h = t / half_, i = t % half_;
        device float* p = q + h * D;
        const float x1 = p[i], x2 = p[i + half_], c = cs[i], s = sn[i];
        p[i] = (x1 * c - x2 * s) * scale;
        p[i + half_] = (x2 * c + x1 * s) * scale;
        return;
    }
    t -= H * half_;
    if (t < HKV * half_) {
        const uint h = t / half_, i = t % half_;
        const float x1 = k[h * D + i], x2 = k[h * D + i + half_], c = cs[i], s = sn[i];
        device float* dst = kAll + h * maxSeq * D + pos * D;
        dst[i] = x1 * c - x2 * s;
        dst[i + half_] = x2 * c + x1 * s;
        return;
    }
    t -= HKV * half_;
    if (t < HKV * D) {
        const uint h = t / D, d = t % D;
        vAll[h * maxSeq * D + pos * D + d] = v[t];
    }
}

// ---- Pipelined decode: the next token is chosen on the GPU ----
// ring[] holds token ids. A step reads its input token from one slot and
// writes the next token to the following slot, so the CPU can encode the next
// step while this one still runs.

// Copies the embedding row of the token in ring[slot] into out[0..<dim].
kernel void embed_token_kernel(
    device const float* table [[buffer(0)]], device const int* ring [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& dim [[buffer(3)]],
    constant uint& slot [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    out[i] = table[uint(ring[slot]) * dim + i];
}

// Greedy choice: the largest finite logit, the lowest index among equals, and
// 0 when no logit is finite. One threadgroup of 256 threads.
kernel void argmax_token_kernel(
    device const float* logits [[buffer(0)]], device int* ring [[buffer(1)]],
    constant uint& n [[buffer(2)]], constant uint& slot [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float bestV[256];
    threadgroup uint bestI[256];
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint i = tid; i < n; i += 256) {
        const float v = logits[i];
        if (isfinite(v) && v > bv) { bv = v; bi = i; }
    }
    bestV[tid] = bv;
    bestI[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            const float ov = bestV[tid + s];
            const uint oi = bestI[tid + s];
            if (ov > bestV[tid] || (ov == bestV[tid] && oi < bestI[tid])) {
                bestV[tid] = ov;
                bestI[tid] = oi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) ring[slot] = bestI[0] == 0xFFFFFFFFu ? 0 : int(bestI[0]);
}
