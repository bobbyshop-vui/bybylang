#version 450
// Mamba2 chunked prefill: FLA (flash linear attention) intra-chunk kernel plus
// the SSD (state space duality) inter-chunk passes, on tensor cores where the
// mode is a matmul. One GLSL source, five pipelines picked by SSD_MODE:
//
//   0 cumsum   per (chunk, head): inclusive prefix sum of a_t = dt_t * A_h
//   1 intra    FLA: Y_intra = (L o (C Bd^T)) X on cooperative matrices,
//              L[i,j] = exp(cum_i - cum_j) for j <= i (no softmax, decay mask)
//   2 states   S_loc[c] = (Bd o exp(cumLast - cum_j))^T X   (per-chunk state)
//   3 combine  sequential scan over the nC chunk states per head (tiny), done
//              in place: S_loc[c] becomes the state ENTERING chunk c; the
//              final state lands in states slot 1 for the decode carry
//   4 inter    Y += (C o exp(cum_i)) S_entering  on cooperative matrices
//
// Everything reads/writes ONE float32 arena buffer (the microVM arena) through
// push-constant offsets, so the kernel drops into the existing prefill layout:
// conv rows [t, cDim] = [x(C) | B(G*N) | C(G*N)] (post conv+SiLU), dt [t, H]
// (post softplus), aLog/dSkip [H], yPre [t, C], states [2, HD, N] (slot 0 =
// carried initial state when cont != 0, slot 1 = final state out).
//
// Compile-time: SSD_MODE, SSD_N (d_state), SSD_D (head dim), SSD_Q (chunk
// length, power of two, multiple of 16). All decays satisfy cum_i - cum_j <= 0
// for j <= i (A < 0, dt > 0), so every masked weight is <= 1 and safe in FP16;
// the FP16 operands carry Bd = B*dt and X, with FP32 accumulation.
// SSD_SCALAR=1 builds modes 1/2/4 as plain FP32 FMA loops over the same
// 16-row tiles (256 lanes) for GPUs without cooperative matrices.

#ifndef SSD_MODE
#define SSD_MODE 0
#endif
#ifndef SSD_N
#define SSD_N 128
#endif
#ifndef SSD_D
#define SSD_D 64
#endif
#ifndef SSD_Q
#define SSD_Q 64
#endif

#ifndef SSD_SCALAR
#define SSD_SCALAR 0
#endif

#if (SSD_MODE == 1 || SSD_MODE == 2 || SSD_MODE == 4) && !SSD_SCALAR
#extension GL_KHR_cooperative_matrix : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require
#endif
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x_id = 0) in;

layout(std430, binding = 0) buffer Arena { float a[]; };

layout(push_constant) uniform Params {
    uint T, H, G, N, D, Q, nC, C, cDim;
    uint offConv, offDt, offCum, offALog, offDSkip, offYPre, offStates, offSloc;
    uint cont;
} p;

#define SLABS (SSD_Q / 16)
#define NBLK (SSD_N / 16)
#define DBLK (SSD_D / 16)
#define LDN (SSD_N + 8)
#define LDD (SSD_D + 8)
#define LDP 24

// B_t,g,n inside a conv row; the group of head h is h / (H / G).
uint convRow(uint t) { return p.offConv + t * p.cDim; }
uint bCol(uint h, uint n) { return p.C + (h / (p.H / p.G)) * p.N + n; }
uint cCol(uint h, uint n) { return p.C + p.G * p.N + (h / (p.H / p.G)) * p.N + n; }

#if SSD_MODE == 0
// ── cumsum: one workgroup per (chunk, head), SSD_Q lanes ─────────────────────
shared float sa[SSD_Q];

void main() {
    uint c = gl_WorkGroupID.x;
    uint h = gl_WorkGroupID.y;
    uint t = gl_LocalInvocationID.x;
    uint gt = c * p.Q + t;
    float A = -exp(a[p.offALog + h]);
    float v = gt < p.T ? a[p.offDt + gt * p.H + h] * A : 0.0;
    sa[t] = v;
    barrier();
    for (uint s = 1u; s < SSD_Q; s <<= 1u) {
        float add = t >= s ? sa[t - s] : 0.0;
        barrier();
        sa[t] += add;
        barrier();
    }
    if (gt < p.T) a[p.offCum + gt * p.H + h] = sa[t];
}

#elif SSD_MODE == 3
// ── combine: one workgroup per head, sequential over the nC chunk states ─────
shared float atot[256];

void main() {
    uint h = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    uint nLanes = gl_WorkGroupSize.x;
    for (uint c = lid; c < p.nC; c += nLanes) {
        uint last = min((c + 1u) * p.Q, p.T) - 1u;
        atot[c] = exp(a[p.offCum + last * p.H + h]);
    }
    barrier();
    for (uint idx = lid; idx < p.N * p.D; idx += nLanes) {
        uint n = idx / p.D;
        uint d = idx - n * p.D;
        uint hd = h * p.D + d;
        float s = p.cont != 0u ? a[p.offStates + hd * p.N + n] : 0.0;
        for (uint c = 0u; c < p.nC; ++c) {
            uint si = p.offSloc + ((c * p.H + h) * p.N + n) * p.D + d;
            float tmp = a[si];
            a[si] = s;                      // state ENTERING chunk c
            s = atot[c] * s + tmp;
        }
        a[p.offStates + (p.C + hd) * p.N + n] = s;   // slot 1: decode carry
    }
}

#elif SSD_SCALAR
// ── scalar FP32 modes (1 intra, 2 states, 4 inter), 256 lanes ────────────────
// Same tiles as the cooperative path. Each lane owns OPL outputs of the
// 16 x SSD_D tile: element e = lane + k * 256 -> (e / SSD_D, e % SSD_D).
#define OPL ((16 * SSD_D + 255) / 256)
#define LSN (SSD_N + 1)

#if SSD_MODE == 1
shared float Cs[16 * LSN];
shared float Bs[16 * LSN];
shared float Xs[16 * SSD_D];
shared float ps[256];
shared float cumI[16];
shared float cumJ[16];
#elif SSD_MODE == 2
shared float Bs[256];
shared float Xs[16 * SSD_D];
#else
shared float Cs[256];
shared float Ss[16 * SSD_D];
shared float dec[16];
#endif

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint h = gl_WorkGroupID.y;
    float acc[OPL];
    [[unroll]] for (uint k = 0u; k < OPL; ++k) acc[k] = 0.0;

#if SSD_MODE == 1
    uint c = gl_WorkGroupID.x / SLABS;
    uint slab = gl_WorkGroupID.x - c * SLABS;
    for (uint i = lane; i < 16u * SSD_N; i += 256u) {
        uint r = i / SSD_N;
        uint n = i - r * SSD_N;
        uint t = c * p.Q + slab * 16u + r;
        Cs[r * LSN + n] = t < p.T ? a[convRow(t) + cCol(h, n)] : 0.0;
    }
    if (lane < 16u) {
        uint t = c * p.Q + slab * 16u + lane;
        cumI[lane] = t < p.T ? a[p.offCum + t * p.H + h] : 0.0;
    }
    for (uint jb = 0u; jb <= slab; ++jb) {
        barrier();
        for (uint i = lane; i < 16u * SSD_N; i += 256u) {
            uint r = i / SSD_N;
            uint n = i - r * SSD_N;
            uint t = c * p.Q + jb * 16u + r;
            Bs[r * LSN + n] = t < p.T ?
                a[convRow(t) + bCol(h, n)] * a[p.offDt + t * p.H + h] : 0.0;
        }
        for (uint i = lane; i < 16u * SSD_D; i += 256u) {
            uint r = i / SSD_D;
            uint d = i - r * SSD_D;
            uint t = c * p.Q + jb * 16u + r;
            Xs[i] = t < p.T ? a[convRow(t) + h * p.D + d] : 0.0;
        }
        if (lane < 16u) {
            uint t = c * p.Q + jb * 16u + lane;
            cumJ[lane] = t < p.T ? a[p.offCum + t * p.H + h] : 0.0;
        }
        barrier();
        {
            uint r = lane / 16u;
            uint jj = lane - r * 16u;
            uint gi = slab * 16u + r;
            uint gj = jb * 16u + jj;
            float s = 0.0;
            if (gj <= gi && c * p.Q + gi < p.T && c * p.Q + gj < p.T) {
                for (uint n = 0u; n < SSD_N; ++n) s = fma(Cs[r * LSN + n], Bs[jj * LSN + n], s);
                s *= exp(cumI[r] - cumJ[jj]);
            }
            ps[lane] = s;
        }
        barrier();
        [[unroll]] for (uint k = 0u; k < OPL; ++k) {
            uint e = lane + k * 256u;
            if (e < 16u * SSD_D) {
                uint r = e / SSD_D;
                uint d = e - r * SSD_D;
                float v = acc[k];
                for (uint j = 0u; j < 16u; ++j) v = fma(ps[r * 16u + j], Xs[j * SSD_D + d], v);
                acc[k] = v;
            }
        }
    }
    float dsk = a[p.offDSkip + h];
    [[unroll]] for (uint k = 0u; k < OPL; ++k) {
        uint e = lane + k * 256u;
        uint r = e / SSD_D;
        uint d = e - r * SSD_D;
        uint t = c * p.Q + slab * 16u + r;
        if (e < 16u * SSD_D && t < p.T)
            a[p.offYPre + t * p.C + h * p.D + d] = acc[k] + dsk * a[convRow(t) + h * p.D + d];
    }

#elif SSD_MODE == 2
    uint c = gl_WorkGroupID.x / NBLK;
    uint nb = gl_WorkGroupID.x - c * NBLK;
    uint last = min((c + 1u) * p.Q, p.T) - 1u;
    float cumLast = a[p.offCum + last * p.H + h];
    for (uint jb = 0u; jb < SLABS; ++jb) {
        barrier();
        {
            uint r = lane / 16u;
            uint nn = lane - r * 16u;
            uint t = c * p.Q + jb * 16u + r;
            Bs[lane] = t < p.T ?
                a[convRow(t) + bCol(h, nb * 16u + nn)] * a[p.offDt + t * p.H + h] *
                exp(cumLast - a[p.offCum + t * p.H + h]) : 0.0;
        }
        for (uint i = lane; i < 16u * SSD_D; i += 256u) {
            uint r = i / SSD_D;
            uint d = i - r * SSD_D;
            uint t = c * p.Q + jb * 16u + r;
            Xs[i] = t < p.T ? a[convRow(t) + h * p.D + d] : 0.0;
        }
        barrier();
        [[unroll]] for (uint k = 0u; k < OPL; ++k) {
            uint e = lane + k * 256u;
            if (e < 16u * SSD_D) {
                uint nn = e / SSD_D;
                uint d = e - nn * SSD_D;
                float v = acc[k];
                for (uint r = 0u; r < 16u; ++r) v = fma(Bs[r * 16u + nn], Xs[r * SSD_D + d], v);
                acc[k] = v;
            }
        }
    }
    [[unroll]] for (uint k = 0u; k < OPL; ++k) {
        uint e = lane + k * 256u;
        if (e < 16u * SSD_D) {
            uint nn = e / SSD_D;
            uint d = e - nn * SSD_D;
            a[p.offSloc + ((c * p.H + h) * p.N + nb * 16u + nn) * p.D + d] = acc[k];
        }
    }

#else // SSD_MODE == 4
    uint c = gl_WorkGroupID.x / SLABS;
    uint slab = gl_WorkGroupID.x - c * SLABS;
    if (c == 0u && p.cont == 0u) return;
    if (lane < 16u) {
        uint t = c * p.Q + slab * 16u + lane;
        dec[lane] = t < p.T ? exp(a[p.offCum + t * p.H + h]) : 0.0;
    }
    for (uint nb = 0u; nb < NBLK; ++nb) {
        barrier();
        {
            uint r = lane / 16u;
            uint nn = lane - r * 16u;
            uint t = c * p.Q + slab * 16u + r;
            Cs[lane] = t < p.T ? a[convRow(t) + cCol(h, nb * 16u + nn)] : 0.0;
        }
        for (uint i = lane; i < 16u * SSD_D; i += 256u) {
            uint nn = i / SSD_D;
            uint d = i - nn * SSD_D;
            Ss[i] = a[p.offSloc + ((c * p.H + h) * p.N + nb * 16u + nn) * p.D + d];
        }
        barrier();
        [[unroll]] for (uint k = 0u; k < OPL; ++k) {
            uint e = lane + k * 256u;
            if (e < 16u * SSD_D) {
                uint r = e / SSD_D;
                uint d = e - r * SSD_D;
                float v = acc[k];
                for (uint nn = 0u; nn < 16u; ++nn) v = fma(Cs[r * 16u + nn], Ss[nn * SSD_D + d], v);
                acc[k] = v;
            }
        }
    }
    [[unroll]] for (uint k = 0u; k < OPL; ++k) {
        uint e = lane + k * 256u;
        uint r = e / SSD_D;
        uint d = e - r * SSD_D;
        uint t = c * p.Q + slab * 16u + r;
        if (e < 16u * SSD_D && t < p.T)
            a[p.offYPre + t * p.C + h * p.D + d] += acc[k] * dec[r];
    }
#endif
}

#else
// ── cooperative-matrix modes (1 intra, 2 states, 4 inter) ────────────────────

#if SSD_MODE == 1
shared float16_t Cs[16 * LDN];
shared float16_t Bs[16 * LDN];
shared float16_t Xs[16 * LDD];
shared float16_t ps[16 * LDP];
shared float sT[256];
shared float cumI[16];
shared float cumJ[16];
#elif SSD_MODE == 2
shared float16_t Bs[16 * LDP];
shared float16_t Xs[16 * LDD];
shared float sT[256];
#else
shared float16_t Cs[16 * LDP];
shared float16_t Ss[16 * LDD];
shared float sT[256];
shared float dec[16];
#endif

void main() {
    uint lane = gl_LocalInvocationID.x;
    uint nLanes = gl_WorkGroupSize.x;
    uint h = gl_WorkGroupID.y;

#if SSD_MODE == 1
    uint c = gl_WorkGroupID.x / SLABS;
    uint slab = gl_WorkGroupID.x - c * SLABS;
    // Stage the 16 C rows of this slab (never decayed: the mask carries the
    // decay) and their inclusive cumsums.
    for (uint i = lane; i < 16u * SSD_N; i += nLanes) {
        uint r = i / SSD_N;
        uint n = i - r * SSD_N;
        uint t = c * p.Q + slab * 16u + r;
        Cs[r * LDN + n] = float16_t(t < p.T ? a[convRow(t) + cCol(h, n)] : 0.0);
    }
    if (lane < 16u) {
        uint t = c * p.Q + slab * 16u + lane;
        cumI[lane] = t < p.T ? a[p.offCum + t * p.H + h] : 0.0;
    }
    barrier();

    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> ca[NBLK];
    [[unroll]] for (uint nt = 0u; nt < NBLK; ++nt)
        coopMatLoad(ca[nt], Cs, nt * 16u, LDN, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> yacc[DBLK];
    [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd)
        yacc[dd] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint jb = 0u; jb <= slab; ++jb) {
        barrier();
        for (uint i = lane; i < 16u * SSD_N; i += nLanes) {
            uint r = i / SSD_N;
            uint n = i - r * SSD_N;
            uint t = c * p.Q + jb * 16u + r;
            float v = 0.0;
            if (t < p.T) v = a[convRow(t) + bCol(h, n)] * a[p.offDt + t * p.H + h];
            Bs[r * LDN + n] = float16_t(v);
        }
        for (uint i = lane; i < 16u * SSD_D; i += nLanes) {
            uint r = i / SSD_D;
            uint d = i - r * SSD_D;
            uint t = c * p.Q + jb * 16u + r;
            Xs[r * LDD + d] = float16_t(t < p.T ? a[convRow(t) + h * p.D + d] : 0.0);
        }
        if (lane < 16u) {
            uint t = c * p.Q + jb * 16u + lane;
            cumJ[lane] = t < p.T ? a[p.offCum + t * p.H + h] : 0.0;
        }
        barrier();

        coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sacc =
            coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
        [[unroll]] for (uint nt = 0u; nt < NBLK; ++nt) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bm;
            coopMatLoad(bm, Bs, nt * 16u, LDN, gl_CooperativeMatrixLayoutColumnMajor);
            sacc = coopMatMulAdd(ca[nt], bm, sacc);
        }
        coopMatStore(sacc, sT, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint idx = lane; idx < 256u; idx += nLanes) {
            uint r = idx / 16u;
            uint jj = idx - r * 16u;
            uint gi = slab * 16u + r;
            uint gj = jb * 16u + jj;
            float w = (gj <= gi && c * p.Q + gj < p.T && c * p.Q + gi < p.T)
                      ? exp(cumI[r] - cumJ[jj]) : 0.0;
            ps[r * LDP + jj] = float16_t(sT[idx] * w);
        }
        barrier();
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> pa;
        coopMatLoad(pa, ps, 0u, LDP, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> xb;
            coopMatLoad(xb, Xs, dd * 16u, LDD, gl_CooperativeMatrixLayoutRowMajor);
            yacc[dd] = coopMatMulAdd(pa, xb, yacc[dd]);
        }
    }

    float dsk = a[p.offDSkip + h];
    [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd) {
        barrier();
        coopMatStore(yacc[dd], sT, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint idx = lane; idx < 256u; idx += nLanes) {
            uint r = idx / 16u;
            uint cc = idx - r * 16u;
            uint t = c * p.Q + slab * 16u + r;
            if (t < p.T) {
                uint d = dd * 16u + cc;
                float x = a[convRow(t) + h * p.D + d];
                a[p.offYPre + t * p.C + h * p.D + d] = sT[idx] + dsk * x;
            }
        }
    }

#elif SSD_MODE == 2
    uint c = gl_WorkGroupID.x / NBLK;
    uint nb = gl_WorkGroupID.x - c * NBLK;
    uint last = min((c + 1u) * p.Q, p.T) - 1u;
    float cumLast = a[p.offCum + last * p.H + h];

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> sacc[DBLK];
    [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd)
        sacc[dd] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint jb = 0u; jb < SLABS; ++jb) {
        barrier();
        for (uint i = lane; i < 256u; i += nLanes) {
            uint r = i / 16u;            // j within the block
            uint nn = i - r * 16u;
            uint t = c * p.Q + jb * 16u + r;
            float v = 0.0;
            if (t < p.T)
                v = a[convRow(t) + bCol(h, nb * 16u + nn)] * a[p.offDt + t * p.H + h] *
                    exp(cumLast - a[p.offCum + t * p.H + h]);
            Bs[r * LDP + nn] = float16_t(v);
        }
        for (uint i = lane; i < 16u * SSD_D; i += nLanes) {
            uint r = i / SSD_D;
            uint d = i - r * SSD_D;
            uint t = c * p.Q + jb * 16u + r;
            Xs[r * LDD + d] = float16_t(t < p.T ? a[convRow(t) + h * p.D + d] : 0.0);
        }
        barrier();
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
        coopMatLoad(am, Bs, 0u, LDP, gl_CooperativeMatrixLayoutColumnMajor);
        [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> xb;
            coopMatLoad(xb, Xs, dd * 16u, LDD, gl_CooperativeMatrixLayoutRowMajor);
            sacc[dd] = coopMatMulAdd(am, xb, sacc[dd]);
        }
    }

    [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd) {
        barrier();
        coopMatStore(sacc[dd], sT, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint idx = lane; idx < 256u; idx += nLanes) {
            uint rr = idx / 16u;         // n within the block
            uint cc = idx - rr * 16u;    // d within the block
            uint n = nb * 16u + rr;
            a[p.offSloc + ((c * p.H + h) * p.N + n) * p.D + dd * 16u + cc] = sT[idx];
        }
    }

#else // SSD_MODE == 4
    uint c = gl_WorkGroupID.x / SLABS;
    uint slab = gl_WorkGroupID.x - c * SLABS;
    if (c == 0u && p.cont == 0u) return;   // entering state is zero

    if (lane < 16u) {
        uint t = c * p.Q + slab * 16u + lane;
        dec[lane] = t < p.T ? exp(a[p.offCum + t * p.H + h]) : 0.0;
    }
    barrier();

    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> yacc[DBLK];
    [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd)
        yacc[dd] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

    for (uint nb = 0u; nb < NBLK; ++nb) {
        barrier();
        for (uint i = lane; i < 256u; i += nLanes) {
            uint r = i / 16u;
            uint nn = i - r * 16u;
            uint t = c * p.Q + slab * 16u + r;
            float v = t < p.T ? a[convRow(t) + cCol(h, nb * 16u + nn)] * dec[r] : 0.0;
            Cs[r * LDP + nn] = float16_t(v);
        }
        for (uint i = lane; i < 16u * SSD_D; i += nLanes) {
            uint nn = i / SSD_D;
            uint d = i - nn * SSD_D;
            uint n = nb * 16u + nn;
            Ss[nn * LDD + d] =
                float16_t(a[p.offSloc + ((c * p.H + h) * p.N + n) * p.D + d]);
        }
        barrier();
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
        coopMatLoad(am, Cs, 0u, LDP, gl_CooperativeMatrixLayoutRowMajor);
        [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd) {
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> sb;
            coopMatLoad(sb, Ss, dd * 16u, LDD, gl_CooperativeMatrixLayoutRowMajor);
            yacc[dd] = coopMatMulAdd(am, sb, yacc[dd]);
        }
    }

    [[unroll]] for (uint dd = 0u; dd < DBLK; ++dd) {
        barrier();
        coopMatStore(yacc[dd], sT, 0u, 16u, gl_CooperativeMatrixLayoutRowMajor);
        barrier();
        for (uint idx = lane; idx < 256u; idx += nLanes) {
            uint r = idx / 16u;
            uint cc = idx - r * 16u;
            uint t = c * p.Q + slab * 16u + r;
            if (t < p.T)
                a[p.offYPre + t * p.C + h * p.D + dd * 16u + cc] += sT[idx];
        }
    }
#endif
}
#endif
