#include <metal_stdlib>
using namespace metal;

// ============================================================
// VECTOR OPERATIONS
// ============================================================
kernel void vecop_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const int& op [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (op == 0) c[id] = a[id] + b[id];
    else if (op == 1) c[id] = a[id] - b[id];
    else if (op == 2) c[id] = a[id] * b[id];
    else if (op == 3) c[id] = a[id] / b[id];
}

// ============================================================
// COPY INTO (GPU-side concat) - NEW: dst[dstOffset + id] = src[id],
// used to join several child buffers (e.g. the outHeads of each KV head in
// attention decode) into one contiguous [H*D] buffer DIRECTLY ON THE GPU, instead of
// sessionRead() to the CPU and sessionUpload() again just to concatenate -
// that was the last blocker preventing the attention-decode session and the
// oProj session from being merged. dstOffset is in ELEMENTS (not
// bytes) - metal_copy_into_enc (metal_shim.m) multiplies by sizeof(float) when
// binding the buffer offset.
kernel void copy_into_kernel(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    dst[id] = src[id];
}

// ============================================================
// SCALE (x *= scalar, in-place) - NEW: applies the attention "scale"
// (1/sqrt(D)) to the WHOLE Q buffer [H*D] ONCE on the GPU, instead
// of reading Q back and multiplying each element in a for loop (see qGroups in
// nim_inference.nim). With scale_kernel the per-group "gather"
// of Q needs no scatter kernel: Q has layout [H,D] with
// h = hkv*nRep + r contiguous in hkv, so each group hkv is just a CONTIGUOUS
// SLICE of nRep*D starting at offset hkv*nRep*D - extracted directly with
// copy_into_kernel (sessionCopyInto, existing, parameter srcOffsetElems),
// no new gather/scatter kernel needed.
kernel void scale_kernel(
    device float* x               [[buffer(0)]],
    device const float& scalar    [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    x[id] *= scalar;
}

// ============================================================
// MATMUL TILED
// ============================================================
// CHANGED (replacing the earlier "hard-code BB_TILE=8 for EVERY GPU" fix):
// The real root cause was NOT "tile 32 is always dangerous" but that
// metal_matmul()/metal_matmul_enc() dispatched groupSize=32x32=1024 threads
// WITHOUT checking the pipeline.maxTotalThreadsPerThreadgroup of the running
// GPU first (unlike metal_vecop in the same file, which correctly uses MIN(...)). Lowering
// the tile to 8 for EVERY GPU avoided that bug safely but hurt performance:
// on Apple Silicon / strong discrete GPUs a 1024-thread threadgroup is perfectly
// valid and tile 32 is many times faster than tile 8 (much better data reuse:
// each tileA/tileB element is reused 32 times instead of 8 before
// reloading from device memory).
// The RIGHT fix: build 3 kernel variants with tiles 8/16/32 (macro-generated,
// NO function constants because the threadgroup array size needs a literal
// when the MSL is built, avoiding incompatibility with old drivers). On the
// metal_shim.m side, the pipeline.maxTotalThreadsPerThreadgroup of EACH
// pipeline is measured after creation (not guessed from the GPU name) and the
// largest variant the threadgroup really allows is chosen, falling back step by step otherwise.
#define BB_MATMUL_VARIANT(TILE, KFUNC)                                       \
kernel void KFUNC(                                                           \
    device const float* a [[buffer(0)]],                                    \
    device const float* b [[buffer(1)]],                                    \
    device float* c [[buffer(2)]],                                          \
    device const int& m [[buffer(3)]],                                      \
    device const int& k [[buffer(4)]],                                      \
    device const int& n [[buffer(5)]],                                      \
    uint2 gid [[thread_position_in_grid]],                                  \
    uint2 tid [[thread_position_in_threadgroup]]                            \
) {                                                                          \
    threadgroup float tileA[TILE][TILE];                                    \
    threadgroup float tileB[TILE][TILE];                                    \
    int row = int(gid.y);                                                   \
    int col = int(gid.x);                                                   \
    int tx = int(tid.x), ty = int(tid.y);                                   \
    float sum = 0.0f;                                                       \
    int numTiles = (k + TILE - 1) / TILE;                                   \
    for (int t = 0; t < numTiles; ++t) {                                    \
        int aCol = t * TILE + tx;                                           \
        int bRow = t * TILE + ty;                                           \
        tileA[ty][tx] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;    \
        tileB[ty][tx] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        for (int p = 0; p < TILE; ++p) {                                    \
            sum += tileA[ty][p] * tileB[p][tx];                             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                        \
    if (row < m && col < n) {                                               \
        c[row * n + col] = sum;                                             \
    }                                                                        \
}

// 3 variants - metal_shim.m picks one from the real pipeline.maxTotalThreadsPerThreadgroup
// of the running GPU (8x8=64, 16x16=256, 32x32=1024 threads/threadgroup).
BB_MATMUL_VARIANT(8,  matmul_kernel_t8)
BB_MATMUL_VARIANT(16, matmul_kernel_t16)
BB_MATMUL_VARIANT(32, matmul_kernel_t32)

// The name "matmul_kernel" stays as an alias of tile 8 so any old call site (if one
// is left outside the dynamic tile choice) still compiles; this name is NEVER
// dispatched with groupSize > 8x8.
BB_MATMUL_VARIANT(8, matmul_kernel)

// ============================================================
// TILED "NT" MATMUL (C = A @ B^T, B stored row-major [N,K] - NO physical
// transpose). NEW: for attention decode - the K-cache/V-cache are stored naturally
// in [S,D] layout (N=S, K=headDim) - removes the O(S*D)/token CPU
// transpose that was needed before to build kBlockT.
// ============================================================
#define BB_MATMUL_NT_VARIANT(TILE, KFUNC)                                    \
kernel void KFUNC(                                                           \
    device const float* a [[buffer(0)]],                                    \
    device const float* b [[buffer(1)]],                                    \
    device float* c [[buffer(2)]],                                          \
    device const int& m [[buffer(3)]],                                      \
    device const int& k [[buffer(4)]],                                      \
    device const int& n [[buffer(5)]],                                      \
    uint2 gid [[thread_position_in_grid]],                                  \
    uint2 tid [[thread_position_in_threadgroup]]                            \
) {                                                                          \
    threadgroup float tileA[TILE][TILE];                                    \
    threadgroup float tileB[TILE][TILE];                                    \
    int tx = int(tid.x), ty = int(tid.y);                                   \
    int rowBase = int(gid.y) - ty;                                          \
    int colBase = int(gid.x) - tx;                                          \
    int row = rowBase + ty;                                                 \
    int col = colBase + tx;                                                 \
    float sum = 0.0f;                                                       \
    int numTiles = (k + TILE - 1) / TILE;                                   \
    for (int t = 0; t < numTiles; ++t) {                                    \
        int aK = t * TILE + tx;                                             \
        int bCol = colBase + ty; /* independent of 'col' - ty is the column when loading B */ \
        int bK = t * TILE + tx;                                             \
        tileA[ty][tx] = (row < m && aK < k) ? a[row * k + aK] : 0.0f;        \
        tileB[ty][tx] = (bCol < n && bK < k) ? b[bCol * k + bK] : 0.0f;      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        for (int p = 0; p < TILE; ++p) {                                    \
            sum += tileA[ty][p] * tileB[tx][p];                             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }                                                                        \
    if (row < m && col < n) {                                               \
        c[row * n + col] = sum;                                             \
    }                                                                        \
}

BB_MATMUL_NT_VARIANT(8, matmul_nt_kernel)

// ============================================================
// MATMUL2 - DUAL MATMUL
// ============================================================
kernel void matmul2_kernel(
    device const float* a1 [[buffer(0)]],
    device const float* b1 [[buffer(1)]],
    device float* c1 [[buffer(2)]],
    device const int32_t* dims1 [[buffer(3)]],
    device const float* a2 [[buffer(4)]],
    device const float* b2 [[buffer(5)]],
    device float* c2 [[buffer(6)]],
    device const int32_t* dims2 [[buffer(7)]],
    uint2 position [[thread_position_in_grid]]
) {
    int m1 = dims1[0], k1 = dims1[1], n1 = dims1[2];
    int m2 = dims2[0], k2 = dims2[1], n2 = dims2[2];
    
    int row = position.y;
    int col = position.x;
    
    if (row < m1 && col < n1) {
        float sum = 0.0f;
        for (int i = 0; i < k1; ++i) {
            sum += a1[row * k1 + i] * b1[i * n1 + col];
        }
        c1[row * n1 + col] = sum;
    }
    
    if (row < m2 && col < n2) {
        float sum = 0.0f;
        for (int i = 0; i < k2; ++i) {
            sum += a2[row * k2 + i] * b2[i * n2 + col];
        }
        c2[row * n2 + col] = sum;
    }
}

// ============================================================
// ACTIVATIONS
// ============================================================
kernel void activation_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int& op [[buffer(2)]],
    device const float& p [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float val = x[id];
    if (op == 0) {
        y[id] = val > 0.0f ? val : 0.0f;
    } else if (op == 1) {
        y[id] = 1.0f / (1.0f + exp(-val));
    } else if (op == 2) {
        y[id] = tanh(val);
    } else if (op == 3) {
        float alpha = 0.1f;
        float beta = 1.0f;
        y[id] = val > 0.0f ? val * (1.0f + alpha * val) : beta * val * exp(val);
    } else if (op == 4) {
        y[id] = exp(val);
    } else if (op == 5) {
        y[id] = log(val);
    } else if (op == 6) {
        y[id] = sqrt(val);
    } else if (op == 7) {
        y[id] = pow(val, p);
    }
}

// ============================================================
// ACTIVATION BACKWARD
// ============================================================
kernel void activation_backward_kernel(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* dx [[buffer(2)]],
    device const int& op [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float val = x[id];
    float d = dy[id];
    if (op == 0) {
        dx[id] = val > 0.0f ? d : 0.0f;
    } else if (op == 1) {
        float sig = 1.0f / (1.0f + exp(-val));
        dx[id] = d * sig * (1.0f - sig);
    } else if (op == 2) {
        float t = tanh(val);
        dx[id] = d * (1.0f - t * t);
    } else if (op == 3) {
        float alpha = 0.1f;
        float beta = 1.0f;
        dx[id] = val > 0.0f ? d * (1.0f + 2.0f * alpha * val) : d * beta * exp(val) * (1.0f + val);
    }
}

// ============================================================
// ROPE (rotate_half, in-place) - NEW
// ============================================================
// Before, RoPE had to: read Q/K back to the CPU, build cosFull/sinFull/rotHalf arrays
// of size [T,H,D] (wasteful, cos/sin repeated H times), upload again, run
// 3 sessionVecOps (mul,mul,add), then read back to the CPU again for
// attention - a SEPARATE GPU session + 2 CPU round trips just for RoPE.
// This kernel applies rotate_half DIRECTLY while Q/K are still on the GPU
// (right after the QKV matmul, IN THE SAME session), with no repeated arrays
// and no CPU reads in between. cosT/sinT are the [T, D/2] slices cut by
// posOffset on the CPU (the original rope.cosT/sinT tables do not change between
// layers/tokens, so the slice is tiny compared to Q/K).
kernel void rope_kernel(
    device float* x               [[buffer(0)]],   // [T, Hn, D] in-place
    device const float* cosT      [[buffer(1)]],   // [T, D/2]
    device const float* sinT      [[buffer(2)]],   // [T, D/2]
    device const int& Hn          [[buffer(3)]],
    device const int& D           [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]          // gid.x = i in [0, D/2), gid.y = h in [0, Hn), gid.z = t in [0, T)
) {
    int half_ = D / 2;
    int i = int(gid.x);
    if (i >= half_) return;
    int h = int(gid.y);
    int t = int(gid.z);
    int base = (t * Hn + h) * D;
    float c = cosT[t * half_ + i];
    float s = sinT[t * half_ + i];
    float x1 = x[base + i];
    float x2 = x[base + i + half_];
    x[base + i]         = x1 * c - x2 * s;
    x[base + i + half_] = x2 * c + x1 * s;
}

// ============================================================
// SOFTMAX
// ============================================================
#define BB_WG 256

// CHANGED (fix for a consistent miscalculation - see the debug report): ALL 3 kernels
// below (softmax/rmsnorm/layernorm) assumed a FIXED BB_WG=256 threads per
// threadgroup for both the size of the "red[]" reduction array AND the loop
// step - but the dispatch side (metal_shim.m) computes threadsPerGroup =
// MIN(pipeline.maxTotalThreadsPerThreadgroup, 256) AT RUNTIME. If a GPU/
// pipeline has maxTotalThreadsPerThreadgroup < 256 (entirely possible
// on weak/old GPUs like Iris 640 - many anomalies were seen on this
// GPU while debugging), fewer than 256 threads are REALLY dispatched, so
// red[realThreads..255] is NEVER written by any thread - it is
// UNINITIALISED threadgroup memory. The reduction still starts at a fixed
// BB_WG/2=128, so it ADDED that garbage into the total (sum of squares for
// rmsnorm, max/sum for softmax) -> CONSISTENTLY wrong results (deterministic,
// because the garbage in threadgroup memory is usually stable between calls in
// the same process) for EVERY layer and EVERY token using these kernels. Fix: use
// [[threads_per_threadgroup]] (provided by Metal, the REAL number of running
// threads) so the existing threads themselves zero the rest of red[]
// (from tgSize to BB_WG-1) BEFORE the reduction, so NO slot stays
// uninitialised however much smaller than BB_WG the dispatched threadgroup is.

kernel void softmax_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    device const int& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float m = -INFINITY;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        m = max(m, x[off + c]);
    }
    red[tid] = m;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxVal = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        float e = exp(x[off + c] - maxVal);
        y[off + c] = e;
        sum += e;
    }
    red[tid] = sum;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float invSum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        y[off + c] *= invSum;
    }
}

// ============================================================
// RMSNORM FORWARD (mirror of layernorm_kernel above without the mean subtraction + beta -
// RMSNorm only scales by sqrt(mean(x^2)), no centering on the mean, no
// bias). The threadgroup reduction is kept INTACT from the proven layernorm_kernel
// (same BB_WG, same tid += BB_WG pattern, same barrier order) -
// only the final elementwise formula differs and one buffer (beta) is gone.
// ============================================================
kernel void rmsnorm_kernel(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int& cols [[buffer(3)]],
    device const float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float sumSq = 0.0f;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        float v = x[off + c];
        sumSq += v * v;
    }
    red[tid] = sumSq;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float meanSq = red[0] / float(cols);
    float invStd = 1.0f / sqrt(meanSq + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        y[off + c] = x[off + c] * invStd * gamma[c];
    }
}

// ============================================================
// LAYERNORM FORWARD
// ============================================================
kernel void layernorm_kernel(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const int& cols [[buffer(4)]],
    device const float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float red[BB_WG];
    uint off = row * cols;

    float sum = 0.0f, sumSq = 0.0f;
    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        float v = x[off + c];
        sum += v;
        sumSq += v * v;
    }
    red[tid] = sum;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = red[0] / float(cols);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    red[tid] = sumSq;
    for (uint idx = tid + tgSize; idx < BB_WG; idx += tgSize) { red[idx] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BB_WG / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float meanSq = red[0] / float(cols);
    float var = meanSq - mean * mean;
    float invStd = 1.0f / sqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int c = int(tid); c < cols; c += int(tgSize)) {   // FIX bug #10: step = REAL thread count (tgSize), not BB_WG
        y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
}

// ============================================================
// LAYERNORM BACKWARD
// ============================================================
kernel void layernorm_backward_kernel(
    device const float* dy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* dx [[buffer(4)]],
    device float* dgamma [[buffer(5)]],
    device float* dbeta [[buffer(6)]],
    device const int& rows [[buffer(7)]],
    device const int& cols [[buffer(8)]],
    device const float& eps [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < (uint)cols) {
        float dg = 0.0f;
        float db = 0.0f;
        for (int r = 0; r < rows; ++r) {
            uint off = r * cols;
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) mean += x[off + c];
            mean /= float(cols);
            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float diff = x[off + c] - mean;
                var += diff * diff;
            }
            var /= float(cols);
            float invStd = 1.0f / sqrt(var + eps);
            
            float norm = (x[off + id] - mean) * invStd;
            dg += dy[off + id] * norm;
            db += dy[off + id];
        }
        dgamma[id] = dg;
        dbeta[id] = db;
    }
    
    uint row = id;
    if (row < (uint)rows) {
        uint off = row * cols;
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += x[off + c];
        mean /= float(cols);
        
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = x[off + c] - mean;
            var += diff * diff;
        }
        var /= float(cols);
        float invStd = 1.0f / sqrt(var + eps);
        
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float grad = dy[off + c] * gamma[c] * invStd;
            sum1 += grad;
            sum2 += grad * (x[off + c] - mean);
        }
        sum2 *= (-invStd * invStd / float(cols));
        
        for (int c = 0; c < cols; ++c) {
            float term1 = dy[off + c] * gamma[c] * invStd;
            float term2 = sum1 / float(cols);
            // sum2 = -invStd^3 * mean(dy*gamma*(x-mean)) (already divided by cols above): NO extra 2/cols
            // (the old version made the variance correction term cols/2 times too small -> wrong dx).
            float term3 = (x[off + c] - mean) * sum2;
            dx[off + c] = term1 - term2 + term3;
        }
    }
}

// ============================================================
// EMBEDDING LOOKUP
// ============================================================
kernel void embedding_lookup_kernel(
    device const float* table [[buffer(0)]],
    device const int32_t* indices [[buffer(1)]],
    device float* y [[buffer(2)]],
    device const int& dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    int32_t idx = indices[id];
    uint dst_off = id * dim;
    uint src_off = idx * dim;
    for (int d = 0; d < dim; ++d) {
        y[dst_off + d] = table[src_off + d];
    }
}

// ============================================================
// MATMUL DIRECTLY ON ASYMMETRIC INT4 (packed, per-group scale/zero_point)
// ============================================================
// CHANGED (item 3 of the performance notes): BEFORE, every Linear.forward()
// had to dequantizeTensorTransposed() ALL int4 weights into a full fp32
// array on the CPU (e.g. 4096x11008 ~180MB fp32) and UPLOAD that whole array to the
// GPU just to multiply once and discard it - exactly what llama.cpp does NOT do (it multiplies
// directly on the quantized data). This kernel multiplies directly on the
// packed int4 buffer (2 values/byte) + per-group scale/zero_point, with NO
// separate CPU dequant and NO full fp32 upload - about 4-8x less upload
// bandwidth (the llama.cpp "reduce memory bandwidth" idea),
// also freeing the CPU time of the dequant loop.
// The layout must match quant.nim EXACTLY (quantizeInt4Asymmetric /
// dequantizeTensorTransposed):
//   - wq: [N, K] row-major, 2 nibbles/byte, low nibble = even
//     element (k%2==0), high nibble = odd element (k%2==1).
//   - scales/zeros: if groupSize>0, each row N is split into
//     ceil(K/groupSize) groups, group[g] uses scales[n*nGroupsPerRow+g].
//   - c[m,n] = sum_k a[m,k] * ((nibble(n,k) - zero(n,g)) * scale(n,g))
//     -- exactly the formula of dequantizeTensorTransposed() + the current beMatmul,
//     only without materialising wT in fp32 first.
// NOT YET RUN ON A REAL METAL GPU (generated on Linux, without
// a real macOS/Metal to build+run) - follows the CPU
// reference formula (quant.nim) closely but MUST be built+tested on a real
// machine before generating real tokens. If tokens come out as garbage, this is the first
// suspect (wrong nibble offset or wrong group order).
// CHANGED ("work like llama.cpp" - the OLD kernel below was renamed
// matmul_q4_kernel_naive for reference/rollback):
// OLD kernel: 1 thread = 1 output element, reading & summing ALL K elements
// sequentially (no SIMD reduction), 16x1 threadgroups = only 16 threads per
// group -> very low occupancy on a real GPU (Iris has hundreds of idle ALUs).
// The NEW kernel follows the strategy of kernel_mul_mv_q4_K in llama.cpp/ggml's
// Metal backend:
//   1. Each threadgroup = EXACTLY one simdgroup (32 threads), handling MMQ4_NDST=4
//      output rows (n) at once for one activation row (m) - dispatched as
//      (rowBlock = ceil(N/4), M).
//   2. The activation a[m,:] (K elements) is LOADED ONCE by the 32 threads into threadgroup
//      shared memory and REUSED for all 4 rows - before, each row n re-read
//      a[] from device memory (N/4 times the bandwidth).
//   3. Per row: the 32 threads split the K elements (k = lane, lane+32, ...),
//      accumulate partials and reduce with simd_sum() in one hardware instruction
//      (instead of one thread looping K times) - the main speedup.
// Result: N/4 times more threadgroups dispatched (enough to fill all
// EUs of even a weak GPU), each doing real parallel work (32-way)
// instead of one thread doing all of K.
// SEVERE BUG FIX (caused a real "GPU Timeout Error" on an Intel Iris machine):
// The previous version cached the whole activation row in K*4 bytes of threadgroup shared memory
// - with K = intermediate_size (e.g. down_proj, K~11008+) that needs
// ~44KB, ABOVE the per-threadgroup shared-memory limit of Iris (usually only
// 16-32KB depending on the generation). Metal gives no clear error at setThreadgroupMemoryLength
// on such old GPUs - it hangs the GPU (timeout), not a compile error.
// FIX: no threadgroup memory at all. Loop order swapped: k is the OUTER loop (each
// thread reads a[m,k] from device memory EXACTLY ONCE into a local register
// `aVal`), rr (the MMQ4_NDST rows) is the INNER loop reusing that register -
// still "no N/4 re-reads of the activation" like the shared-memory version,
// but absolutely safe because no shared memory is allocated.
// SEVERE BUG FIX (garbage tokens on a real Intel Iris Plus 640 through Metal):
// The previous version used simd_sum() + thread_index_in_simdgroup, ASSUMING a
// 32-thread threadgroup == EXACTLY one hardware simdgroup (as the old comment said,
// "NOT YET RUN ON A REAL METAL GPU" - never verified on a
// real machine). On Intel GPUs (Gen9 and many others) through Metal,
// the REAL threads_per_simdgroup can be 8 or 16, 32 is NOT guaranteed -
// then a 32-thread threadgroup is split into SEVERAL separate hardware
// simdgroups, each with its OWN tiisg==0 -> SEVERAL threads overwrote
// c[m*N+n] (race condition, the last writer wins) AND each simd_sum()
// summed only the PART of K handled by THAT simdgroup, not all of K -> WRONG
// results for EVERY Linear with Q4 weights (almost the whole quantized
// model) - the cause of the "garbage tokens" on Metal (exactly the bug
// found and fixed in the Vulkan version - matmul_q4.comp).
// FIX: no simd_sum/thread_index_in_simdgroup; threadgroup (shared)
// memory + a tree reduction over thread_position_in_threadgroup (always
// unique 0..31 within the WHOLE threadgroup, whatever simdgroup size the hardware
// reports) - the same principle as the Vulkan version.
#define MMQ4_NDST 8
// CHANGED: raised from 4 to 8 (in sync with Vulkan + const MMQ4_NDST in
// vulkan_driver.nim) - each a[k] read is reused for 8 columns instead
// of 4, about 50% fewer threadgroups dispatched, a better FLOPs/byte ratio
// for buffer A. SAFE to raise because this kernel uses a threadgroup shared-
// memory reduction (no reliance on simd_sum/thread_index_in_simdgroup),
// no hardware simdgroup assumption at all.
#define MMQ4_LANES 32
// PERFORMANCE FIX (this kernel was SLOWER than the old simd_sum version for 2 reasons): (1) each
// threadgroup had only 32 threads -> 4 times more threadgroups dispatched
// than the old version (which grouped MMQ4_SG_PER_TG=4 simdgroups per threadgroup), (2)
// 6 threadgroup_barriers per block (1 tier2 build + 5 tree-reduce levels) on
// a GPU with less parallelism, because 9.7KB of threadgroup memory per threadgroup limited
// occupancy. FIX: MMQ4_GROUPS_PER_TG=4 32-lane "logical groups" share
// ONE 128-thread threadgroup - each logical group handles its own
// rowBlock, using "lane = lid % 32" instead of "lid" as the index within the group -
// 4 times fewer threadgroups dispatched (same as the old simd_sum version) WHILE keeping
// the barrier + threadgroup-memory principle (no return to the assumption of a
// 32-wide hardware simdgroup). tier2 (nibble split) is shared by all 128 threads
// (not per group). tier1 (dequant LUT) + curGroup are now
// threadgroup memory shared by the WHOLE logical group (no longer per
// lane) - SAFE because gr=k/groupSize depends ONLY on k, and k in one iteration
// is IDENTICAL for all 128 threads (every logical group loops
// k=lane, lane+32,... with the same lane=0..31) => the "gr changed" condition is
// UNIFORM across the threadgroup, so the threadgroup_barrier inside that branch
// is VALID (no thread takes a different branch than another).
// ASSUMPTION: groupSize % 32 == 0 (true for the usual Q4 groupSizes 32/
// 64/128) so 32 consecutive k values in one iteration never
// cross a group boundary - with a groupSize that is not a multiple of 32,
// tier1 may be rebuilt one iteration late for some lanes (the final values are still CORRECT,
// only some cache benefit is lost, the results are not wrong).
#define MMQ4_GROUPS_PER_TG 4
#define MMQ4_TG_SIZE (MMQ4_LANES * MMQ4_GROUPS_PER_TG)   // 128

// ============================================================
// CHANGED (borrowing the IDEA of llama.cpp/ggml's Metal backend - kernel_mul_mv_q4_K_f32
// - see ggml-metal.metal, N_SIMDWIDTH=32 "assuming SIMD group size is 32"):
// ggml HARD-ASSUMES simdgroup=32 on EVERY Apple GPU (including Intel iGPUs through
// Metal) and uses simd_sum()/thread_index_in_simdgroup DIRECTLY - CONFIRMED
// CORRECT + FAST (3 tok/s) on this very Intel Iris Plus 640 (real
// evidence from running llama-cli, not a guess). This shows the
// earlier worry of this kernel ("threads_per_simdgroup on Intel Gen9 can
// be 8/16, 32 not guaranteed" - see the history of the threadgroup-
// memory version below) was EXCESSIVE/over-safe on the current Iris 640 driver - slower
// than necessary (threadgroup shared memory + more barriers than a real simd_sum).
// The project's OWN data layout is KEPT (nibble-packed wq + per-group scales/
// zeros, NOT ggml's block_q4_K - changing the layout would mean
// rewriting quant.nim too, with bit-packing bug risks, not worth it).
// ONLY the dispatch TECHNIQUE is borrowed: 1 threadgroup = EXACTLY 1 simdgroup (32 threads,
// like ggml's N_SIMDWIDTH), MMQ4_NDST=8 output rows per threadgroup, a
// real simd_sum() instead of a hand-written threadgroup tree reduce - ALL
// threadgroup memory is gone (tier1Tg/tier2Lo/tier2Hi/sdata of the previous version), because
// simd_sum needs no shared memory.
// SAFE regarding dynamic-index registers (the REAL cause of earlier timeouts):
// sc[]/zp[] are FIXED-SIZE private arrays (MMQ4_NDST=8), indexed ONLY
// through #pragma unroll (compile-time constant rr), NEVER
// with a runtime variable (nibble/k) - unlike the old tier1[16], which was indexed
// by "nibble" (runtime) and spilled registers into slow scratch memory.
kernel void matmul_q4_kernel(
    device const float* a          [[buffer(0)]],   // [M,K]
    device const uchar* wq         [[buffer(1)]],   // [N, ceil(K/2)] packed int4
    device const float* scales     [[buffer(2)]],   // [N * nGroupsPerRow]
    device const float* zeros      [[buffer(3)]],   // [N * nGroupsPerRow]
    device float* c                [[buffer(4)]],   // [M,N]
    device const int& M            [[buffer(5)]],
    device const int& K            [[buffer(6)]],
    device const int& N            [[buffer(7)]],
    device const int& groupSize    [[buffer(8)]],
    device const int& nGroupsPerRow [[buffer(9)]],
    device const int& simdWidth    [[buffer(10)]],  // NEW - see the explanation below
    uint3 tgpig [[threadgroup_position_in_grid]],   // (rowBlock, m, 0)
    uint3 tiisgv [[thread_position_in_threadgroup]] // .x = 0..simdWidth-1 - BAT BUOC dispatch threadgroup size = (simdWidth,1,1)
    // uint3 (not ushort/a plain uint) because mixing vector+scalar attributes
    // in one argument list is an MSL compile error (hit and fixed in other
    // kernels of this file - see attention_fused_flash_kernel).
    // SEVERE BUG FIX (the REAL cause of the "garbage tokens" after switching
    // to simd_sum): the previous version HARDCODED 32 as both the loop stride AND
    // the condition for writing the result (tiisg==0), ARBITRARILY ASSUMING "a 32-thread
    // threadgroup = EXACTLY one hardware simdgroup". That assumption was NEVER
    // really verified (it relied only on ggml's comment "assuming SIMD group size
    // is 32" - that is ggml's DISPATCH CHOICE, NOT
    // hardware evidence, and ggml's code IS STILL CORRECT when the assumption is wrong
    // because it uses the builtin [[thread_index_in_simdgroup]] and handles
    // several simdgroups per threadgroup through sgitg - the previous version did NOT).
    // If the REAL hardware simdgroup of Iris 640 is 8 or 16 (quite likely
    // - Intel Gen9 architecture), a 32-thread threadgroup is AUTOMATICALLY split
    // into 2-4 SEPARATE hardware simdgroups - simd_sum() ONLY adds
    // WITHIN one hardware simdgroup, and "tiisg==0" (position 0
    // IN THE THREADGROUP) is only lane 0 of the FIRST simdgroup - the
    // contributions of the other simdgroups are LOST entirely -> WRONG sum -> garbage
    // tokens (no crash, because this is valid but WRONG computation, not an
    // out-of-bounds memory access).
    // ROOT FIX: NO more guessing 32 - the host (metal_shim.m) reads the
    // REAL pipeline.threadExecutionWidth (official Metal API, returning
    // the actual hardware simdgroup width of THAT PIPELINE on the RUNNING
    // GPU) and dispatches threadgroups of EXACTLY that size - guaranteeing a
    // threadgroup ALWAYS EQUALS one REAL hardware simdgroup, whether the hardware
    // is 8/16/32. Then "thread_position_in_threadgroup" and
    // "thread_index_in_simdgroup" COINCIDE (0..simdWidth-1) - tiisg==0
    // is then REALLY the ONLY lane 0 of the WHOLE threadgroup.
) {
    uint tiisg = tiisgv.x;   // 0..simdWidth-1
    int m = int(tgpig.y);
    if (m >= M) return;

    #define MMQ4_NDST 8   // output rows per threadgroup (= per simdgroup)
    int rowBlock = int(tgpig.x);
    int n0 = rowBlock * MMQ4_NDST;
    int bytesPerRow = (K + 1) / 2;

    float sum[MMQ4_NDST];
    #pragma unroll
    for (int rr = 0; rr < MMQ4_NDST; ++rr) sum[rr] = 0.0f;

    int prevGr = -1;
    float sc[MMQ4_NDST];
    float zp[MMQ4_NDST];

    for (int k = int(tiisg); k < K; k += simdWidth) {
        float aVal = a[m * K + k];
        int gr = groupSize > 0 ? (k / groupSize) : 0;

        // Cache scale/zero for the CURRENT group - device memory is re-read ONLY
        // when the group really changes (like tier1 in the old version, but now a
        // private per-thread array instead of threadgroup memory - SAFE because the index rr
        // is a compile-time constant through #pragma unroll, NOT runtime).
        if (gr != prevGr) {
            prevGr = gr;
            #pragma unroll
            for (int rr = 0; rr < MMQ4_NDST; ++rr) {
                int n = n0 + rr;
                if (n < N) {
                    sc[rr] = scales[n * nGroupsPerRow + gr];
                    zp[rr] = zeros[n * nGroupsPerRow + gr];
                } else {
                    sc[rr] = 0.0f;
                    zp[rr] = 0.0f;
                }
            }
        }

        int byteIdx = k >> 1;
        bool lowNibble = (k & 1) == 0;
        #pragma unroll
        for (int rr = 0; rr < MMQ4_NDST; ++rr) {
            int n = n0 + rr;
            if (n >= N) continue;
            uchar byteVal = wq[n * bytesPerRow + byteIdx];
            uchar nibble = lowNibble ? (byteVal & 0x0Fu) : ((byteVal >> 4) & 0x0Fu);
            sum[rr] += aVal * ((float(nibble) - zp[rr]) * sc[rr]);
        }
    }

    // A REAL simd_sum() - like ggml's kernel_mul_mv_q4_K_f32 (no
    // threadgroup_barrier/hand-written tree reduce at all). CORRECT because the threadgroup
    // = EXACTLY one REAL hardware simdgroup (the host dispatches with this
    // simdWidth), no assumptions left.
    #pragma unroll
    for (int rr = 0; rr < MMQ4_NDST; ++rr) {
        float total = simd_sum(sum[rr]);
        if (tiisg == 0u) {
            int n = n0 + rr;
            if (n < N) c[m * N + n] = total;
        }
    }
    #undef MMQ4_NDST
}

// ============================================================
// EXPERIMENTAL VARIANT - SMALLER MMQ4_NDST (8 -> 2/4, following
// N_R0_Q4_K=2 of ggml's kernel_mul_mv_q4_K_f32_impl) + nibbles read
// IN PAIRS (k2,k2+1) with one uchar read + plain bitmasks, NO
// runtime branches (the "difference #1" pointed out
// when comparing with ggml).
//
// HISTORY: NDST=8 + pairing was tried first -> MEASURED on a real Intel
// Iris Plus 640 as SLOWER than the original (from the user's log: ~2-3x
// slower at the same token position). Suspected root cause: NDST=8 means
// EVERY iteration keeps 8 sum[]/sc[]/zp[] elements alive AT ONCE
// PLUS aVal0/aVal1/nib0/nib1/hasK1 every iteration - Iris has a
// very small register file (other register-overflow bugs happened in
// this very file - see the history of tier1[]/acc[] in other
// kernels) => most likely SPILLED, losing all the benefit of fewer
// memory reads.
//
// ggml picks N_R0_Q4_K = ONLY 2 (not 8) - much less register
// pressure. This variant does what ggml does: a small NDST (2 or 4) +
// pairing, as a NEW kernel (NOT the old
// "matmul_q4_kernel" name) to A/B test on a real machine without
// overwriting the version measured as STABLE (0.86 tok/s).
//
// NOT YET RUN ON A REAL METAL GPU - the same initial status as
// matmul_q4_kernel had, it MUST be measured on a real machine
// before generating real tokens. If slower/wrong, go back to
// "matmul_q4_kernel" (unchanged, still above).
#define BB_MATMUL_Q4_VARIANT(NDST_VAL, KFUNC)                                \
kernel void KFUNC(                                                          \
    device const float* a          [[buffer(0)]],                          \
    device const uchar* wq         [[buffer(1)]],                          \
    device const float* scales     [[buffer(2)]],                          \
    device const float* zeros      [[buffer(3)]],                          \
    device float* c                [[buffer(4)]],                          \
    device const int& M            [[buffer(5)]],                          \
    device const int& K            [[buffer(6)]],                          \
    device const int& N            [[buffer(7)]],                          \
    device const int& groupSize    [[buffer(8)]],                          \
    device const int& nGroupsPerRow [[buffer(9)]],                         \
    device const int& simdWidth    [[buffer(10)]],                         \
    threadgroup float* partial [[threadgroup(0)]],                         \
    uint3 tgpig [[threadgroup_position_in_grid]],                          \
    uint3 tiisgv [[thread_position_in_threadgroup]]                        \
) {                                                                          \
    uint tiisg = tiisgv.x;                                                  \
    int m = int(tgpig.y);                                                   \
    if (m >= M) return;                                                     \
                                                                              \
    const int NDST = NDST_VAL;                                              \
    int rowBlock = int(tgpig.x);                                            \
    int n0 = rowBlock * NDST;                                               \
    int bytesPerRow = (K + 1) / 2;                                          \
                                                                              \
    float sum[NDST_VAL];                                                    \
    _Pragma("unroll")                                                       \
    for (int rr = 0; rr < NDST_VAL; ++rr) sum[rr] = 0.0f;                   \
                                                                              \
    int prevGr = -1;                                                        \
    float sc[NDST_VAL];                                                     \
    float zp[NDST_VAL];                                                     \
                                                                              \
    /* CHANGED (zero point moved out of the main loop - learned from how ggml     \
       handles q4_1 with sumy): instead of subtracting zp[rr] and multiplying by sc[rr] for EVERY      \
       element and EVERY rr (1 sub + 2 muls x NDST per element), now only  \
       dot[rr] += aVal*nibble (1 mul x NDST per element) and one     \
       sumA (Sigma aVal) SHARED by every rr in the SAME group. When the   \
       group changes (or at the end of the loop), one flush:                  \
       sum[rr] += sc[rr]*dot[rr] - zp[rr]*sc[rr]*sumA - mathematically identical to   \
       the old formula (distributivity), only the accumulation ORDER changes to   \
       cut ALU work in the hot loop. Safe with NDST=2/4 (extra     \
       registers: dot[NDST_VAL] + one scalar sumA, much lighter than NDST=8 \
       which spilled - see the history comment above). */                      \
    float dot[NDST_VAL];                                                    \
    _Pragma("unroll")                                                       \
    for (int rr = 0; rr < NDST_VAL; ++rr) dot[rr] = 0.0f;                   \
    float sumA = 0.0f;                                                      \
                                                                              \
    /* groupSize is assumed EVEN (true for the usual 32/64/128/256)      \
       so k2 and k2+1 are always in the same group - see the details in         \
       the NDST=8 version tried earlier (comment above, same logic). */            \
    for (int k2 = int(tiisg) * 2; k2 < K; k2 += int(simdWidth) * 2) {       \
        float aVal0 = a[m * K + k2];                                        \
        bool hasK1 = (k2 + 1) < K;                                          \
        float aVal1 = hasK1 ? a[m * K + k2 + 1] : 0.0f;                     \
        int gr = groupSize > 0 ? (k2 / groupSize) : 0;                      \
                                                                              \
        if (gr != prevGr) {                                                 \
            if (prevGr != -1) {                                             \
                _Pragma("unroll")                                           \
                for (int rr = 0; rr < NDST_VAL; ++rr) {                     \
                    sum[rr] += sc[rr] * dot[rr] - zp[rr] * sc[rr] * sumA;   \
                    dot[rr] = 0.0f;                                          \
                }                                                            \
                sumA = 0.0f;                                                \
            }                                                               \
            prevGr = gr;                                                    \
            _Pragma("unroll")                                               \
            for (int rr = 0; rr < NDST_VAL; ++rr) {                         \
                int n = n0 + rr;                                            \
                if (n < N) {                                                \
                    sc[rr] = scales[n * nGroupsPerRow + gr];                \
                    zp[rr] = zeros[n * nGroupsPerRow + gr];                 \
                } else {                                                    \
                    sc[rr] = 0.0f;                                          \
                    zp[rr] = 0.0f;                                          \
                }                                                           \
            }                                                               \
        }                                                                   \
                                                                              \
        sumA += aVal0 + (hasK1 ? aVal1 : 0.0f);                             \
                                                                              \
        int byteIdx = k2 >> 1;                                              \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) {                             \
            int n = n0 + rr;                                                \
            if (n >= N) continue;                                          \
            uchar byteVal = wq[n * bytesPerRow + byteIdx];                  \
            uchar nib0 = byteVal & 0x0Fu;                                   \
            uchar nib1 = (byteVal >> 4) & 0x0Fu;                            \
            dot[rr] += aVal0 * float(nib0);                                 \
            if (hasK1) dot[rr] += aVal1 * float(nib1);                     \
        }                                                                    \
    }                                                                        \
    /* flush the last group (the loop ended without another group change \
       to trigger the flush above). */                                \
    if (prevGr != -1) {                                                     \
        _Pragma("unroll")                                                   \
        for (int rr = 0; rr < NDST_VAL; ++rr) {                             \
            sum[rr] += sc[rr] * dot[rr] - zp[rr] * sc[rr] * sumA;          \
        }                                                                    \
    }                                                                        \
                                                                              \
    for (int rr = 0; rr < NDST_VAL; ++rr) { \
        partial[rr * simdWidth + tiisg] = sum[rr]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint stride = uint(simdWidth) / 2; stride > 0; stride /= 2) { \
        if (tiisg < stride) { \
            for (int rr = 0; rr < NDST_VAL; ++rr) \
                partial[rr * simdWidth + tiisg] += partial[rr * simdWidth + tiisg + stride]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    if (tiisg == 0) { \
        for (int rr = 0; rr < NDST_VAL; ++rr) \
            if (n0 + rr < N) c[m * N + n0 + rr] = partial[rr * simdWidth]; \
    } \
}

// 2 variants to measure: NDST=2 (matches ggml's N_R0_Q4_K), NDST=4
// (in between). The host (metal_shim.m) MUST use the right value when computing
// rowBlocks (kMMQ4_NDST) for the kernel name being called - see the comment
// in metal_shim.m on renaming the kernel for testing.
BB_MATMUL_Q4_VARIANT(2, matmul_q4_kernel_ndst2_paired)
BB_MATMUL_Q4_VARIANT(4, matmul_q4_kernel_ndst4_paired)
#undef BB_MATMUL_Q4_VARIANT

// The original (naive) kernel is kept for comparison/rollback if the simd-reduction kernel
// above gives wrong results on a real machine (first suspect: the nibble layout
// or a wrong group offset - same formula, only the reduction differs).
kernel void matmul_q4_kernel_naive(
    device const float* a          [[buffer(0)]],
    device const uchar* wq         [[buffer(1)]],
    device const float* scales     [[buffer(2)]],
    device const float* zeros      [[buffer(3)]],
    device float* c                [[buffer(4)]],
    device const int& M            [[buffer(5)]],
    device const int& K            [[buffer(6)]],
    device const int& N            [[buffer(7)]],
    device const int& groupSize    [[buffer(8)]],
    device const int& nGroupsPerRow [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int n = int(gid.x);
    int m = int(gid.y);
    if (m >= M || n >= N) return;

    int bytesPerRow = (K + 1) / 2;
    int rowByteOff = n * bytesPerRow;
    int rowGroupOff = n * nGroupsPerRow;

    // TIER1 like the optimised version above (no TIER2 here - the naive kernel
    // is 1 thread/row/col, so nothing is "shared between threads";
    // splitting bits with shift/mask is as cheap as a table lookup for a single thread).
    float sum = 0.0f;
    int curGroup = -1;
    float tier1[16];
    for (int k = 0; k < K; ++k) {
        int g = groupSize > 0 ? (k / groupSize) : 0;
        if (g != curGroup) {
            curGroup = g;
            float sc = scales[rowGroupOff + g];
            float zp = zeros[rowGroupOff + g];
            for (int v = 0; v < 16; ++v) tier1[v] = (float(v) - zp) * sc;
        }
        uchar byteVal = wq[rowByteOff + (k >> 1)];
        uchar nibble = (k & 1) == 0 ? (byteVal & 0x0F) : ((byteVal >> 4) & 0x0F);
        sum += a[m * K + k] * tier1[nibble];
    }
    c[m * N + n] = sum;
}

// ============================================================
// ATOMIC ADD FLOAT
// ============================================================
inline void atomicAddFloatDevice(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    float expectedF = as_type<float>(expected);
    while (!atomic_compare_exchange_weak_explicit(
              addr, &expected,
              as_type<uint>(expectedF + val),
              memory_order_relaxed, memory_order_relaxed)) {
        expectedF = as_type<float>(expected);
    }
}

// ============================================================
// ATTENTION FUSED FORWARD
// ============================================================
kernel void attention_fused_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device float* s_matrix [[buffer(4)]],
    device const int& B [[buffer(5)]],
    device const int& H [[buffer(6)]],
    device const int& S [[buffer(7)]],
    device const int& D [[buffer(8)]],
    device const float& scale [[buffer(9)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint bh = position.x;
    uint ti = position.y;
    
    if (bh < (uint)(B * H) && ti < (uint)S) {
        uint base_idx = bh * S * D;
        uint base_s = bh * S * S;
        uint row = base_s + ti * S;

        // CHANGED: removed the local array "float scores[256]" (overflowed when S > 256,
        // because it was a fixed array on the thread stack, unrelated
        // to the actual S). Uses s_matrix directly (a global buffer,
        // allocated at S*S for each actual call from Nim)
        // as scratch -> no hard limit on S any more.
        float mx = -1e30f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d];
            }
            float sc = dot * scale;
            s_matrix[row + tj] = sc;
            if (sc > mx) mx = sc;
        }

        float sum_exp = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            float e = exp(s_matrix[row + tj] - mx);
            s_matrix[row + tj] = e;
            sum_exp += e;
        }

        for (uint tj = 0; tj <= ti; ++tj) {
            s_matrix[row + tj] /= sum_exp;
        }
        for (uint tj = ti + 1; tj < (uint)S; ++tj) {
            s_matrix[row + tj] = 0.0f;
        }

        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (uint tj = 0; tj <= ti; ++tj) {
                acc += s_matrix[row + tj] * v[base_idx + tj * D + d];
            }
            o[base_idx + ti * D + d] = acc;
        }
    }
}

// ============================================================
// ATTENTION FUSED - FLASH (prefill, causal), NEW.
// attention_fused_kernel (above) materialises the whole s_matrix
// [S,S] in device memory - O(S^2) memory + 3 read/write passes over the matrix
// (score, exp, divide by sum). This kernel uses an online softmax in ONE
// pass over tj=0..ti (the flash-attention definition: keep a running
// max m_i, running sum l_i and a running output accumulator acc[d]; whenever
// a new score exceeds the old max, RESCALE acc/l by
// exp(m_old - m_new) instead of recomputing) - NO S*S buffer,
// q/k/v read once, no intermediate values read/written back.
// SAME dispatch pattern as before (1 thread / (bh,ti), no
// threadgroups) - ONLY the kernel body changes, not how it is called.
// The old attention_fused_kernel is NOT replaced (still above) because
// attention_fused_backward_kernel (training) needs s_matrix (the materialised
// softmax probabilities) as input - this flash kernel is ONLY for
// inference (forward), no gradients.
// LIMIT: D (head_dim) <= FLASH_MAX_D (128, enough for most common
// architectures: Llama/Mistral D=128, many others D=64/80/96). If D>128
// the kernel returns early (writes nothing to o[]) - falls back to the old
// attention_fused_kernel in that case.
// WHY #pragma unroll for the d loop instead of a plain "for (int d=0;d<D;++d)":
// acc[] is a private per-thread array - indexing it with a RUNTIME
// variable (runtime d) gets NO dynamic-index registers on
// Intel Iris (exactly the "GPU Timeout" cause hit with tier1[] in
// matmul_q4_kernel before) -> it MUST be unrolled to compile-time indices.
#define FLASH_MAX_D 128

// REWRITTEN (the real cause of the GPU hang/timeout): the previous version
// used a PRIVATE array acc[FLASH_MAX_D] (128 floats = 512 bytes) ALIVE through the whole
// tj loop (up to S iterations) IN EACH THREAD - Intel Iris has a
// very small register file, 512 bytes/thread for acc[] alone (not counting
// dot/sc/m_i/l_i/aOld/p) surely OVERFLOWS the registers, the driver has to spill to
// per-thread memory - EXACTLY the class of bug hit with tier1[] in
// matmul_q4_kernel (except tier1[] is a small LUT, while acc[] here is larger
// and lives much longer). THE REAL FIX (not falling back to non-flash): split
// D across A WHOLE THREADGROUP working together (as attention_decode_gqa_
// kernel does), acc[] moved into THREADGROUP MEMORY (no longer per-thread
// registers - threadgroup memory has NO dynamic-index register-overflow
// problem, it is real on-chip memory, not the
// register file). Still REAL flash attention: one pass over
// tj=0..ti, running max/sum, NO S*S materialisation, NO large
// private array.
#define ATTN_FLASH_TG 64

kernel void attention_fused_flash_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device const int& B [[buffer(4)]],
    device const int& H [[buffer(5)]],
    device const int& S [[buffer(6)]],
    device const int& D [[buffer(7)]],
    device const float& scale [[buffer(8)]],
    uint3 tgpigv [[threadgroup_position_in_grid]],   // .x = bh, .y = ti (uint3 - see the reason below)
    uint3 tidv   [[thread_position_in_threadgroup]]  // .x = 0..ATTN_FLASH_TG-1 - BAT BUOC dispatch threadgroup size = (ATTN_FLASH_TG,1,1)
    // MSL compile fix (same fix as applied to matmul_q4_kernel):
    // "expecting input declarations with either all scalar types or all
    // vector types with the same number of elements" - the Metal compiler
    // does NOT allow mixing a vector parameter (uint2 tgpig) with a scalar
    // parameter (uint tid) that both have thread-position attributes in the SAME
    // argument list. Fix: declare both as uint3 and take .xy/.x.
) {
    uint2 tgpig = tgpigv.xy;
    uint tid = tidv.x;
    uint bh = tgpig.x;
    uint ti = tgpig.y;
    if (bh >= (uint)(B * H) || ti >= (uint)S) return;
    if (D > FLASH_MAX_D) return;   // see the LIMIT at the top of the file (128, enough for most common architectures)

    // CHANGED (keep acc in "each thread's small RAM" = registers/private,
    // NO threadgroup memory for acc any more). FLASH_MAX_D=128,
    // ATTN_FLASH_TG=64 => each thread holds AT MOST 2 elements (FLASH_MAX_D/
    // ATTN_FLASH_TG). A FIXED-SIZE private array of that many
    // "slots" (not FLASH_MAX_D), accessed ONLY through #pragma
    // unroll with a STATIC slot index (0,1,...) - NEVER indexed by the
    // running variable "d" directly - so the compiler can keep it in registers instead
    // of spilling to scratch memory (exactly what caused the GPU Timeout
    // before with the dynamically indexed tier1[]/acc[]).
    #define ACC_SLOTS ((FLASH_MAX_D + ATTN_FLASH_TG - 1) / ATTN_FLASH_TG)
    float acc[ACC_SLOTS];
    #pragma unroll
    for (int s = 0; s < ACC_SLOTS; ++s) acc[s] = 0.0f;

    threadgroup float redBuf[ATTN_FLASH_TG];
    threadgroup float sM, sL, sAOld, sP;

    if (tid == 0u) { sM = -1e30f; sL = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base = bh * (uint)S * (uint)D;
    uint qOff = base + ti * (uint)D;

    for (uint tj = 0u; tj <= ti; ++tj) {
        uint kOff = base + tj * (uint)D;

        // --- dot(q_ti, k_tj) cooperatively: each thread adds part of D, then
        // tree-reduce in threadgroup memory (NOT a private array) ---
        float partial = 0.0f;
        for (int d = int(tid); d < D; d += ATTN_FLASH_TG) partial += q[qOff + (uint)d] * k[kOff + (uint)d];
        redBuf[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_FLASH_TG / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) redBuf[tid] += redBuf[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float dot = redBuf[0] * scale;   // every thread reads the same value after the last barrier

        // --- online-softmax update: ONLY thread 0 computes it (scalar, no need
        // for parallelism), broadcast through a threadgroup scalar to the whole group ---
        if (tid == 0u) {
            float m_old = sM;
            float new_m = max(m_old, dot);
            float aOld = exp(m_old - new_m);
            float p = exp(dot - new_m);
            sL = sL * aOld + p;
            sM = new_m;
            sAOld = aOld;
            sP = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float aOld = sAOld;
        float p = sP;
        // s is a STATIC slot (compile time through unroll) - d = tid + s*ATTN_FLASH_TG
        // is still runtime but ONLY used to read/write device memory (v[], valid
        // whatever D is), NOT to index the private acc[] array any more.
        #pragma unroll
        for (int s = 0; s < ACC_SLOTS; ++s) {
            int d = int(tid) + s * ATTN_FLASH_TG;
            if (d < D) acc[s] = acc[s] * aOld + p * v[kOff + (uint)d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // done with this tj, safe to overwrite redBuf in the next iteration
    }

    float invL = 1.0f / sL;
    #pragma unroll
    for (int s = 0; s < ACC_SLOTS; ++s) {
        int d = int(tid) + s * ATTN_FLASH_TG;
        if (d < D) o[qOff + (uint)d] = acc[s] * invL;
    }
}
#undef ACC_SLOTS

// ============================================================
// ATTENTION DECODE (GQA) - v2, REALLY PARALLEL + FENCE FIX.
// v1 used 1 thread/head (only nRep=4-8 threads in total) - wasting almost the whole GPU,
// measured ~4x slower than the old path (1.9s/token -> 8s/token).
// v2 uses 1 threadgroup (128 threads)/head + tree reduction, BUT the first
// version was wrong: it used threadgroup_barrier(mem_flags::mem_threadgroup) while
// `scores` is `device` memory (not `threadgroup`) - the
// mem_threadgroup flag ONLY synchronises threadgroup variables (shared_reduce),
// it does NOT guarantee other threads have finished writing `scores` (device memory)
// before the cross reads in step 3 - errors of ~0.02-0.1 (reading
// stale/partial scores). FIXED: mem_flags::mem_device_and_threadgroup
// for EVERY barrier in this kernel (fences both threadgroup and device memory).
//   q: [H, D], k/v: [S, D] (shared by H heads), o: [H, D]
//   scores: [H, S] scratch (device memory, READ/WRITTEN ACROSS threads)
// ============================================================
#define ATTN_DECODE_TG 128

// CHANGED (flash attention, NEW): the version above used 3 loops over S + one
// device buffer `scores[H,S]` as the intermediate (written in step 1, overwritten in step
// 2, read across threads in step 3) - with a wide device-memory fence (memory
// traffic ~3x the size of scores). This FLASH version makes ONE loop
// over BLOCKS of TG=128 keys at a time, keeping a running max/sum/output
// (the flash-attention algorithm applied to the 1-query
// - S-key decode case) - NO more reads/writes of `scores` (device), replaced
// by `pBlock` (threadgroup memory, only 512 bytes) holding the probabilities of
// the ONE block being processed. The `scores` parameter STAYS in the signature
// (unused) so the shim/Nim binding need NOT change (the old ABI still
// works unchanged).
kernel void attention_decode_gqa_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* o [[buffer(3)]],
    device float* scores [[buffer(4)]],   // NO longer used - kept so the ABI/shim does not change
    device const int& H [[buffer(5)]],
    device const int& S [[buffer(6)]],
    device const int& D [[buffer(7)]],
    device const float& scale [[buffer(8)]],
    uint h [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_reduce[ATTN_DECODE_TG];
    threadgroup float pBlock[ATTN_DECODE_TG];   // (unnormalised) probabilities of the key block being processed
    if (h >= (uint)H) return;
    if (D > FLASH_MAX_D) return;   // xem gioi han o attention_fused_flash_kernel

    uint qBase = h * (uint)D;

    float m_i = -1e30f;    // running max
    float l_i = 0.0f;      // running sum (already on the m_i scale)
    // Each thread keeps only the acc PART of (at most) one index d = tid (because
    // ATTN_DECODE_TG=128 >= FLASH_MAX_D=128 => each d is handled by EXACTLY one
    // thread, no private array at all -> no dynamic-index
    // register problem as in attention_fused_flash_kernel).
    float accD = 0.0f;
    bool hasD = (tid < (uint)D);

    for (uint blockStart = 0u; blockStart < (uint)S; blockStart += ATTN_DECODE_TG) {
        uint tj = blockStart + tid;
        bool valid = tj < (uint)S;

        float dot = -1e30f;
        if (valid) {
            dot = 0.0f;
            uint kBase = tj * (uint)D;
            for (int d = 0; d < D; ++d) dot += q[qBase + (uint)d] * k[kBase + (uint)d];
            dot *= scale;
        }

        // --- max of THIS block (not global - flash attention
        // rescales gradually, the global max is not needed first) ---
        shared_reduce[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_DECODE_TG / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float blockMax = shared_reduce[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float new_m = max(m_i, blockMax);
        float aOld = exp(m_i - new_m);
        float p = valid ? exp(dot - new_m) : 0.0f;
        pBlock[tid] = p;

        // --- sum of p over the block (tree-reduce) ---
        shared_reduce[tid] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = ATTN_DECODE_TG / 2; stride > 0; stride >>= 1) {
            if (tid < stride) shared_reduce[tid] += shared_reduce[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float blockSum = shared_reduce[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // make sure pBlock[] is written before the cross reads below

        l_i = l_i * aOld + blockSum;

        // --- output update: each thread (if hasD) reads ALL of
        // pBlock[] (of this block, TG elements) times v[.., d=tid] ---
        if (hasD) {
            float blockAcc = 0.0f;
            uint blockLen = min((uint)ATTN_DECODE_TG, (uint)S - blockStart);
            for (uint j = 0u; j < blockLen; ++j) {
                blockAcc += pBlock[j] * v[(blockStart + j) * (uint)D + tid];
            }
            accD = accD * aOld + blockAcc;
        }
        m_i = new_m;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // block done, safe to overwrite shared_reduce/pBlock in the next iteration
    }

    if (hasD) {
        o[qBase + tid] = accD / l_i;
    }
}

// ============================================================
// ATTENTION FUSED BACKWARD
// ============================================================
kernel void attention_fused_backward_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* s_matrix [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* dq [[buffer(5)]],
    device atomic_uint* dk [[buffer(6)]],
    device atomic_uint* dv [[buffer(7)]],
    device const int& B [[buffer(8)]],
    device const int& H [[buffer(9)]],
    device const int& S [[buffer(10)]],
    device const int& D [[buffer(11)]],
    device const float& scale [[buffer(12)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint bh = position.x;
    uint ti = position.y;
    
    if (bh < (uint)(B * H) && ti < (uint)S) {
        uint base_idx = bh * S * D;
        uint base_s = bh * S * S;
        
        float softmaxW[256];
        for (uint tj = 0; tj <= ti; ++tj) {
            softmaxW[tj] = s_matrix[base_s + ti * S + tj];
        }
        
        float dSoftmax[256];
        for (uint tj = 0; tj <= ti; ++tj) {
            float dotVal = 0.0f;
            for (int d = 0; d < D; ++d) {
                float dyVal = dy[base_idx + ti * D + d];
                atomicAddFloatDevice(&dv[base_idx + tj * D + d], softmaxW[tj] * dyVal);
                dotVal += dyVal * v[base_idx + tj * D + d];
            }
            dSoftmax[tj] = dotVal;
        }
        
        float dotSum = 0.0f;
        for (uint tj = 0; tj <= ti; ++tj) {
            dotSum += softmaxW[tj] * dSoftmax[tj];
        }
        
        for (uint tj = 0; tj <= ti; ++tj) {
            float dScore = softmaxW[tj] * (dSoftmax[tj] - dotSum) * scale;
            for (int d = 0; d < D; ++d) {
                dq[base_idx + ti * D + d] += dScore * k[base_idx + tj * D + d];
                atomicAddFloatDevice(&dk[base_idx + tj * D + d], dScore * q[base_idx + ti * D + d]);
            }
        }
    }
}
// ============================================================
// GATHER - used by permute/concat/split/stack on the GPU instead of a CPU loop.
// dst[id] = src[idx[id]].
// ============================================================
kernel void gather_kernel(
    device const float* src [[buffer(0)]],
    device const int*   idx [[buffer(1)]],
    device float*       dst [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    dst[id] = src[idx[id]];
}

// ============================================================
// ARGMAX (value + index) per threadgroup, the same tree-reduce algorithm
// as argmax_reduce (CUDA) / argmax.comp (Vulkan). Each threadgroup handles
// one (o,i) pair; the threadgroup size must be >= dimSize.
// ============================================================
kernel void argmax_kernel(
    device const float* x       [[buffer(0)]],
    device float*       outVal  [[buffer(1)]],
    device int*         outIdx  [[buffer(2)]],
    constant int&       outer   [[buffer(3)]],
    constant int&       dimSize [[buffer(4)]],
    constant int&       inner   [[buffer(5)]],
    uint pair [[threadgroup_position_in_grid]],
    uint d    [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    threadgroup float* sVal [[threadgroup(0)]],
    threadgroup int*   sIdx [[threadgroup(1)]]
) {
    if (pair >= uint(outer * inner)) return;
    int o = int(pair) / inner;
    int i = int(pair) % inner;

    if (int(d) < dimSize) {
        sVal[d] = x[(o * dimSize + int(d)) * inner + i];
        sIdx[d] = int(d);
    } else {
        sVal[d] = -FLT_MAX;
        sIdx[d] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            if (sVal[d + stride] > sVal[d]) {
                sVal[d] = sVal[d + stride];
                sIdx[d] = sIdx[d + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d == 0) {
        outVal[pair] = sVal[0];
        outIdx[pair] = sIdx[0];
    }
}

// ============================================================
// COMPARE - a real bitmask on the GPU. op: 0=gt 1=lt 2=eq 3=ge 4=le.
// ============================================================
kernel void compare_kernel(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device uchar*       mask [[buffer(2)]],
    constant int&        op  [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float av = a[id], bv = b[id];
    uchar r = 0;
    if (op == 0) r = (av >  bv) ? 1 : 0;
    else if (op == 1) r = (av <  bv) ? 1 : 0;
    else if (op == 2) r = (av == bv) ? 1 : 0;
    else if (op == 3) r = (av >= bv) ? 1 : 0;
    else if (op == 4) r = (av <= bv) ? 1 : 0;
    mask[id] = r;
}

// ============================================================
// DROPOUT RNG - counter-based xorshift32, the same formula as CUDA/Vulkan.
// ============================================================
inline uint xorshift32_m(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

kernel void dropout_rng_kernel(
    device float*  mask   [[buffer(0)]],
    constant uint& seed   [[buffer(1)]],
    constant float& p     [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint state = seed ^ (id * 2654435761u + 1u);
    state = xorshift32_m(state);
    state = xorshift32_m(state);
    float u = float(state >> 8) * (1.0f / 16777216.0f);
    mask[id] = (u < p) ? 0.0f : scale;
}

// ============================================================
// PERMUTE - the device decodes each index from the strides. No O(n) index
// array built on the host.
// ============================================================
kernel void permute_strided(
    device const float* src        [[buffer(0)]],
    device float*       dst        [[buffer(1)]],
    constant int*       newStrides [[buffer(2)]],
    constant int*       oldStrides [[buffer(3)]],
    constant int*       dimsMap    [[buffer(4)]],
    constant int&       ndim       [[buffer(5)]],
    uint linear [[thread_position_in_grid]]
) {
    int rem = int(linear);
    int oldLinear = 0;
    for (int i = 0; i < ndim; i++) {
        int idxI = rem / newStrides[i];
        rem = rem % newStrides[i];
        oldLinear += idxI * oldStrides[dimsMap[i]];
    }
    dst[linear] = src[oldLinear];
}

// ============================================================
// SPLIT - a contiguous slice along dim, the device computes the offsets.
// ============================================================
kernel void split_slice(
    device const float* src      [[buffer(0)]],
    device float*       dst      [[buffer(1)]],
    constant int&       dimTotal [[buffer(2)]],
    constant int&       thisSize [[buffer(3)]],
    constant int&       offset   [[buffer(4)]],
    constant int&       inner    [[buffer(5)]],
    uint linear [[thread_position_in_grid]]
) {
    int i = int(linear) % inner;
    int rem = int(linear) / inner;
    int d = rem % thisSize;
    int o = rem / thisSize;
    int srcLinear = (o * dimTotal + offset + d) * inner + i;
    dst[linear] = src[srcLinear];
}

// ============================================================
// CONCAT - the device picks the source through a small O(numTensors) table. flatSrc is
// flattened by the host (sequential memcpy, no per-element index computation).
// ============================================================
kernel void concat_gather(
    device const float* flatSrc      [[buffer(0)]],
    device const int*   srcBase      [[buffer(1)]],
    device const int*   srcDimSize   [[buffer(2)]],
    device const int*   srcDimOffset [[buffer(3)]],
    constant int&       numTensors   [[buffer(4)]],
    device float*       dst          [[buffer(5)]],
    constant int&       totalDim     [[buffer(6)]],
    constant int&       inner        [[buffer(7)]],
    uint linear [[thread_position_in_grid]]
) {
    int i = int(linear) % inner;
    int rem = int(linear) / inner;
    int dGlobal = rem % totalDim;
    int o = rem / totalDim;

    int ti = 0;
    for (int k = 0; k < numTensors; k++) {
        if (dGlobal < srcDimOffset[k] + srcDimSize[k]) { ti = k; break; }
    }
    int dLocal = dGlobal - srcDimOffset[ti];
    int srcLinear = srcBase[ti] + (o * srcDimSize[ti] + dLocal) * inner + i;
    dst[linear] = flatSrc[srcLinear];
}
