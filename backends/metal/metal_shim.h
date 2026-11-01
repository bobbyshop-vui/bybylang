// metal_shim.h - C interface exposed to Nim for the Metal compute backend (macOS only).
#ifndef METAL_SHIM_H
#define METAL_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if the machine has a Metal-capable GPU, 0 otherwise.
int metal_available(void);
// 1 when the Metal GPU is an Apple GPU (Apple Silicon), 0 for Intel/AMD.
int metal_is_apple_gpu(void);

// kernel_src: Metal Shading Language source (read from kernels/vecop_matmul.metal
// at Nim compile time through staticRead, NO longer hardcoded in the .m file).
// op: 0=add, 1=sub, 2=mul, 3=div. Returns 1 on success, 0 on error.
// error_out: if not NULL, set to point to an error string (free it with free()).
int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n, char** error_out);

// C(m x n) = A(m x k) * B(k x n), row-major. Returns 1 on success, 0 on error.
int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n, char** error_out);

// Like metal_matmul but uses Apple's MPSMatrixMultiplication (Metal Performance
// Shaders) instead of the hand-written tile kernel. MPS picks the algorithm
// (simdgroup_matrix on Apple Silicon, dedicated paths for Intel/AMD)
// and autotunes for the actual GPU - usually >= the hand-written kernel, often
// much faster. NO kernel_src needed (our own compiled MSL library is not
// used on this path). Returns 1 on success, 0 on error
// (e.g. no MetalPerformanceShaders framework / unsupported GPU) -
// the caller MUST fall back to metal_matmul() when this returns 0.
int metal_matmul_mps(const float* a, const float* b, float* c, int m, int k, int n, char** error_out);

// Matmul directly on PACKED int4-asymmetric weights (no fp32 dequant
// on the CPU first). wq: packed bytes [N * ceil(K/2)]. scales/zeros:
// [N * nGroupsPerRow] (nGroupsPerRow = ceil(K/groupSize), groupSize<=0 means
// one group per row). See the detailed notes in vecop_matmul.metal
// (matmul_q4_kernel) before use - NOT YET RUN ON A REAL GPU.
int metal_matmul_q4(const char* kernel_src, const float* a, const unsigned char* wq,
                     const float* scales, const float* zeros, float* c,
                     int m, int k, int n, int groupSize, int nGroupsPerRow, char** error_out);

// Runs 2 INDEPENDENT matmuls (a1*b1->c1 and a2*b2->c2) in the SAME command
// buffer / one pipeline compile, with ONE commit+wait. Used to
// dispatch 2 independent matmuls (e.g. dW and dX in backward)
// without paying the overhead of 2 separate commit+waits.
// Returns 1 on success, 0 on error.
int metal_matmul2(const char* kernel_src,
                   const float* a1, const float* b1, float* c1, int m1, int k1, int n1,
                   const float* a2, const float* b2, float* c2, int m2, int k2, int n2, char** error_out);

int metal_activation(const char* kernel_src, int op, const float* x, float* y, int n, float p, char** error_out);
int metal_softmax(const char* kernel_src, const float* x, float* y, int rows, int cols, char** error_out);
int metal_layernorm(const char* kernel_src, const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps, char** error_out);
int metal_embedding_lookup(const char* kernel_src, const float* table, const int* indices, float* y, int vocab, int dim, int num_indices, char** error_out);

// op: 0=relu, 1=sigmoid, 2=tanh, 3=apflu (same code table as metal_activation).
// Calls activation_backward_kernel in vecop_matmul.metal (the kernel existed but
// was NOT bound to a C function -> gpubackend.nim had no way to call it,
// so apflu backward always fell back to the CPU even with the metal backend).
int metal_activation_backward(const char* kernel_src, int op, const float* x, const float* dy, float* dx, int n, char** error_out);

// Calls layernorm_backward_kernel. dx: rows*cols elements, dgamma/dbeta: cols elements.
int metal_layernorm_backward(const char* kernel_src, const float* dy, const float* x,
                              const float* gamma, const float* beta,
                              float* dx, float* dgamma, float* dbeta,
                              int rows, int cols, float eps, char** error_out);

// Causal fused attention forward. q/k/v: B*H*S*D elements, o: B*H*S*D,
// s_matrix: B*H*S*S. `mask` is unused by the kernel (causality comes from the
// tj<=ti loop like the CPU reference) but is still accepted so the signature matches
// beAttentionFused in backend.nim.
int metal_attention_fused(const char* kernel_src, const float* q, const float* k, const float* v,
                           float* o, float* s_matrix, int B, int H, int S, int D, float scale, char** error_out);

// Causal fused attention backward. dq/dk/dv: B*H*S*D elements EACH, MUST
// be zero-initialised by the caller (the kernel uses "+=" / atomic adds on the
// output buffers, it does not overwrite them like the forward).
int metal_attention_fused_backward(const char* kernel_src, const float* q, const float* k, const float* v,
                                    const float* s_matrix, const float* dy,
                                    float* dq, float* dk, float* dv,
                                    int B, int H, int S, int D, float scale, char** error_out);

// ─────────────────────────────────────────────────────────────────────────
// Resident API: many ops encoded into the SAME MTLCommandBuffer (as metal_matmul2
// does for 2 matmuls), with ONE commit+waitUntilCompleted in
// metal_session_end(). Returned buffers are opaque handles (pointers to an id<MTLBuffer>
// that was CFBridgingRetain'ed) - alive until metal_buffer_free is called.
// NOTE: written without a real Metal GPU (generated on Linux,
// no macOS/Metal framework to build+run) - it follows 100% the idioms
// already WORKING in metal_vecop/metal_matmul/metal_matmul2
// above (same buffer creation, same pipeline lookup, same dispatch), only
// sharing one command buffer instead of creating one per command. It must
// be built + tested on a real Mac before being used for real training.
typedef void* MetalBufferHandle;

// Starts a session: creates a new command buffer + encoder kept in static variables.
// Call before the first resident op of a layer/forward pass.
int metal_session_begin(const char* kernel_src, char** error_out);

// Uploads host data -> MTLBuffer (CPU memcpy; Apple Silicon has unified memory,
// so there is NO real H2D DMA as with discrete CUDA - still a separate function for clarity
// and to work on Intel Macs with a discrete GPU, where the storage mode
// differs and the driver really needs a transfer).
MetalBufferHandle metal_upload(const float* data, int n);
// Session-only borrowed upload: reuses page-aligned host storage when Metal
// accepts it. The caller must keep `data` alive and unchanged until the
// session has ended and the buffer has been freed. Falls back to a copied
// Shared buffer when zero-copy is not supported for this address/length.
MetalBufferHandle metal_upload_borrowed(const float* data, int n);
// Zero-copy view of long-lived host memory (a read-only model mmap) for GPUs
// that share system memory. `data` must be page-aligned and readable up to the
// next page boundary after data+bytes; the caller keeps it mapped until the
// buffer is freed. NULL means the caller should copy instead (discrete GPU,
// misaligned region or longer than maxBufferLength).
MetalBufferHandle metal_wrap_host_nocopy(const void* data, size_t bytes);
MetalBufferHandle metal_upload_indices(const int* data, int n);
// Allocates a resident output buffer (the kernel overwrites it), data NOT initialised.
MetalBufferHandle metal_alloc_scratch(int n);

// NEW: MTLHeap - for allocating MANY buffers at once with the same
// lifetime (typically the resident KV cache, each layer needs nKVHeads*2
// K/V buffers). metal_alloc_scratch() calls [gDevice newBufferWithLength:] DIRECTLY
// each time - a full device-level allocation (mapping shared CPU/GPU
// pages) with a significant fixed cost on the old Intel Iris driver. For
// the resident KV: nKVHeads*2*nLayers separate calls (e.g. 32 heads*2*32 layers =
// 2048 calls) measured at tens of seconds AT STARTUP (before the first
// token) - a REAL performance bug, not an impression.
// Fix: allocate ONE large region with an MTLHeap (once per layer, or once for the
// whole model) and "suballocate" each child buffer from it with
// metal_alloc_from_heap() - suballocating from a reserved heap is MUCH CHEAPER
// than calling the device each time (as Apple's Metal Best Practices Guide
// recommends in "Heaps" for exactly this "many
// buffers of the same size and lifetime" case). It still returns a normal
// MetalBufferHandle - EVERY call site using metal_alloc_scratch()
// elsewhere (matmul/attention/...) needs NO change, because a buffer suballocated from a
// heap is still a valid id<MTLBuffer>, usable exactly like a directly allocated one.
void* metal_alloc_heap(long bytes);
MetalBufferHandle metal_alloc_from_heap(void* heap, int n);
void metal_free_heap(void* heap);
// Encodes one more op into the open command buffer (NO commit) - returns 1/0.
int metal_vecop_enc(int op, MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int n);
int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n);
int metal_scale_enc(MetalBufferHandle x, float scalar, int n);
int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols);
int metal_copy_into_enc(MetalBufferHandle src, MetalBufferHandle dst, int srcOffsetElems, int dstOffsetElems, int n);
int metal_rmsnorm_enc(MetalBufferHandle x, MetalBufferHandle gamma,
                       MetalBufferHandle y, int rows, int cols, float eps);
int metal_layernorm_enc(MetalBufferHandle x, MetalBufferHandle gamma, MetalBufferHandle beta,
                         MetalBufferHandle y, int rows, int cols, float eps);
int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices);
int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n);
// C(m,n) = A(m,k) @ B(n,k)^T - B is passed in its original [n,k] layout
// (no CPU transpose first). Used by attention decode (K/V cache).
int metal_matmul_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n);
// metal_matmul_nt_enc with B starting bOffsetBytes (a multiple of 4) into its
// buffer, e.g. a weight wrapped inside a model mmap.
int metal_matmul_nt_off_enc(MetalBufferHandle a, MetalBufferHandle b, int bOffsetBytes,
                            MetalBufferHandle c, int m, int k, int n);
// Resident prefix GEMM. C may be a column tile of a wider row-major matrix.
int metal_matmul_mps_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c,
                            int m, int k, int n, int cRowStride, int cColumnOffset);
int metal_mps_supported(void);
// Runtime machine check. Returns 1 on Apple Silicon (including Rosetta), 0 otherwise.
int metal_is_apple_silicon(void);
// Tune Q4 workgroup/kernel geometry outside an open inference session.
int metal_prepare_q4_tuned(MetalBufferHandle w, MetalBufferHandle scales,
    MetalBufferHandle zeros, int k, int n, int groupSize, int groupsPerRow, int m, int wqOffset);
// In-place rotate_half RoPE in the open session. x: [T,Hn,D]. cosT/sinT: [T,D/2]
// (already sliced on the CPU before upload). No CPU reads in between.
int metal_rope_enc(MetalBufferHandle x, MetalBufferHandle cosT, MetalBufferHandle sinT,
                    int T, int Hn, int D);
// Encodes attention_fused_kernel into the open session (NO commit). q,k,v are
// resident buffers (metal_upload before), o/s_matrix are scratch buffers
// (metal_alloc_scratch). S must be <= 256 (kernel limit, like the non-session version).
int metal_attention_fused_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                               MetalBufferHandle o, MetalBufferHandle s_matrix,
                               int B, int H, int S, int D, float scale);
// NEW: flash variant (online softmax, no s_matrix) of attention_fused
// - for INFERENCE prefill. D (head_dim) must be <= 128. NOT for
// training (backward needs s_matrix -> still calls the old metal_attention_fused_enc).
int metal_attention_fused_flash_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                     MetalBufferHandle o,
                                     int B, int H, int S, int D, float scale);
// NEW: attention decode for GQA - 1 query/head (the new token), K/V share
// S rows for all H heads of one GQA group. No S<=256 limit (scores
// is an H*S scratch buffer allocated at the actual size).
int metal_attention_decode_gqa_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                    MetalBufferHandle o, MetalBufferHandle scores,
                                    int H, int S, int D, float scale);
// NEW: FUSED attention decode for ALL H heads in EXACTLY one
// dispatch (instead of the CPU loop of 32 heads x 5 dispatches/head before).
// K/V must be CONTIGUOUS per head: a buffer of HKV*maxSeqLen*D,
// offset of head hkv = hkv*maxSeqLen*D (use kResidentCombined/vResidentCombined
// in Nim, NOT the old kResident/vResident, because heap suballocation does not
// guarantee even spacing between heads). Q already has RoPE+scale applied before
// this call (same as the old path). The kernel uses the classic 4-phase softmax,
// written from scratch - it shares NO code with
// metal_attention_decode_gqa_enc (that kernel is known to produce wrong results).
int metal_attention_decode_all_heads_enc(MetalBufferHandle q, MetalBufferHandle kAll, MetalBufferHandle vAll,
                                          MetalBufferHandle o,
                                          int H, int HKV, int D, int S, int maxSeqLen);
// Prefill flash attention: o[T, H, D] = causal softmax(q K^T * scale) V over the
// cache positions 0 .. pos + T - 1 (q position-major [T, H, D], K/V
// [HKV, maxSeqLen, D]). D <= 128. See attention_prefill_flash.metal.
int metal_attention_prefill_flash_enc(MetalBufferHandle q, MetalBufferHandle kAll, MetalBufferHandle vAll,
                                      MetalBufferHandle o, int T, int H, int HKV, int D, int pos,
                                      int maxSeqLen, float scale);
// KV cache append: dst[h, pos + t, :] = src[t, h, :].
int metal_kv_scatter_enc(MetalBufferHandle src, MetalBufferHandle dst, int T, int HKV, int D,
                         int pos, int maxSeqLen);
// Uploads raw bytes (packed int4 weights) -> a resident MTLBuffer, for
// metal_matmul_q4_enc below. The handle lives until metal_buffer_free -
// UPLOAD ONCE when loading the model and keep the handle for the process lifetime, do NOT
// call it again per token (that is the whole point of resident weights).
// CHANGED (a real bug hit at run time): metal_upload/metal_upload_bytes (used for
// resident weights - called BEFORE any metal_session_begin/metal_matmul,
// right at model load) did NOT call ensureMetalInit() like every other function
// in this file -> gDevice still NULL -> newBufferWithBytes on a nil
// device returned nil -> mass "buffer creation FAILED" (seen directly in the log of
// a real Intel Iris machine). An explicit init function is called ONCE before the first
// resident upload - ensureMetalInit() uses dispatch_once, so calling it repeatedly
// is safe/cheap (it only really initialises on the first call).
int metal_init(const char* kernel_src);

MetalBufferHandle metal_upload_bytes(const unsigned char* data, int n);
// NEW: Private-storage variant - ONLY for resident weights (GPU-only
// after the upload). metal_upload/metal_upload_bytes (Shared) are still used for
// every activation buffer (touched by the CPU: uploaded every token or read back).
MetalBufferHandle metal_upload_private(const float* data, int n);
MetalBufferHandle metal_upload_bytes_private(const unsigned char* data, int n);
// Resident/session variant of metal_matmul_q4 (see metal_matmul_q4 above):
// wq/scales/zeros are handles ALREADY uploaded ONCE (metal_upload_bytes/metal_upload),
// NOT copied again on each call - only the activation `a` is small/changes every
// token. Encoded into the open session (between metal_session_begin/metal_session_end),
// exactly like metal_matmul_enc but reading packed int4 instead of fp32.
// wqOffset: byte position of the packed weights inside wq (any value; a
// weight wrapped in place inside an unaligned .nimq mmap).
int metal_matmul_q4_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                         MetalBufferHandle zeros, MetalBufferHandle c,
                         int m, int k, int n, int groupSize, int nGroupsPerRow, int wqOffset);
// NEW: Q4 matmul with a REAL batch M, only for M>1 (prefill). The kernel
// reads each weight byte EXACTLY ONCE and reuses it for MBLOCK=4 consecutive rows -
// unlike metal_matmul_q4_enc (dispatch Y=M, each row re-reads all the
// weights independently - right for M=1/decode, but very
// inefficient when called with M>1). See matmul_q4_batched.metal.
int metal_matmul_q4_batched_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                                 MetalBufferHandle zeros, MetalBufferHandle c,
                                 int m, int k, int n, int groupSize, int nGroupsPerRow, int wqOffset);

// FP4/FP8/quant 1-byte/weight: wq=uchar[N*K], lut=float[256] (buildDecodeLUT*scale).
// Kernel: matmul_fp4_lut_kernel_ndst2_paired (vecop_matmul.metal). Session+TG nhu Q4.
int metal_matmul_fp4_lut_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle lut,
                              MetalBufferHandle c, int m, int k, int n);

// NEW: an activation_enc variant with parameter p (for "pow", opcode 7).
// metal_activation_enc (old signature, no p) is unchanged and calls this function
// with p=0.0 so the existing relu/sigmoid/tanh/apflu call sites keep working.
int metal_activation_enc_p(int op, MetalBufferHandle x, MetalBufferHandle y, int n, float p);
// activation_backward_kernel in a session - for apflu_backward
// (op=3, alpha/beta fixed at 0.1/1.0 like the immediate metalApfluBackward).
int metal_activation_backward_enc(int op, MetalBufferHandle x, MetalBufferHandle dy, MetalBufferHandle dx, int n);
int metal_layernorm_backward_enc(MetalBufferHandle dy, MetalBufferHandle x,
                                  MetalBufferHandle gamma, MetalBufferHandle beta,
                                  MetalBufferHandle dx, MetalBufferHandle dgamma, MetalBufferHandle dbeta,
                                  int rows, int cols, float eps);
int metal_attention_fused_backward_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                        MetalBufferHandle s_matrix, MetalBufferHandle dy,
                                        MetalBufferHandle dq, MetalBufferHandle dk, MetalBufferHandle dv,
                                        int B, int H, int S, int D, float scale);
int metal_gather_enc(MetalBufferHandle src, MetalBufferHandle idx, MetalBufferHandle dst, int n);
int metal_argmax_enc(MetalBufferHandle x, MetalBufferHandle outVal, MetalBufferHandle outIdx,
                      int outer, int dimSize, int inner);
int metal_compare_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle mask, int op, int n);
int metal_dropout_mask_enc(MetalBufferHandle mask, unsigned int seed, float p, float scale, int n);
int metal_permute_enc(MetalBufferHandle src, MetalBufferHandle dst,
                       MetalBufferHandle newStrides, MetalBufferHandle oldStrides,
                       MetalBufferHandle dimsMap, int ndim, int n);
int metal_split_slice_enc(MetalBufferHandle src, MetalBufferHandle dst,
                           int dimTotal, int thisSize, int offset, int inner, int n);
int metal_concat_gather_enc(MetalBufferHandle flatSrc, MetalBufferHandle srcBase,
                             MetalBufferHandle srcDimSize, MetalBufferHandle srcDimOffset,
                             int numTensors, MetalBufferHandle dst, int totalDim, int inner, int n);
// ONE commit + wait for every op encoded since metal_session_begin.
// After this call every MetalBufferHandle of the session holds the right data,
// readable with metal_buffer_read.
int metal_session_end(void);
// NEW: "commitAndContinue" - commits the CURRENT buffer (WITHOUT waiting -
// no waitUntilCompleted) and opens a new buffer RIGHT AWAY to keep encoding.
// See the full explanation (following Apple's MPSCommandBuffer.commitAndContinue /
// PyTorch MPS SyncType::COMMIT_ADAPTIVE) in metal_shim.m.
// Call periodically during a long session (e.g. every N layers) to bound the size
// of EACH buffer (avoids the watchdog) WITHOUT paying a
// CPU<->GPU sync each time (unlike calling metal_session_end() + metal_session_begin()
// in the middle - SLOW because that is a REAL wait). Returns 1 on
// success, 0 on failure (e.g. no open session).
int metal_session_commit_continue(void);
// NEW: the REAL NSError description of the last failed metal_session_end()
// (commit/waitUntilCompleted not Completed) - this error used to go ONLY
// to NSLog (Console/syslog) and could not reach Nim, so the caller
// (nim_inference.nim, Mode 2/Mode 3 paths) only knew a generic "sessionEnd failed",
// not the real reason (e.g. GPU watchdog timeout, out of
// memory, etc.). Returns "" if the last call did NOT fail (or
// metal_session_end was never called). The returned pointer lives until the
// NEXT metal_session_end() (a static NSString holds a strong ref) - Nim
// must copy it right away (with $cstring), not keep the pointer.
const char* metal_get_last_session_error(void);
// NEW: recreates the MTLCommandQueue (gQueue) when it is "poisoned" after a real GPU error
// (Timeout/StackOverflow/PageFault...). See the details in
// metal_shim.m::metal_recover_queue. Returns 1 on success, 0 on failure.
// Call right after metal_session_end() returns 0 (failure) BEFORE opening a
// new session/reading results - otherwise every later session stays
// "Ignored" forever and sessionRead returns OLD/GARBAGE data.
int metal_recover_queue(void);
// Reads data back to the host (valid only AFTER metal_session_end of the session that produced the handle).
int metal_buffer_read(MetalBufferHandle h, float* outData, int n);
// Frees right away when no longer needed (e.g. an intermediate activation after use
// in backward) - safe to call at any time, even in the middle of an open
// session, because the buffer is only really released once ARC sees no reference
// left in the running command buffer (Metal retains buffers when encoding).
void metal_buffer_free(MetalBufferHandle h);

// REAL GPU PORT (round 2): gather (permute/concat/split/stack), argmax
// (value+index), compare (bitmask), dropout RNG - kernels appended to
// the end of vecop_matmul.metal ("gather_kernel"/"argmax_kernel"/"compare_kernel"
// /"dropout_rng_kernel"), sharing kMetalKernelSrc, no separate
// library. srcLen is the REAL element count of the source buffer (idx[] points
// into it) - separate from n (OUTPUT/idx element count) because for concat the
// source buffer holds several tensors and is longer than each single gather.
int metal_gather(const char* kernelSrc, const float* src, int srcLen,
                  const int* idx, float* dst, int n, char** error_out);
int metal_argmax(const char* kernelSrc, const float* x, float* outVal,
                  int* outIdx, int outer, int dimSize, int inner, char** error_out);
int metal_compare(const char* kernelSrc, const float* a, const float* b,
                   unsigned char* mask, int op, int n, char** error_out);
int metal_dropout_mask(const char* kernelSrc, float* mask,
                        unsigned int seed, float p, float scale, int n, char** error_out);

// REAL GPU PORT (round 3): permute/split/concat - the device decodes the indices
// from a small O(rank)/O(numTensors) table, no O(n) index array built on the
// host (see permute_strided/split_slice/concat_gather in
// vecop_matmul.metal).
int metal_permute(const char* kernelSrc, const float* src, float* dst,
                   const int* newStrides, const int* oldStrides,
                   const int* dimsMap, int ndim, int n, char** error_out);
int metal_split_slice(const char* kernelSrc, const float* src, int srcLen, float* dst,
                       int dimTotal, int thisSize, int offset, int inner, int n, char** error_out);
int metal_concat_gather(const char* kernelSrc, const float* flatSrc, int flatLen,
                         const int* srcBase, const int* srcDimSize,
                         const int* srcDimOffset, int numTensors,
                         float* dst, int totalDim, int inner, int n, char** error_out);

/* ============================================================
 * LOW-LEVEL session — maps 1:1 to MTLComputeCommandEncoder
 * Paste these lines into metal_shim.h BEFORE the final #ifdef __cplusplus / #endif.
 * ============================================================ */

/* Selects a kernel compiled into gLibrary (setComputePipelineState). */
int metal_enc_pipeline(const char* kernelName);

/* Binds an MTLBuffer to [[buffer(index)]]. offsetBytes is usually 0. */
int metal_enc_bind_buffer(int index, MetalBufferHandle buf, int offsetBytes);

/* Small constants through setBytes (int / float / raw). */
int metal_enc_bind_int(int index, int v);
int metal_enc_bind_float(int index, float v);
int metal_enc_bind_bytes(int index, const void* data, int nBytes);

/* dispatchThreadgroups — tgX=0 => uses the threadExecutionWidth of the current pipeline. */
int metal_enc_dispatch(int gridX, int gridY, int gridZ, int tgX, int tgY, int tgZ);

/* dispatchThreads (for kernels using thread_position_in_grid instead of threadgroups). */
int metal_enc_dispatch_threads(int threadsX, int threadsY, int threadsZ,
                               int tgX, int tgY, int tgZ);

/* memoryBarrierWithScope: MTLBarrierScopeBuffers */
int metal_enc_barrier(void);

/* setThreadgroupMemoryLength:atIndex: — MUST be called before every dispatch of a
 * JIT kernel (lowerBytecodeToMetal on the Nim side always declares a
 * threadgroup(0) parameter, even for kernels that use no threadgroup
 * memory — without setting the length Metal asserts in
 * dispatchThreads). index=0, nBytes=4 is enough for kernels without real TG
 * memory; kernels with reductions (RMSNorm...) need more bytes. */
int metal_enc_set_threadgroup_bytes(int index, int nBytes);

/* threadExecutionWidth of the selected pipeline (after metal_enc_pipeline). */
int metal_enc_simd_width(void);

/* maxTotalThreadsPerThreadgroup of the current pipeline. */
int metal_enc_max_tg_threads(void);



/* JIT: compile Metal source luc runtime (bytecode lower -> kernel moi). */
int metal_jit_compile(const char* source, const char* kernelName);


int metal_device_max_tg_threads(void);
int metal_device_max_tg_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
