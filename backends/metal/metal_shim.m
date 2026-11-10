// metal_shim.m - Metal compute backend implementation (macOS only).
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import "metal_shim.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysctl.h>
#include <math.h>

// ─────────────────────────────────────────────────────────────
// matmul_q4 kernel A/B TEST (NEW): change only the 2 lines below to
// try another variant, NO scattered edits in metal_matmul_q4() and
// metal_matmul_q4_enc() (2 functions, each with its own kernel name and
// kMMQ4_NDST - easy to forget one of the 4 places -> WRONG rowBlocks
// -> WRONG results or a crash, see the explanation in vecop_matmul.metal).
//
//   "matmul_q4_kernel"              NDST=8  <- DEFAULT, measured as
//                                              STABLE (0.86 tok/s,
//                                              1156.6 ms/token) on
//                                              Intel Iris Plus 640.
//   "matmul_q4_kernel_ndst2_paired" NDST=2  <- experimental, like
//   "matmul_q4_kernel_ndst4_paired" NDST=4     ggml's N_R0_Q4_K=2,
//                                              NOT YET RUN ON A
//                                              REAL METAL GPU.
//
// Change BOTH lines below (kernel name + NDST) TOGETHER when testing:
#define MMQ4_KERNEL_NAME_ACTIVE @"matmul_q4_kernel_ndst2_paired"
#define MMQ4_NDST_ACTIVE 2

// ─────────────────────────────────────────────────────────────
// Global state, initialised once, alive until the process exits.
// ─────────────────────────────────────────────────────────────
static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLLibrary> gLibrary = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *gPipelineCache = nil;
static dispatch_once_t gInitOnce;

// MTLCreateSystemDefaultDevice() returns EXACTLY ONE "default" GPU chosen
// by the system. Apple Silicon Macs have one GPU so that is fine, but
// Intel Macs can have SEVERAL GPUs (Intel iGPU + AMD dGPU, or an extra
// eGPU) -- the default device is sometimes not the GPU really "alive"
// (e.g. a dGPU asleep because of automatic graphics switching, or the app lacks permission to
// power on the dGPU). MTLCopyAllDevices() (macOS only, not on
// iOS) lists every GPU so we choose/try them instead of trusting one default.
static id<MTLDevice> pickMetalDeviceUncached(void) {
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    if (all.count == 0) {
        NSLog(@"[metal] MTLCopyAllDevices() rong, thu MTLCreateSystemDefaultDevice()");
        return MTLCreateSystemDefaultDevice();
    }

    NSLog(@"[metal] Tim thay %lu GPU:", (unsigned long)all.count);
    for (id<MTLDevice> d in all) {
        NSLog(@"[metal]   - %@ (lowPower=%d, removable=%d, headless=%d)",
              d.name, d.isLowPower, d.isRemovable, d.isHeadless);
    }

    // Preferred: a discrete/high-performance GPU (not lowPower) that is not headless.
    for (id<MTLDevice> d in all) {
        if (!d.isLowPower) {
            NSLog(@"[metal] Chon GPU hieu nang cao: %@", d.name);
            return d;
        }
    }
    // No discrete GPU -> take the first GPU found (e.g. the integrated iGPU).
    NSLog(@"[metal] No discrete GPU, using the first GPU: %@", all[0].name);
    return all[0];
}

// Picks the GPU ONCE and logs ONCE (metal_available() and others are called hundreds of times).
static id<MTLDevice> pickMetalDevice(void) {
    static id<MTLDevice> cached = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ cached = pickMetalDeviceUncached(); });
    return cached;
}

// Initialises device + queue + library ONCE for the whole process.
// Returns NO on failure (e.g. no GPU, or an MSL compile error).
static BOOL ensureMetalInit(const char* kernel_src) {
    __block BOOL ok = YES;
    dispatch_once(&gInitOnce, ^{
        gDevice = pickMetalDevice();
        if (!gDevice) { NSLog(@"[metal] no Metal GPU found"); ok = NO; return; }

        gQueue = [gDevice newCommandQueue];
        if (!gQueue) { NSLog(@"[metal] newCommandQueue returned nil"); ok = NO; return; }

        NSString *source = [NSString stringWithUTF8String:kernel_src];
        NSError *error = nil;
        gLibrary = [gDevice newLibraryWithSource:source options:nil error:&error];
        if (!gLibrary) { NSLog(@"[metal] newLibraryWithSource FAILED (compile 1 lan luc khoi dong): %@", error); ok = NO; return; }

        gPipelineCache = [NSMutableDictionary dictionary];
        NSLog(@"[metal] Initialised: device=%@, kernel library compiled once.", gDevice.name);
        // NEW: prints CLEARLY which matmul_q4 kernel is active and its
        // NDST - avoids "did I change it or not" during A/B tests
        // (see MMQ4_KERNEL_NAME_ACTIVE/MMQ4_NDST_ACTIVE at the top of the file). This log
        // is ALWAYS shown (no BYBY_DEBUG_PATH needed), like the GPU log
        // above - it is static configuration, not runtime debugging.
        NSLog(@"[metal] matmul_q4 dang active: kernel=%@ NDST=%d", MMQ4_KERNEL_NAME_ACTIVE, MMQ4_NDST_ACTIVE);
    });
    return ok && gDevice != nil && gLibrary != nil;
}

// Gets the cached pipeline state by function name; the pipeline is compiled (the source is not
// recompiled) only the first time a name is seen, later calls hit the cache.
static id<MTLComputePipelineState> getPipeline(NSString *fname) {
    id<MTLComputePipelineState> cached = gPipelineCache[fname];
    if (cached) return cached;

    id<MTLFunction> fn = [gLibrary newFunctionWithName:fname];
    if (!fn) { NSLog(@"[metal] newFunctionWithName:%@ returned nil", fname); return nil; }

    NSError *error = nil;
    MTLComputePipelineDescriptor *pdesc = [MTLComputePipelineDescriptor new];
    pdesc.computeFunction = fn;
    pdesc.maxCallStackDepth = 32;
    id<MTLComputePipelineState> pipeline =
        [gDevice newComputePipelineStateWithDescriptor:pdesc
                                               options:MTLPipelineOptionNone
                                            reflection:nil
                                                 error:&error];
    if (!pipeline) { NSLog(@"[metal] newComputePipelineStateWithFunction FAILED cho %@: %@", fname, error); return nil; }

    gPipelineCache[fname] = pipeline;
    return pipeline;
}

// Picks the largest matmul_kernel_t{8,16,32} variant the running GPU REALLY
// ALLOWS, measured with the pipeline.maxTotalThreadsPerThreadgroup of that
// pipeline (not guessed from the GPU name / lowPower flag - which can differ between
// drivers/OS versions). Cached once because the result does not change during
// the process (same device, same library).
static int gMatmulTile = 0;                    // 0 = not chosen yet
static id<MTLComputePipelineState> gMatmulPipeline = nil;

static id<MTLComputePipelineState> pickMatmulPipeline(int *outTile) {
    if (gMatmulPipeline != nil) { *outTile = gMatmulTile; return gMatmulPipeline; }
    // CHANGED: NO auto choice of 16/32 - measured on Iris Plus 640 as not noticeably
    // faster, and since the NSLog only runs on the FIRST CALL (even with
    // a warm-up) it could still coincide with generation if the warm-up was skipped by the JIT/
    // compiler with a tiny input. Hard-wired to t8 - known
    // stable, simple, no runtime measurement any more.
    id<MTLComputePipelineState> p = getPipeline(@"matmul_kernel_t8");
    gMatmulPipeline = p;
    gMatmulTile = 8;
    *outTile = 8;
    return p;
}

int metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = pickMetalDevice();
        return device != nil ? 1 : 0;
    }
}

// 1 when the Metal GPU is an Apple GPU (Apple Silicon), 0 for Intel/AMD.
int metal_is_apple_gpu(void) {
    @autoreleasepool {
        id<MTLDevice> device = gDevice ? gDevice : pickMetalDevice();
        if (!device) return 0;
        if (@available(macOS 10.15, *)) return [device supportsFamily:MTLGPUFamilyApple1] ? 1 : 0;
        return 0;
    }
}

int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n, char** error_out) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) {
            if (error_out) *error_out = strdup("ensureMetalInit failed");
            return 0;
        }

        id<MTLComputePipelineState> pipeline = getPipeline(@"vecop_kernel");
        if (!pipeline) {
            if (error_out) *error_out = strdup("getPipeline(vecop_kernel) returned nil");
            return 0;
        }

        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [gDevice newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) {
            if (error_out) *error_out = strdup("[metal_vecop] tao buffer FAILED");
            return 0;
        }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&op length:sizeof(int) atIndex:3];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSString *err = [NSString stringWithFormat:@"[metal_vecop] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error];
            if (error_out) *error_out = strdup([err UTF8String]);
            return 0;
        }

        memcpy(c, [bufC contents], bytes);
        return 1;
    }
}

int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n, char** error_out) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) {
            if (error_out) *error_out = strdup("ensureMetalInit failed");
            return 0;
        }

        int tile = 8;
        id<MTLComputePipelineState> pipeline = pickMatmulPipeline(&tile);
        if (!pipeline) {
            if (error_out) *error_out = strdup("pickMatmulPipeline returned nil");
            return 0;
        }

        size_t bytesA = (size_t)m * (size_t)k * sizeof(float);
        size_t bytesB = (size_t)k * (size_t)n * sizeof(float);
        size_t bytesC = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [gDevice newBufferWithBytes:b length:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) {
            if (error_out) *error_out = strdup("[metal_matmul] tao buffer FAILED");
            return 0;
        }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&m length:sizeof(int) atIndex:3];
        [encoder setBytes:&k length:sizeof(int) atIndex:4];
        [encoder setBytes:&n length:sizeof(int) atIndex:5];

        NSUInteger t = (NSUInteger)tile;
        MTLSize groupSize = MTLSizeMake(t, t, 1);
        MTLSize threadgroupCount = MTLSizeMake(((NSUInteger)n + t - 1) / t, ((NSUInteger)m + t - 1) / t, 1);
        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSString *err = [NSString stringWithFormat:@"[metal_matmul] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error];
            if (error_out) *error_out = strdup([err UTF8String]);
            return 0;
        }

        memcpy(c, [bufC contents], bytesC);
        return 1;
    }
}

// ADDED: matmul through MPS instead of the hand-written kernel. Advantages over
// matmul_kernel_t{8,16,32}: MPS picks its own algorithm/tiling for
// the running GPU (including simdgroup_matrix on Apple Silicon, which the
// hand-written MSL kernel above does NOT use), no tile-size guessing and no
// GPU Timeout from oversized threadgroups - MPS guarantees a valid dispatch
// for the current GPU. Drawbacks: an extra step wrapping MPSMatrix around the MTLBuffer,
// and availability depends on MPS support of the GPU (checked through
// MPSSupportsMTLDevice) - so this function MUST have a fallback in the caller.
int metal_matmul_mps(const float* a, const float* b, float* c, int m, int k, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    // Hard policy: M=1 decode must use native kernels, even for direct callers.
    if (m <= 1) return 0;
    @autoreleasepool {
        // Shares the device/queue initialised by ensureMetalInit through one light
        // metal_available()-style init (our MSL library is not compiled
        // because MPS needs no kernel_src).
        if (gDevice == nil) {
            gDevice = pickMetalDevice();
            if (gDevice) gQueue = [gDevice newCommandQueue];
        }
        if (!gDevice || !gQueue) { NSLog(@"[metal_matmul_mps] no GPU/queue"); return 0; }
        if (!MPSSupportsMTLDevice(gDevice)) {
            NSLog(@"[metal_matmul_mps] GPU %@ does not support MPS - falling back to the hand-written kernel", gDevice.name);
            return 0;
        }

        size_t bytesA = (size_t)m * (size_t)k * sizeof(float);
        size_t bytesB = (size_t)k * (size_t)n * sizeof(float);
        size_t bytesC = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [gDevice newBufferWithBytes:b length:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) { NSLog(@"[metal_matmul_mps] tao buffer FAILED"); return 0; }

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k
                                                                            rowBytes:k * sizeof(float)
                                                                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n
                                                                            rowBytes:n * sizeof(float)
                                                                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
                                                                            rowBytes:n * sizeof(float)
                                                                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice
                                                                          transposeLeft:NO
                                                                         transposeRight:NO
                                                                             resultRows:m
                                                                          resultColumns:n
                                                                        interiorColumns:k
                                                                                  alpha:1.0
                                                                                   beta:0.0];
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        [mm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_matmul_mps] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c, [bufC contents], bytesC);
        return 1;
    }
}

int metal_matmul2(const char* kernel_src,
                   const float* a1, const float* b1, float* c1, int m1, int k1, int n1,
                   const float* a2, const float* b2, float* c2, int m2, int k2, int n2, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;

        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul2_kernel");
        if (!pipeline) return 0;

        size_t bytesA1 = (size_t)m1 * (size_t)k1 * sizeof(float);
        size_t bytesB1 = (size_t)k1 * (size_t)n1 * sizeof(float);
        size_t bytesC1 = (size_t)m1 * (size_t)n1 * sizeof(float);
        size_t bytesA2 = (size_t)m2 * (size_t)k2 * sizeof(float);
        size_t bytesB2 = (size_t)k2 * (size_t)n2 * sizeof(float);
        size_t bytesC2 = (size_t)m2 * (size_t)n2 * sizeof(float);

        id<MTLBuffer> bufA1 = [gDevice newBufferWithBytes:a1 length:bytesA1 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB1 = [gDevice newBufferWithBytes:b1 length:bytesB1 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC1 = [gDevice newBufferWithLength:bytesC1 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufA2 = [gDevice newBufferWithBytes:a2 length:bytesA2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB2 = [gDevice newBufferWithBytes:b2 length:bytesB2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC2 = [gDevice newBufferWithLength:bytesC2 options:MTLResourceStorageModeShared];
        if (!bufA1 || !bufB1 || !bufC1 || !bufA2 || !bufB2 || !bufC2) {
            NSLog(@"[metal_matmul2] tao buffer FAILED"); return 0;
        }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        [encoder setBuffer:bufA1 offset:0 atIndex:0];
        [encoder setBuffer:bufB1 offset:0 atIndex:1];
        [encoder setBuffer:bufC1 offset:0 atIndex:2];
        int32_t dims1[3] = {m1, k1, n1};
        [encoder setBytes:dims1 length:sizeof(dims1) atIndex:3];

        [encoder setBuffer:bufA2 offset:0 atIndex:4];
        [encoder setBuffer:bufB2 offset:0 atIndex:5];
        [encoder setBuffer:bufC2 offset:0 atIndex:6];
        int32_t dims2[3] = {m2, k2, n2};
        [encoder setBytes:dims2 length:sizeof(dims2) atIndex:7];

        int max_m = (m1 > m2) ? m1 : m2;
        int max_n = (n1 > n2) ? n1 : n2;

        MTLSize gridSize = MTLSizeMake((NSUInteger)max_n, (NSUInteger)max_m, 1);
        NSUInteger tw = MIN((NSUInteger)16, (NSUInteger)max_n); if (tw == 0) tw = 1;
        NSUInteger th = MIN((NSUInteger)16, (NSUInteger)max_m); if (th == 0) th = 1;
        MTLSize groupSize = MTLSizeMake(tw, th, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_matmul2] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c1, [bufC1 contents], bytesC1);
        memcpy(c2, [bufC2 contents], bytesC2);
        return 1;
    }
}

int metal_matmul_q4(const char* kernel_src, const float* a, const unsigned char* wq,
                     const float* scales, const float* zeros, float* c,
                     int m, int k, int n, int groupSize, int nGroupsPerRow, char** error_out) {
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) {
            if (error_out) *error_out = strdup("ensureMetalInit failed");
            return 0;
        }

        id<MTLComputePipelineState> pipeline = getPipeline(MMQ4_KERNEL_NAME_ACTIVE);
        if (!pipeline) {
            if (error_out) *error_out = strdup("getPipeline(MMQ4_KERNEL_NAME_ACTIVE) returned nil");
            return 0;
        }

        size_t bytesPerRow = (size_t)((k + 1) / 2);
        size_t bytesA = (size_t)m * (size_t)k * sizeof(float);
        size_t bytesWq = (size_t)n * bytesPerRow;
        size_t bytesScales = (size_t)n * (size_t)nGroupsPerRow * sizeof(float);
        size_t bytesC = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufWq = [gDevice newBufferWithBytes:wq length:bytesWq options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScales = [gDevice newBufferWithBytes:scales length:bytesScales options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZeros = [gDevice newBufferWithBytes:zeros length:bytesScales options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufWq || !bufScales || !bufZeros || !bufC) {
            if (error_out) *error_out = strdup("[metal_matmul_q4] tao buffer FAILED");
            return 0;
        }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufWq offset:0 atIndex:1];
        [encoder setBuffer:bufScales offset:0 atIndex:2];
        [encoder setBuffer:bufZeros offset:0 atIndex:3];
        [encoder setBuffer:bufC offset:0 atIndex:4];
        [encoder setBytes:&m length:sizeof(int) atIndex:5];
        [encoder setBytes:&k length:sizeof(int) atIndex:6];
        [encoder setBytes:&n length:sizeof(int) atIndex:7];
        [encoder setBytes:&groupSize length:sizeof(int) atIndex:8];
        [encoder setBytes:&nGroupsPerRow length:sizeof(int) atIndex:9];

        NSUInteger simdWidthNS = pipeline.threadExecutionWidth;
        int simdWidthI = (int)simdWidthNS;
        [encoder setBytes:&simdWidthI length:sizeof(int) atIndex:10];
        [encoder setThreadgroupMemoryLength:MMQ4_NDST_ACTIVE * simdWidthNS * sizeof(float) atIndex:0];

        const NSUInteger kMMQ4_NDST = MMQ4_NDST_ACTIVE;
        NSUInteger rowBlocks = ((NSUInteger)n + kMMQ4_NDST - 1) / kMMQ4_NDST;
        MTLSize groupSz = MTLSizeMake(simdWidthNS, 1, 1);
        MTLSize gridCount = MTLSizeMake(rowBlocks, (NSUInteger)m, 1);
        [encoder dispatchThreadgroups:gridCount threadsPerThreadgroup:groupSz];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSString *err = [NSString stringWithFormat:@"[metal_matmul_q4] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error];
            if (error_out) *error_out = strdup([err UTF8String]);
            return 0;
        }

        memcpy(c, [bufC contents], bytesC);
        return 1;
    }
}

// SEVERE BUG FIX: this function used to be missing its signature line entirely
// (only a bare "@autoreleasepool {" outside any C function)
// -> this .m file did not compile with clang, or worse, it was merged by mistake
// into the body of metal_matmul_q4() above depending on the compiler/preprocessor,
// so metal_activation() (used for the sigmoid in the SwiGLU FFN) did not exist
// or ran completely wrong -> very likely the cause of the
// garbage tokens (a wrong sigmoid in SwiGLU makes the whole FFN of every layer wrong).
int metal_activation(const char* kernel_src, int op, const float* x, float* y, int n, float p, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_kernel");
        if (!pipeline) return 0;
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufY offset:0 atIndex:1];
        [encoder setBytes:&op length:sizeof(int) atIndex:2];
        [encoder setBytes:&p length:sizeof(float) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytes);
        return 1;
    }
}

int metal_softmax(const char* kernel_src, const float* x, float* y, int rows, int cols, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"softmax_kernel");
        if (!pipeline) return 0;
        size_t bytes = (size_t)rows * (size_t)cols * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufY offset:0 atIndex:1];
        [encoder setBytes:&cols length:sizeof(int) atIndex:2];
        // Kernel is block-per-row (BB_WG=256 threads reduce one row via
        // threadgroup memory) -> dispatch one threadgroup per row, not one
        // thread per row. Must use dispatchThreadgroups (not dispatchThreads)
        // since the kernel reads threadgroup_position_in_grid for the row.
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytes);
        return 1;
    }
}

int metal_layernorm(const char* kernel_src, const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_kernel");
        if (!pipeline) return 0;
        size_t bytesX = (size_t)rows * (size_t)cols * sizeof(float);
        size_t bytesC = (size_t)cols * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGamma = [gDevice newBufferWithBytes:gamma length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBeta = [gDevice newBufferWithBytes:beta length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytesX options:MTLResourceStorageModeShared];
        if (!bufX || !bufGamma || !bufBeta || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufGamma offset:0 atIndex:1];
        [encoder setBuffer:bufBeta offset:0 atIndex:2];
        [encoder setBuffer:bufY offset:0 atIndex:3];
        [encoder setBytes:&cols length:sizeof(int) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytesX);
        return 1;
    }
}

int metal_embedding_lookup(const char* kernel_src, const float* table, const int* indices, float* y, int vocab, int dim, int num_indices, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"embedding_lookup_kernel");
        if (!pipeline) return 0;
        size_t bytesTable = (size_t)vocab * (size_t)dim * sizeof(float);
        size_t bytesIndices = (size_t)num_indices * sizeof(int);
        size_t bytesY = (size_t)num_indices * (size_t)dim * sizeof(float);
        id<MTLBuffer> bufTable = [gDevice newBufferWithBytes:table length:bytesTable options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufIndices = [gDevice newBufferWithBytes:indices length:bytesIndices options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [gDevice newBufferWithLength:bytesY options:MTLResourceStorageModeShared];
        if (!bufTable || !bufIndices || !bufY) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufTable offset:0 atIndex:0];
        [encoder setBuffer:bufIndices offset:0 atIndex:1];
        [encoder setBuffer:bufY offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_indices);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)num_indices, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [bufY contents], bytesY);
        return 1;
    }
}

int metal_activation_backward(const char* kernel_src, int op, const float* x, const float* dy, float* dx, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_backward_kernel");
        if (!pipeline) return 0;
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDx = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufX || !bufDy || !bufDx) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufDy offset:0 atIndex:1];
        [encoder setBuffer:bufDx offset:0 atIndex:2];
        [encoder setBytes:&op length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(dx, [bufDx contents], bytes);
        return 1;
    }
}

int metal_layernorm_backward(const char* kernel_src, const float* dy, const float* x,
                              const float* gamma, const float* beta,
                              float* dx, float* dgamma, float* dbeta,
                              int rows, int cols, float eps, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_backward_kernel");
        if (!pipeline) return 0;
        size_t bytesX = (size_t)rows * (size_t)cols * sizeof(float);
        size_t bytesC = (size_t)cols * sizeof(float);
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGamma = [gDevice newBufferWithBytes:gamma length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBeta = [gDevice newBufferWithBytes:beta length:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDx = [gDevice newBufferWithLength:bytesX options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDgamma = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDbeta = [gDevice newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufDy || !bufX || !bufGamma || !bufBeta || !bufDx || !bufDgamma || !bufDbeta) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufDy offset:0 atIndex:0];
        [encoder setBuffer:bufX offset:0 atIndex:1];
        [encoder setBuffer:bufGamma offset:0 atIndex:2];
        [encoder setBuffer:bufBeta offset:0 atIndex:3];
        [encoder setBuffer:bufDx offset:0 atIndex:4];
        [encoder setBuffer:bufDgamma offset:0 atIndex:5];
        [encoder setBuffer:bufDbeta offset:0 atIndex:6];
        [encoder setBytes:&rows length:sizeof(int) atIndex:7];
        [encoder setBytes:&cols length:sizeof(int) atIndex:8];
        [encoder setBytes:&eps length:sizeof(float) atIndex:9];
        // The kernel uses one thread per row (dx) AND, when id < cols, the same
        // thread also reduces dgamma/dbeta per column -> the dispatch must cover
        // max(rows, cols).
        int gridN = rows > cols ? rows : cols;
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)gridN);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)gridN, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) return 0;
        memcpy(dx, [bufDx contents], bytesX);
        memcpy(dgamma, [bufDgamma contents], bytesC);
        memcpy(dbeta, [bufDbeta contents], bytesC);
        return 1;
    }
}

int metal_attention_fused(const char* kernel_src, const float* q, const float* k, const float* v,
                           float* o, float* s_matrix, int B, int H, int S, int D, float scale, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (S > 256) { NSLog(@"[metal_attention_fused] S=%d exceeds the kernel limit of 256 (fixed scores[256] array)", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_kernel");
        if (!pipeline) return 0;
        size_t bytesQKV = (size_t)B * (size_t)H * (size_t)S * (size_t)D * sizeof(float);
        size_t bytesS = (size_t)B * (size_t)H * (size_t)S * (size_t)S * sizeof(float);
        id<MTLBuffer> bufQ = [gDevice newBufferWithBytes:q length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDevice newBufferWithBytes:k length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDevice newBufferWithBytes:v length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufO = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [gDevice newBufferWithLength:bytesS options:MTLResourceStorageModeShared];
        if (!bufQ || !bufK || !bufV || !bufO || !bufS) return 0;
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufV offset:0 atIndex:2];
        [encoder setBuffer:bufO offset:0 atIndex:3];
        [encoder setBuffer:bufS offset:0 atIndex:4];
        [encoder setBytes:&B length:sizeof(int) atIndex:5];
        [encoder setBytes:&H length:sizeof(int) atIndex:6];
        [encoder setBytes:&S length:sizeof(int) atIndex:7];
        [encoder setBytes:&D length:sizeof(int) atIndex:8];
        [encoder setBytes:&scale length:sizeof(float) atIndex:9];
        // Grid: x = bh (0..B*H-1), y = ti (0..S-1). Each thread handles one whole
        // causal row - no threadgroup memory / barrier in this kernel
        // (unlike softmax/layernorm), so a threadgroup size of 1 is safe and the
        // simplest, no divisibility of the grid needed.
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_attention_fused] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(o, [bufO contents], bytesQKV);
        memcpy(s_matrix, [bufS contents], bytesS);
        return 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Resident (session) API: many ops encoded into the SAME MTLCommandBuffer, with
// ONE commit+waitUntilCompleted in metal_session_end(). Follows 100% the
// idioms that work in metal_matmul/metal_softmax above (same buffer
// creation, same pipeline lookup, same dispatch) - only sharing
// one encoder/command buffer instead of creating one per command.
static id<MTLCommandBuffer> gSessionCmdBuf = nil;
static id<MTLComputeCommandEncoder> gSessionEncoder = nil;
// NEW: keeps the REAL NSError description of the last failed
// metal_session_end() - see the full explanation in metal_shim.h. A strong (static) ref, so it survives
// the getter call and is only REPLACED (not lost) by the next
// metal_session_end().
static NSString *gLastSessionEndError = nil;

int metal_session_begin(const char* kernel_src, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (gSessionCmdBuf != nil) { NSLog(@"[metal_session_begin] the previous session has not ended, skipping"); return 0; }
        gSessionCmdBuf = [gQueue commandBuffer];
        if (!gSessionCmdBuf) { NSLog(@"[metal_session_begin] commandBuffer returned nil"); return 0; }
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        if (!gSessionEncoder) { NSLog(@"[metal_session_begin] computeCommandEncoder returned nil"); gSessionCmdBuf = nil; return 0; }
        return 1;
    }
}

MetalBufferHandle metal_upload(const float* data, int n) {
    @autoreleasepool {
        // CHANGED: see the explanation in metal_upload_bytes right below - it also lacked
        // ensureMetalInit before, so gDevice was NULL if called before metal_session_begin/
        // the first metal_matmul (exactly the resident-upload case at model load).
        if (!gDevice) { NSLog(@"[metal_upload] gDevice not initialised - call metal_init() first"); return NULL; }
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

MetalBufferHandle metal_upload_borrowed(const float* data, int n) {
    @autoreleasepool {
        if (!gDevice) {
            NSLog(@"[metal_upload_borrowed] gDevice is not initialised");
            return NULL;
        }
        if (!data || n <= 0) return NULL;
        const size_t bytes = (size_t)n * sizeof(float);
        const NSUInteger page = (NSUInteger)getpagesize();
        const uintptr_t address = (uintptr_t)data;
        id<MTLBuffer> buf = nil;
        // Metal's no-copy API requires a page-aligned region. Never round the
        // length up here: that could expose memory beyond the Nim seq.
        if (page > 0 && address % page == 0 && bytes % page == 0 &&
            bytes <= gDevice.maxBufferLength) {
            buf = [gDevice newBufferWithBytesNoCopy:(void *)data
                                             length:bytes
                                            options:MTLResourceStorageModeShared
                                        deallocator:^(void *pointer, NSUInteger length) {
                                            (void)pointer; (void)length;
                                        }];
            if (buf) {
                NSLog(@"[metal_upload_borrowed] zero-copy Shared OK bytes=%zu maxBuffer=%llu",
                      bytes, (unsigned long long)gDevice.maxBufferLength);
                return (__bridge_retained MetalBufferHandle)buf;
            }
        }
        // Correctness fallback for unaligned allocator results, unsupported
        // no-copy mappings, and small/irregular buffers. This can temporarily
        // require a second copy, so large-scratch callers get a diagnostic.
        NSLog(@"[metal_upload_borrowed] zero-copy unavailable (aligned=%d, pageBytes=%d, bytes=%zu, maxBuffer=%llu)",
              (page > 0 && address % page == 0), (page > 0 && bytes % page == 0), bytes,
              (unsigned long long)gDevice.maxBufferLength);
        // Do not silently recreate the original multi-GiB double-allocation
        // failure. For large scratch, fail closed if zero-copy isn't viable.
        if (bytes > (size_t)768 * 1024 * 1024) {
            NSLog(@"[metal_upload_borrowed] refusing copied fallback for large scratch; "
                  "fix alignment/storage or split the VM arena");
            return NULL;
        }
        buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) {
            NSLog(@"[metal_upload_borrowed] copied Shared buffer FAILED bytes=%zu maxBuffer=%llu",
                  bytes, (unsigned long long)gDevice.maxBufferLength);
            return NULL;
        }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

MetalBufferHandle metal_wrap_host_nocopy(const void* data, size_t bytes) {
    @autoreleasepool {
        if (!gDevice || !data || bytes == 0) return NULL;
        // Intel iGPUs report isLowPower; Apple Silicon reports unified memory.
        // A discrete GPU would read these weights over PCIe on every token.
        NSString *name = gDevice.name ?: @"";
        BOOL discreteName = [name containsString:@"AMD"] || [name containsString:@"Radeon"] ||
                            [name containsString:@"NVIDIA"];
        BOOL shared = (gDevice.hasUnifiedMemory || gDevice.isLowPower) && !discreteName &&
                      !gDevice.isRemovable;
        const size_t page = (size_t)getpagesize();
        const size_t length = (bytes + page - 1) / page * page;
        if (!shared || page == 0 || (uintptr_t)data % page != 0 ||
            length > gDevice.maxBufferLength) {
            return NULL;
        }
        id<MTLBuffer> buf = [gDevice newBufferWithBytesNoCopy:(void *)data
                                                       length:length
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        return buf ? (__bridge_retained MetalBufferHandle)buf : NULL;
    }
}

MetalBufferHandle metal_upload_indices(const int* data, int n) {
    @autoreleasepool {
        if (!gDevice) { NSLog(@"[metal_upload_indices] gDevice not initialised - call metal_init() first"); return NULL; }
        size_t bytes = (size_t)n * sizeof(int32_t);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload_indices] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

MetalBufferHandle metal_alloc_scratch(int n) {
    @autoreleasepool {
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_alloc_scratch] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

// NEW: see the full explanation in metal_shim.h. The heap is allocated once
// (device level, with a fixed cost like metal_alloc_scratch, but ONLY once
// instead of thousands of times), then cheap suballocations come from it.
void* metal_alloc_heap(long bytes) {
    @autoreleasepool {
        if (!gDevice) { NSLog(@"[metal_alloc_heap] gDevice not initialised - call metal_init() first"); return NULL; }
        MTLHeapDescriptor *desc = [[MTLHeapDescriptor alloc] init];
        desc.size = (NSUInteger)bytes;
        // CHANGED: the storageMode is chosen from the GPU ARCHITECTURE instead of
        // one hardcoded mode for both kinds of machine.
        // - Apple Silicon (gDevice.hasUnifiedMemory == YES): CPU/GPU share
        //   one physical RAM region -> Shared is VALID for heaps and the BEST
        //   choice (real zero copy, no blit when reading back for verification,
        //   no coherency traffic because there is really no "copy").
        // - Intel/AMD (hasUnifiedMemory == NO, separate VRAM): heaps REJECT
        //   Shared ("Heap Descriptor Validation: Shared storage mode
        //   disallowed", a real SIGABRT) -> Private is REQUIRED.
        // DIAGNOSIS + SAFETY: the crash log points here (SIGABRT right inside
        // newHeapWithDescriptor:, before the "OK" NSLog could run) -> meaning
        // desc.storageMode was Shared at runtime on an Intel iGPU machine, i.e.
        // gDevice.hasUnifiedMemory WRONGLY returned YES ("Heap Descriptor
        // Validation: Shared storage mode disallowed" only happens when
        // Shared heaps are NOT really supported, although the property may report
        // otherwise on some old Intel/AMD drivers). SAFE FIX: do NOT trust only
        // hasUnifiedMemory - cross-check the device name (Intel/AMD/
        // "Iris"/"Radeon" -> force Private whatever hasUnifiedMemory says).
        BOOL unified = gDevice.hasUnifiedMemory;
        NSString *devName = gDevice.name ? gDevice.name : @"";
        BOOL looksNonAppleSilicon = ([devName rangeOfString:@"Intel" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"AMD" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"Radeon" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"Iris" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"NVIDIA" options:NSCaseInsensitiveSearch].location != NSNotFound);
        NSLog(@"[metal_alloc_heap] gDevice.name=%@ hasUnifiedMemory=%d looksNonAppleSilicon=%d", devName, unified, looksNonAppleSilicon);
        if (looksNonAppleSilicon) unified = NO;   // force Private even if the property reports otherwise
        desc.storageMode = unified ? MTLStorageModeShared : MTLStorageModePrivate;
        // CHANGED: type = MTLHeapTypePlacement (the default Automatic works too, but
        // Placement gives clearer control over child buffer lifetimes, so the GPU does not
        // keep resources longer than needed because of automatic heap residency tracking).
        desc.type = MTLHeapTypeAutomatic;
        id<MTLHeap> heap = [gDevice newHeapWithDescriptor:desc];
        if (!heap) { NSLog(@"[metal_alloc_heap] tao heap FAILED (bytes=%ld, unified=%d)", bytes, unified); return NULL; }
        NSLog(@"[metal_alloc_heap] OK: unified=%d -> storageMode=%s", unified, unified ? "Shared" : "Private");
        return (__bridge_retained void*)heap;
    }
}

MetalBufferHandle metal_alloc_from_heap(void* heapPtr, int n) {
    @autoreleasepool {
        if (!heapPtr) { NSLog(@"[metal_alloc_from_heap] heap NULL"); return NULL; }
        id<MTLHeap> heap = (__bridge id<MTLHeap>)heapPtr;
        size_t bytes = (size_t)n * sizeof(float);
        // CHANGED: reads the REAL storageMode of the heap (heap.storageMode) instead of
        // hardcoding Private - a suballocation MUST match the parent heap's mode (Metal
        // does not allow a buffer and its heap to differ), and the heap can now
        // be Shared (Apple Silicon) or Private (Intel/AMD) depending on
        // what metal_alloc_heap decided above.
        MTLResourceOptions opts = (heap.storageMode == MTLStorageModeShared)
            ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> buf = [heap newBufferWithLength:bytes options:opts];
        if (!buf) {
            // Suballocation failed (heap full, e.g. a wrong total size
            // in metal_alloc_heap) - NO silent fallback, return NULL so the
            // caller (Nim ensureKvResident) can handle it (e.g. report a clear error
            // instead of reading/writing a buffer that does not exist).
            NSLog(@"[metal_alloc_from_heap] suballocation FAILED (n=%d) - the heap may be full", n);
            return NULL;
        }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

void metal_free_heap(void* heapPtr) {
    @autoreleasepool {
        if (!heapPtr) return;
        id<MTLHeap> heap = (__bridge_transfer id<MTLHeap>)heapPtr;
        (void)heap;   // released by ARC when leaving this autoreleasepool scope
    }
}

// Register-blocked FP32 GEMM (kernels/gemm_blocked.metal). Picks the tile height from m so a short
// matrix does not pad to 64 rows. Returns 0 when the pipeline is unavailable or the GPU cannot run
// 256 threads per group, and the caller keeps the older tiled kernel.
static int encode_gemm_rb(int transposeB, MetalBufferHandle a, MetalBufferHandle b, int bOffsetBytes,
                          MetalBufferHandle c, int m, int k, int n) {
    if (!gSessionEncoder || m <= 0 || k <= 0 || n <= 0) return 0;
    if (getenv("BYBY_METAL_GEMM_RB") && getenv("BYBY_METAL_GEMM_RB")[0] == '0') return 0;
    // The old NN kernel is still ahead for a handful of rows; the NT one is ahead at every size.
    if (!transposeB && m < 16) return 0;
    int bm = m <= 16 ? 16 : (m <= 32 ? 32 : 64);
    NSString *name = [NSString stringWithFormat:@"gemm_rb_%s_%d", transposeB ? "nt" : "nn", bm];
    id<MTLComputePipelineState> pipeline = getPipeline(name);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) return 0;
    [gSessionEncoder setComputePipelineState:pipeline];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)b offset:(NSUInteger)bOffsetBytes atIndex:1];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
    [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:3];
    [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:4];
    [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:5];
    MTLSize groups = MTLSizeMake(((NSUInteger)n + 63) / 64, ((NSUInteger)m + bm - 1) / bm, 1);
    [gSessionEncoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return 1;
}

// ---- FP4-KM GEMM for prefill (kernels/gemm_fp4km.metal, gemm_fp4km_sg.metal) -------------------
// The simdgroup_matrix kernel lives in its own library that is compiled only on an Apple GPU, so a
// source an Intel/AMD driver cannot build never touches the main library.
static id<MTLLibrary> gSgLibrary = nil;
static int gSgState = 0;   // 0 untried, 1 ready, -1 unavailable
static id<MTLComputePipelineState> gSgPipeline = nil;

static id<MTLComputePipelineState> fp4kmSgPipeline(const char* source) {
    if (gSgState < 0) return nil;
    if (gSgState > 0) return gSgPipeline;
    gSgState = -1;
    if (!source || !gDevice) return nil;
    BOOL apple = NO;
    if (@available(macOS 10.15, *)) apple = [gDevice supportsFamily:MTLGPUFamilyApple7];
    // BYBY_FP4_SG_COMPILE_CHECK=1 compiles the source on any GPU to catch syntax errors, but a
    // non-Apple GPU never gets a pipeline from it.
    BOOL compileOnly = !apple && getenv("BYBY_FP4_SG_COMPILE_CHECK") != NULL;
    if (!apple && !compileOnly) return nil;
    NSError *error = nil;
    gSgLibrary = [gDevice newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&error];
    if (!gSgLibrary) { NSLog(@"[fp4km-sg] library compile failed: %@", error); return nil; }
    if (compileOnly) { NSLog(@"[fp4km-sg] library compiled (compile check only, no pipeline on this GPU)"); return nil; }
    id<MTLFunction> fn = [gSgLibrary newFunctionWithName:@"gemm_fp4km_sg"];
    if (!fn) return nil;
    gSgPipeline = [gDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!gSgPipeline || gSgPipeline.maxTotalThreadsPerThreadgroup < 128) { gSgPipeline = nil; return nil; }
    gSgState = 1;
    return gSgPipeline;
}

// mode: 0 = register-blocked (any GPU), 1 = simdgroup_matrix (Apple GPU only). Returns 0 when the
// requested kernel is not available so the caller can fall back to the packed micro-VM tiles.
int metal_fp4km_gemm_enc(MetalBufferHandle a, MetalBufferHandle w, MetalBufferHandle lut,
                         MetalBufferHandle c, int m, int k, int n, int mode, const char* sgSource) {
    @autoreleasepool {
        if (!gSessionEncoder || m <= 0 || k <= 0 || n <= 0 || k % 32 != 0) return 0;
        id<MTLComputePipelineState> pipeline = nil;
        NSUInteger bm = 64, threads = 256;
        if (mode == 1) {
            pipeline = fp4kmSgPipeline(sgSource);
            threads = 128;
        } else {
            bm = m <= 16 ? 16 : (m <= 32 ? 32 : 64);
            pipeline = getPipeline([NSString stringWithFormat:@"gemm_fp4km_rb_%d", (int)bm]);
            if (pipeline && pipeline.maxTotalThreadsPerThreadgroup < 256) pipeline = nil;
        }
        if (!pipeline) return 0;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:1];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)lut offset:0 atIndex:2];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:3];
        [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:6];
        MTLSize groups = MTLSizeMake(((NSUInteger)n + 63) / 64, ((NSUInteger)m + bm - 1) / bm, 1);
        [gSessionEncoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_enc] metal_session_begin not called"); return 0; }
        if (encode_gemm_rb(0, a, b, 0, c, m, k, n)) return 1;
        int tile = 8;
        id<MTLComputePipelineState> pipeline = pickMatmulPipeline(&tile);
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufB offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:2];
        [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:5];
        // CHANGED: dynamic tile (8/16/32) from the real GPU, see pickMatmulPipeline().
        NSUInteger t = (NSUInteger)tile;
        MTLSize groupSize = MTLSizeMake(t, t, 1);
        MTLSize threadgroupCount = MTLSizeMake(((NSUInteger)n + t - 1) / t, ((NSUInteger)m + t - 1) / t, 1);
        [gSessionEncoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:groupSize];
        // REAL BUG FIX: one encoder shared by several dispatches (matmul then
        // softmax then matmul...) does NOT automatically make the writes of one
        // dispatch visible to the reads of the NEXT one on GPUs other than Apple
        // Silicon (e.g. Intel Iris) - Metal only guarantees execution ORDER, not
        // memory visibility between different pipelines without an
        // explicit barrier. This was exactly the "wrong results on the
        // Intel Iris driver" the old comment suspected but did not fix - a
        // memoryBarrierWithScope: is inserted right after EVERY dispatch writing a buffer
        // the next dispatch reads, forcing the GPU to finish + flush caches before
        // the next step (matmul->softmax->matmul, or mul->mul->add...) reads it.
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_init(const char* kernel_src) {
    @autoreleasepool {
        return ensureMetalInit(kernel_src) ? 1 : 0;
    }
}

int metal_mps_supported(void) { return gDevice && MPSSupportsMTLDevice(gDevice); }

int metal_is_apple_silicon(void) {
    int arm64 = 0;
    size_t size = sizeof(arm64);
    return sysctlbyname("hw.optional.arm64", &arm64, &size, NULL, 0) == 0 && arm64 == 1;
}

int metal_matmul_mps_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c,
                            int m, int k, int n, int cRowStride, int cColumnOffset) {
    // Reject decode before any MPS descriptor/encoder is constructed.
    if (m <= 1) return 0;
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf || !MPSSupportsMTLDevice(gDevice) ||
            !a || !b || !c || m <= 0 || k <= 0 || n <= 0 ||
            cColumnOffset < 0 || cRowStride < cColumnOffset + n) return 0;
        id<MTLBuffer> ba = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bb = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bc = (__bridge id<MTLBuffer>)c;
        if (ba.length < (NSUInteger)m * k * sizeof(float) ||
            bb.length < (NSUInteger)n * k * sizeof(float) ||
            bc.length < ((NSUInteger)(m - 1) * cRowStride + cColumnOffset + n) * sizeof(float)) return 0;
        MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k
            rowBytes:(NSUInteger)k * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *db = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k
            rowBytes:(NSUInteger)k * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
            rowBytes:(NSUInteger)cRowStride * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:ba descriptor:da];
        MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:bb descriptor:db];
        MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:bc
            offset:(NSUInteger)cColumnOffset * sizeof(float) descriptor:dc];
        static NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *cache;
        if (!cache) cache = [NSMutableDictionary new];
        NSString *key = [NSString stringWithFormat:@"%d/%d/%d", m, k, n];
        MPSMatrixMultiplication *mm = cache[key];
        if (!mm) {
            mm = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice
                transposeLeft:NO transposeRight:YES resultRows:m resultColumns:n
                interiorColumns:k alpha:1.0 beta:0.0];
            cache[key] = mm;
        }
        // End the compute encoder, not the session. MPS and later VM dispatches
        // remain ordered in the same command buffer, without a CPU wait.
        [gSessionEncoder endEncoding];
        [mm encodeToCommandBuffer:gSessionCmdBuf leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        return gSessionEncoder != nil;
    }
}

// MPS GEMM with offsets and an optional transposed B. A is [m,k] at aOffsetElems; B is [k,n]
// (transposeRight = 0) or [n,k] (transposeRight = 1); C is [m,n] at cOffsetElems with row stride
// cRowStride. Returns 0 (nothing encoded) when MPS cannot take the shapes.
static int mps_gemm_impl(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c,
                         int m, int k, int n, int cRowStride, int aOffsetElems, int cOffsetElems,
                         int transposeRight) {
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf || !MPSSupportsMTLDevice(gDevice) ||
            !a || !b || !c || m <= 0 || k <= 0 || n <= 0 || aOffsetElems < 0 || cOffsetElems < 0 ||
            cRowStride < n) return 0;
        id<MTLBuffer> ba = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bb = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bc = (__bridge id<MTLBuffer>)c;
        if (ba.length < ((NSUInteger)aOffsetElems + (NSUInteger)m * k) * sizeof(float) ||
            bb.length < (NSUInteger)n * k * sizeof(float) ||
            bc.length < ((NSUInteger)cOffsetElems + (NSUInteger)(m - 1) * cRowStride + n) * sizeof(float)) return 0;
        MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k
            rowBytes:(NSUInteger)k * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *db = transposeRight
            ? [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:(NSUInteger)k * sizeof(float) dataType:MPSDataTypeFloat32]
            : [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:(NSUInteger)n * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n
            rowBytes:(NSUInteger)cRowStride * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix *ma = [[MPSMatrix alloc] initWithBuffer:ba offset:(NSUInteger)aOffsetElems * sizeof(float) descriptor:da];
        MPSMatrix *mb = [[MPSMatrix alloc] initWithBuffer:bb descriptor:db];
        MPSMatrix *mc = [[MPSMatrix alloc] initWithBuffer:bc offset:(NSUInteger)cOffsetElems * sizeof(float) descriptor:dc];
        static NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *cache;
        if (!cache) cache = [NSMutableDictionary new];
        NSString *key = [NSString stringWithFormat:@"%d/%d/%d/%d", transposeRight, m, k, n];
        MPSMatrixMultiplication *mm = cache[key];
        if (!mm) {
            mm = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice
                transposeLeft:NO transposeRight:transposeRight ? YES : NO resultRows:m resultColumns:n
                interiorColumns:k alpha:1.0 beta:0.0];
            cache[key] = mm;
        }
        [gSessionEncoder endEncoding];
        [mm encodeToCommandBuffer:gSessionCmdBuf leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        return gSessionEncoder != nil;
    }
}

// NN variant of metal_matmul_mps_nt_enc: B keeps its [k,n] layout (FIR fir.linear weights); m >= 2.
int metal_matmul_mps_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c,
                         int m, int k, int n, int cRowStride, int cColumnOffset) {
    if (m <= 1 || cColumnOffset < 0) return 0;
    return mps_gemm_impl(a, b, c, m, k, n, cRowStride, 0, cColumnOffset, 0);
}

// Any m (including 1), offsets into A and C, B transposed or not. The caller decides when a single
// row should go to MPS (the framework's decode keeps its native GEMV on Apple Silicon).
int metal_matmul_mps_any_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c,
                             int m, int k, int n, int aOffsetElems, int cOffsetElems, int transposeRight) {
    return mps_gemm_impl(a, b, c, m, k, n, n, aOffsetElems, cOffsetElems, transposeRight);
}

int metal_matmul_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n) {
    return metal_matmul_nt_off_enc(a, b, 0, c, m, k, n);
}

int metal_matmul_nt_off_enc(MetalBufferHandle a, MetalBufferHandle b, int bOffsetBytes,
                            MetalBufferHandle c, int m, int k, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_nt_enc] metal_session_begin not called"); return 0; }
        if (encode_gemm_rb(1, a, b, bOffsetBytes, c, m, k, n)) return 1;
        // Fixed tile=8 (matmul_nt_kernel), the same stable choice
        // as pickMatmulPipeline() for the normal version - no need to measure
        // maxTotalThreadsPerThreadgroup again for this kernel.
        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul_nt_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufB offset:(NSUInteger)bOffsetBytes atIndex:1];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:2];
        [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:5];
        NSUInteger t = 8;
        MTLSize groupSize = MTLSizeMake(t, t, 1);
        MTLSize threadgroupCount = MTLSizeMake(((NSUInteger)n + t - 1) / t, ((NSUInteger)m + t - 1) / t, 1);
        [gSessionEncoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:groupSize];
        // See the memoryBarrierWithScope explanation in metal_matmul_enc - required
        // on drivers other than Apple Silicon (e.g. Intel Iris).
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

MetalBufferHandle metal_upload_bytes(const unsigned char* data, int n) {
    @autoreleasepool {
        // No kernel_src argument (the signature matches the header) - the CALLER
        // (Nim: metalUploadBytes) MUST call metal_init(kernel_src) once first
        // (dispatch_once inside, so repeated calls are cheap) so gDevice is ready.
        if (!gDevice) { NSLog(@"[metal_upload_bytes] gDevice not initialised - call metal_init() first"); return NULL; }
        size_t bytes = (size_t)n;
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload_bytes] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

// ============================================================
// PRIVATE UPLOAD (GPU-only) - for RESIDENT WEIGHTS (Q4: wq/scales/
// zeros, OR resident fp32 for int8/int4-symmetric/tied embedding through
// beResidentUploadFp32) - buffers uploaded EXACTLY ONCE when the model loads,
// after which the CPU NEVER reads/writes them again (only the GPU reads them every token). Uses
// .private so the driver need not keep CPU cache coherency on every GPU read
// (which Shared requires). Activation buffers (uploaded from the CPU every token,
// or read back with sessionRead) STILL use Shared as before (metal_upload/
// metal_upload_bytes) - a real "hybrid": touched only by the GPU
// -> Private, touched by the CPU (upload/read) -> Shared.
// A Private buffer cannot be created with newBufferWithBytes directly - it goes through a
// Shared staging buffer and a blit to Private, one extra copy ONLY at model
// load (not every token).
//
// CHANGED (fix for OOM/Killed:9 - see the 2026-09-14 debug report; NOT built/run
// on a real machine, the environment of this fix had NO macOS/Metal toolchain -
// rebuild + watch the RSS before fully trusting it, revert to Private
// if anything looks wrong): low-power/integrated GPUs (e.g. Intel Iris Plus
// 640, isLowPower=YES - the flag logged in pickMetalDevice, "lowPower=1")
// have NO dedicated VRAM - Private on them still lives in THE SAME physical RAM
// as the CPU, only WIRED by the kernel - it CANNOT be compressed/swapped
// however much system swap is free. Summed over the 129
// quant-resident parameters (64 blocks x2 + lm_head) this wired memory was the
// main reason Jetsam killed the process. isLowPower is used (NOT hasUnifiedMemory -
// that property is known to be unreliable for MTLHeap, see the
// metal_alloc_heap comment, but that is a separate Shared-heap validation bug, not
// applicable to the plain MTLBuffers used here) so ONLY integrated GPUs change behaviour;
// discrete/high-performance GPUs keep Private as originally designed.
static MetalBufferHandle metalUploadPrivateRaw(const void* data, size_t bytes) {
    if (!gDevice) { NSLog(@"[metalUploadPrivateRaw] gDevice is not initialised"); return NULL; }
    id<MTLBuffer> staging = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    if (!staging) { NSLog(@"[metalUploadPrivateRaw] tao staging buffer FAILED (bytes=%zu)", bytes); return NULL; }
    if (gDevice.isLowPower) {
        // UMA/integrated: the staging buffer IS ALREADY usable (same RAM region,
        // Shared, pageable) - no extra Private copy + blit (it would only waste
        // wired memory on this kind of GPU). Return the staging buffer directly.
        return (__bridge_retained MetalBufferHandle)staging;
    }
    id<MTLBuffer> priv = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!priv) { NSLog(@"[metalUploadPrivateRaw] tao private buffer FAILED (bytes=%zu)", bytes); return NULL; }
    id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:priv destinationOffset:0 size:bytes];
    [blit endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
    if (cmdBuf.status == MTLCommandBufferStatusError) {
        NSLog(@"[metalUploadPrivateRaw] blit FAILED, error=%@ - falling back to Shared to keep the data", cmdBuf.error);
        return (__bridge_retained MetalBufferHandle)staging;   // safe: correct data, only the speedup is lost
    }
    return (__bridge_retained MetalBufferHandle)priv;
}

MetalBufferHandle metal_upload_private(const float* data, int n) {
    @autoreleasepool { return metalUploadPrivateRaw((const void*)data, (size_t)n * sizeof(float)); }
}

MetalBufferHandle metal_upload_bytes_private(const unsigned char* data, int n) {
    @autoreleasepool { return metalUploadPrivateRaw((const void*)data, (size_t)n); }
}

#include "q4_dynamic.inc"

int metal_matmul_q4_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                         MetalBufferHandle zeros, MetalBufferHandle c,
                         int m, int k, int n, int groupSize, int nGroupsPerRow, int wqOffset) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_q4_enc] metal_session_begin not called"); return 0; }
        if (q4EncodeSelected(a, wq, scales, zeros, c, m, k, n, groupSize, nGroupsPerRow, wqOffset)) return 1;
        id<MTLComputePipelineState> pipeline = getPipeline(MMQ4_KERNEL_NAME_ACTIVE);
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufWq = (__bridge id<MTLBuffer>)wq;
        id<MTLBuffer> bufScales = (__bridge id<MTLBuffer>)scales;
        id<MTLBuffer> bufZeros = (__bridge id<MTLBuffer>)zeros;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        // uchar reads: any byte offset is a valid binding offset.
        [gSessionEncoder setBuffer:bufWq offset:(NSUInteger)wqOffset atIndex:1];
        [gSessionEncoder setBuffer:bufScales offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufZeros offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:4];
        [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&groupSize length:sizeof(int) atIndex:8];
        [gSessionEncoder setBytes:&nGroupsPerRow length:sizeof(int) atIndex:9];
        // SEVERE BUG FIX (the real cause of the "garbage tokens" - see the full
        // explanation in the non-session metal_matmul_q4() and in
        // vecop_matmul.metal): NO more hardcoded 32 - reads the REAL
        // pipeline.threadExecutionWidth and dispatches threadgroups of EXACTLY
        // that size -> a threadgroup ALWAYS = EXACTLY one real hardware simdgroup.
        NSUInteger simdWidthNS = pipeline.threadExecutionWidth;
        int simdWidthI = (int)simdWidthNS;
        [gSessionEncoder setBytes:&simdWidthI length:sizeof(int) atIndex:10];
        [gSessionEncoder setThreadgroupMemoryLength:MMQ4_NDST_ACTIVE * simdWidthNS * sizeof(float) atIndex:0];

        const NSUInteger kMMQ4_NDST = MMQ4_NDST_ACTIVE;  // kept in sync with MMQ4_KERNEL_NAME_ACTIVE at the top of the file
        NSUInteger rowBlocks = ((NSUInteger)n + kMMQ4_NDST - 1) / kMMQ4_NDST;
        MTLSize groupSz = MTLSizeMake(simdWidthNS, 1, 1);
        MTLSize gridCount = MTLSizeMake(rowBlocks, (NSUInteger)m, 1);
        [gSessionEncoder dispatchThreadgroups:gridCount threadsPerThreadgroup:groupSz];
        // CHANGED: see the memoryBarrierWithScope explanation in metal_matmul_enc - needed
        // on GPUs other than Apple Silicon (Intel Iris) so the next dispatch
        // (e.g. sessionActivation reading bufC) sees the right data.
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// NEW: Q4 matmul with a real batch M (weights read once, reused for MBLOCK=4
// rows) - differs from metal_matmul_q4_enc ONLY in the kernel name + the grid Y computation (divided
// by MBLOCK instead of = M). See matmul_q4_batched.metal.
// Tiled prefill GEMM (matmul_q4_gemm.metal): a 64 x 128 output tile per
// threadgroup by default, the weights decoded once per 64 rows into
// threadgroup memory. BYBY_METAL_Q4_GEMM_TILE=0..2 picks a smaller tile.
// Used for M >= BYBY_METAL_Q4_GEMM_MIN_M (default 8); BYBY_METAL_Q4_GEMM=0
// keeps the batched/autotuned kernels.
typedef struct { uint32_t M, K, N, groupSize, groupsPerRow, wqShift; } MetalQ4GemmParams;

static int metal_matmul_q4_gemm_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                                    MetalBufferHandle zeros, MetalBufferHandle c,
                                    int m, int k, int n, int groupSize, int nGroupsPerRow, int wqOffset) {
    static int enabled = -1, minM = 8;
    if (enabled < 0) {
        const char *v = getenv("BYBY_METAL_Q4_GEMM");
        enabled = !(v && strcmp(v, "0") == 0);
        const char *mm = getenv("BYBY_METAL_Q4_GEMM_MIN_M");
        if (mm && atoi(mm) > 1) minM = atoi(mm);
    }
    if (!enabled || m < minM || k % 32 != 0 || (groupSize != 0 && groupSize % 32 != 0) ||
        nGroupsPerRow <= 0 || wqOffset < 0) return 0;
    static int variant = -1;
    if (variant < 0) {
        const char *v = getenv("BYBY_METAL_Q4_GEMM_TILE");
        // 3 = 64 x 128 with an 8 x 8 register tile, the fastest on Intel Iris
        // at every M from 32 to 256 (the others: 0 = 32x64, 1 = 64x64, 2 = 32x128).
        variant = v ? atoi(v) : 3;
        if (variant < 0 || variant > 3) variant = 3;
    }
    static NSString *names[4] = { @"matmul_q4_gemm_kernel", @"matmul_q4_gemm_64x64_kernel",
                                  @"matmul_q4_gemm_32x128_kernel", @"matmul_q4_gemm_64x128_kernel" };
    static const int tileM[4] = { 32, 64, 32, 64 };
    static const int tileN[4] = { 64, 64, 128, 128 };
    id<MTLComputePipelineState> pipeline = getPipeline(names[variant]);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) return 0;
    MetalQ4GemmParams params = { (uint32_t)m, (uint32_t)k, (uint32_t)n, (uint32_t)groupSize,
                                 (uint32_t)nGroupsPerRow, (uint32_t)(wqOffset & 3) };
    [gSessionEncoder setComputePipelineState:pipeline];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    // 32-bit weight loads: bind 4-byte aligned, the kernel shifts the rest.
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)wq offset:(NSUInteger)(wqOffset & ~3) atIndex:1];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)scales offset:0 atIndex:2];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)zeros offset:0 atIndex:3];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:4];
    [gSessionEncoder setBytes:&params length:sizeof(params) atIndex:5];
    MTLSize grid = MTLSizeMake((NSUInteger)((n + tileN[variant] - 1) / tileN[variant]),
                               (NSUInteger)((m + tileM[variant] - 1) / tileM[variant]), 1);
    [gSessionEncoder dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return 1;
}

int metal_matmul_q4_batched_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                                 MetalBufferHandle zeros, MetalBufferHandle c,
                                 int m, int k, int n, int groupSize, int nGroupsPerRow, int wqOffset) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_q4_batched_enc] metal_session_begin not called"); return 0; }
        if (metal_matmul_q4_gemm_enc(a, wq, scales, zeros, c, m, k, n, groupSize, nGroupsPerRow, wqOffset)) return 1;
        if (q4EncodeSelected(a, wq, scales, zeros, c, m, k, n, groupSize, nGroupsPerRow, wqOffset)) return 1;
        id<MTLComputePipelineState> pipeline = getPipeline(@"matmul_q4_batched_m4_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufWq = (__bridge id<MTLBuffer>)wq;
        id<MTLBuffer> bufScales = (__bridge id<MTLBuffer>)scales;
        id<MTLBuffer> bufZeros = (__bridge id<MTLBuffer>)zeros;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        // uchar reads: any byte offset is a valid binding offset.
        [gSessionEncoder setBuffer:bufWq offset:(NSUInteger)wqOffset atIndex:1];
        [gSessionEncoder setBuffer:bufScales offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufZeros offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:4];
        [gSessionEncoder setBytes:&m length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&k length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&n length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&groupSize length:sizeof(int) atIndex:8];
        [gSessionEncoder setBytes:&nGroupsPerRow length:sizeof(int) atIndex:9];
        NSUInteger simdWidthNS = pipeline.threadExecutionWidth;
        int simdWidthI = (int)simdWidthNS;
        [gSessionEncoder setBytes:&simdWidthI length:sizeof(int) atIndex:10];
        [gSessionEncoder setThreadgroupMemoryLength:8 * simdWidthNS * sizeof(float) atIndex:0];

        const NSUInteger kBATCHED_NDST = 2;    // MUST match NDST_VAL in BB_MATMUL_Q4_BATCHED(2,4,...)
        const NSUInteger kMBLOCK = 4;          // MUST match MBLOCK_VAL in BB_MATMUL_Q4_BATCHED(2,4,...)
        NSUInteger rowBlocksB = ((NSUInteger)n + kBATCHED_NDST - 1) / kBATCHED_NDST;
        NSUInteger mBlocks = ((NSUInteger)m + kMBLOCK - 1) / kMBLOCK;   // UNLIKE the original: divided by MBLOCK
        MTLSize groupSzB = MTLSizeMake(simdWidthNS, 1, 1);
        MTLSize gridCountB = MTLSizeMake(rowBlocksB, mBlocks, 1);
        [gSessionEncoder dispatchThreadgroups:gridCountB threadsPerThreadgroup:groupSzB];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_softmax_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"softmax_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:1];
        [gSessionEncoder setBytes:&cols length:sizeof(int) atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n) {
    return metal_activation_enc_p(op, x, y, n, 0.0f);
}

// NEW (BYBY_MERGE_QKV_ATTN_SESSION): in-place scalar multiply inside a session,
// using the existing scale_kernel of vecop_matmul.metal (before it was only called
// through the non-session metal_scale_c, with no _enc variant). Used to apply
// attn.scale to Q DIRECTLY ON THE GPU instead of reading Q back, scaling and uploading again -
// required to merge Session #1 (QKV+RoPE) and Session #2
// (attention+oProj+FFN) into one command buffer for the whole layer.
// Safe: mirrors the pattern of metal_activation_enc_p exactly (pipeline lookup,
// dispatchThreads, memoryBarrier) - the .metal kernel is unchanged, only glue is added.
int metal_scale_enc(MetalBufferHandle x, float scalar, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_scale_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"scale_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBytes:&scalar length:sizeof(float) atIndex:1];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// NEW: a variant with parameter p (for op=7 "pow" - opcodes match the code table of
// the immediate metalActivation in metal_backend.nim: 0=relu,1=sigmoid,2=tanh,
// 3=apflu,4=exp,5=log,6=sqrt,7=pow). The other ops (0-6) do not read buffer(3)
// in activation_kernel, so passing p=0.0 is harmless; the old metal_activation_enc
// keeps its signature so the existing relu/sigmoid/tanh/apflu call sites do NOT break.
int metal_activation_enc_p(int op, MetalBufferHandle x, MetalBufferHandle y, int n, float p) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_activation_enc_p] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:1];
        [gSessionEncoder setBytes:&op length:sizeof(int) atIndex:2];
        [gSessionEncoder setBytes:&p length:sizeof(float) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

// NEW: activation_backward_kernel (for apflu_backward in a
// session - relu/sigmoid/tanh backward do not go through a session yet, so only
// apflu is wired, but this function covers all 4 opcodes if needed later).
int metal_activation_backward_enc(int op, MetalBufferHandle x, MetalBufferHandle dy, MetalBufferHandle dx, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_activation_backward_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"activation_backward_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufDy = (__bridge id<MTLBuffer>)dy;
        id<MTLBuffer> bufDx = (__bridge id<MTLBuffer>)dx;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufDy offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufDx offset:0 atIndex:2];
        [gSessionEncoder setBytes:&op length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_rope_enc(MetalBufferHandle x, MetalBufferHandle cosT, MetalBufferHandle sinT,
                    int T, int Hn, int D) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_rope_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"rope_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufCos = (__bridge id<MTLBuffer>)cosT;
        id<MTLBuffer> bufSin = (__bridge id<MTLBuffer>)sinT;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufCos offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufSin offset:0 atIndex:2];
        [gSessionEncoder setBytes:&Hn length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&D length:sizeof(int) atIndex:4];
        int half_ = D / 2;
        MTLSize gridSize = MTLSizeMake((NSUInteger)half_, (NSUInteger)Hn, (NSUInteger)T);
        NSUInteger tgx = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)half_);
        if (tgx == 0) tgx = 1;
        MTLSize groupSize = MTLSizeMake(tgx, 1, 1);
        [gSessionEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_vecop_enc(int op, MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_vecop_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"vecop_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufB offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufC offset:0 atIndex:2];
        [gSessionEncoder setBytes:&op length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

// NEW: dst[dstOffset..dstOffset+n) = src[0..n). dstOffset is in float32
// ELEMENTS - multiplied by sizeof(float) for setBuffer:offset: (the real
// Metal byte offset). Used to join several child buffers on the GPU
// into one contiguous buffer in the SAME session (see the explanation in
// copy_into_kernel in vecop_matmul.metal) - removes the last blocker
// that kept the attention-decode session and the oProj session apart.
// NOTE: the offset used for setBuffer MUST be a multiple of
// device.minimumLinearTextureAlignmentForPixelFormat or, more simply -
// according to the Metal documentation, the offset of a PLAIN buffer (not a texture) only
// needs to be a multiple of 4 bytes on most GPUs (Apple/AMD/Intel) because this is a
// buffer of floats (naturally 4-byte aligned) - no special alignment
// needed, unlike textures. If a particular driver demands a larger alignment
// (rare, not seen in this version), the function returns 0 through the adjustedOffset check.
int metal_copy_into_enc(MetalBufferHandle src, MetalBufferHandle dst, int srcOffsetElems, int dstOffsetElems, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_copy_into_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"copy_into_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufSrc = (__bridge id<MTLBuffer>)src;
        id<MTLBuffer> bufDst = (__bridge id<MTLBuffer>)dst;
        NSUInteger srcByteOffset = (NSUInteger)srcOffsetElems * sizeof(float);
        NSUInteger dstByteOffset = (NSUInteger)dstOffsetElems * sizeof(float);
        // NEW: srcOffsetElems - uses EXACTLY the setBuffer:offset: mechanism that
        // dstOffsetElems already used (known to work, no new
        // math) - only applied to the SOURCE buffer instead of always 0.
        // Purpose: merge many small uploads into one large upload and extract each
        // slice with copy_into, instead of uploading each small slice separately.
        if (srcByteOffset % 4 != 0) { NSLog(@"[metal_copy_into_enc] src offset is not a multiple of 4 bytes"); return 0; }
        if (dstByteOffset % 4 != 0) { NSLog(@"[metal_copy_into_enc] dst offset is not a multiple of 4 bytes"); return 0; }
        if (dstByteOffset >= bufDst.length) { NSLog(@"[metal_copy_into_enc] dst offset exceeds the destination buffer size"); return 0; }
        if (srcByteOffset >= bufSrc.length) { NSLog(@"[metal_copy_into_enc] src offset exceeds the source buffer size"); return 0; }
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufSrc offset:srcByteOffset atIndex:0];
        [gSessionEncoder setBuffer:bufDst offset:dstByteOffset atIndex:1];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

int metal_rmsnorm_enc(MetalBufferHandle x, MetalBufferHandle gamma,
                       MetalBufferHandle y, int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_rmsnorm_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"rmsnorm_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufGamma = (__bridge id<MTLBuffer>)gamma;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufGamma offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:2];
        [gSessionEncoder setBytes:&cols length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&eps length:sizeof(float) atIndex:4];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_layernorm_enc(MetalBufferHandle x, MetalBufferHandle gamma, MetalBufferHandle beta,
                         MetalBufferHandle y, int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_layernorm_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufGamma = (__bridge id<MTLBuffer>)gamma;
        id<MTLBuffer> bufBeta = (__bridge id<MTLBuffer>)beta;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufGamma offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufBeta offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:3];
        [gSessionEncoder setBytes:&cols length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&eps length:sizeof(float) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_embedding_lookup_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"embedding_lookup_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufTable = (__bridge id<MTLBuffer>)table;
        id<MTLBuffer> bufIdx = (__bridge id<MTLBuffer>)indices;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufTable offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufIdx offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufY offset:0 atIndex:2];
        [gSessionEncoder setBytes:&dim length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)dim);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreadgroups:MTLSizeMake((NSUInteger)num_indices, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

// Session variant of metal_attention_fused above: creates NO buffer/command
// buffer of its own, uses the resident buffers directly (q/k/v uploaded, o/s_matrix
// alloc_scratch'ed) and the open gSessionEncoder - encodes WITHOUT committing,
// so metal_session_end() commits+waits once together with every other op of the layer
// (add/layernorm/matmul/apflu). Same S<=256 limit as the non-session version because
// it shares the attention_fused_kernel.
int metal_attention_fused_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                               MetalBufferHandle o, MetalBufferHandle s_matrix,
                               int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_fused_enc] metal_session_begin not called"); return 0; }
        if (S > 256) { NSLog(@"[metal_attention_fused_enc] S=%d exceeds the kernel limit of 256", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)q;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)k;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)v;
        id<MTLBuffer> bufO = (__bridge id<MTLBuffer>)o;
        id<MTLBuffer> bufS = (__bridge id<MTLBuffer>)s_matrix;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufQ offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufK offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufV offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufO offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufS offset:0 atIndex:4];
        [gSessionEncoder setBytes:&B length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&H length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&S length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&D length:sizeof(int) atIndex:8];
        [gSessionEncoder setBytes:&scale length:sizeof(float) atIndex:9];
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(1, 1, 1);
        [gSessionEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // CHANGED: see the explanation in metal_matmul_enc
        return 1;
    }
}

// NEW: session variant of attention_fused_flash_kernel (see the algorithm
// in the .metal). No s_matrix buffer (index 4) any more -> B
// moves to index 4, H->5, S->6, D->7, scale->8 (one less than the old
// metal_attention_fused_enc above). For INFERENCE (prefill) -
// NOT for training (backward needs s_matrix and still calls the old
// metal_attention_fused_enc above on the training path).
int metal_attention_fused_flash_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                     MetalBufferHandle o,
                                     int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_fused_flash_enc] metal_session_begin not called"); return 0; }
        if (D > 128) { NSLog(@"[metal_attention_fused_flash_enc] D=%d exceeds the flash kernel limit of 128", D); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_flash_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)q;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)k;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)v;
        id<MTLBuffer> bufO = (__bridge id<MTLBuffer>)o;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufQ offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufK offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufV offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufO offset:0 atIndex:3];
        [gSessionEncoder setBytes:&B length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&H length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&S length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&D length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&scale length:sizeof(float) atIndex:8];
        // CHANGED: the kernel needs a REAL THREADGROUP (64 threads cooperating per query row,
        // see the explanation in the .metal) for the reduction (redBuf) and the shared
        // scalar variables (sM/sL/sAOld/sP) - dispatchThreadgroups must be used,
        // NO more dispatchThreads with groupSize=(1,1,1). NOTE: acc[] (the accumulated
        // output) MOVED BACK to a private per-thread array (registers)
        // in the .metal - NO longer threadgroup memory, but this does
        // NOT affect the shim: acc[] is declared inside the kernel body
        // (not a threadgroup pointer parameter passed by the host through
        // setThreadgroupMemoryLength), so the dispatch/buffer binding here STAYS
        // the same, nothing else to change.
        const NSUInteger kAttnFlashTg = 64;   // MUST match #define ATTN_FLASH_TG in vecop_matmul.metal
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(kAttnFlashTg, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// NEW (v2 - parallelism fix): session variant of attention_decode_gqa_kernel.
// The first version dispatched 1 thread/head (only 4-8 threads in total) - wasting almost the whole
// GPU, measured ~4x slower than the old path. Now it dispatches EXACTLY 1
// THREADGROUP (128 threads, matching ATTN_DECODE_TG in the .metal) / query head -
// the threadgroup splits the S and D loops inside the kernel with a tree reduction.
int metal_attention_decode_gqa_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                    MetalBufferHandle o, MetalBufferHandle scores,
                                    int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_decode_gqa_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_decode_gqa_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)q;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)k;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)v;
        id<MTLBuffer> bufO = (__bridge id<MTLBuffer>)o;
        id<MTLBuffer> bufScores = (__bridge id<MTLBuffer>)scores;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufQ offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufK offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufV offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufO offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufScores offset:0 atIndex:4];
        [gSessionEncoder setBytes:&H length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&S length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&D length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&scale length:sizeof(float) atIndex:8];
        // Grid = H threadgroups, EACH with 128 threads (matching ATTN_DECODE_TG
        // in the .metal) - the real thread total = H*128 (e.g. H=8 -> 1024 threads),
        // NOT just H threads like the faulty v1.
        const NSUInteger TG = 128;
        MTLSize gridSize = MTLSizeMake((NSUInteger)H, 1, 1);
        MTLSize groupSize = MTLSizeMake(TG, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// NEW: FUSED attention decode - ALL H heads in EXACTLY one dispatch.
// See the full explanation in metal_shim.h and attention_decode_all_heads.metal.
// Written from scratch, sharing NO pipeline/logic with the GQA function above
// (that kernel is known to have produced wrong results).
int metal_attention_decode_all_heads_enc(MetalBufferHandle q, MetalBufferHandle kAll, MetalBufferHandle vAll,
                                          MetalBufferHandle o,
                                          int H, int HKV, int D, int S, int maxSeqLen) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_decode_all_heads_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_decode_all_heads_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)q;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)kAll;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)vAll;
        id<MTLBuffer> bufO = (__bridge id<MTLBuffer>)o;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufQ offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufK offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufV offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufO offset:0 atIndex:3];
        [gSessionEncoder setBytes:&H         length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&HKV       length:sizeof(int) atIndex:5];
        [gSessionEncoder setBytes:&D         length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&S         length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&maxSeqLen length:sizeof(int) atIndex:8];
        // The "scores" threadgroup memory (threadgroup(0) in the kernel) is sized
        // DYNAMICALLY by S (unlike the resident KV, S changes EVERY TOKEN) - setThreadgroupMemoryLength
        // MUST be called BEFORE dispatch, with S floats.
        // (shared_reduce in the kernel is a FIXED array ATTN_ALL_TG=128, declared
        // inside the kernel, NO separate host allocation needed.)
        NSUInteger scoresBytes = (NSUInteger)S * sizeof(float);
        [gSessionEncoder setThreadgroupMemoryLength:scoresBytes atIndex:0];
        const NSUInteger TG = 128;
        MTLSize gridSize = MTLSizeMake((NSUInteger)H, 1, 1);
        MTLSize groupSize = MTLSizeMake(TG, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// Prefill flash attention (attention_prefill_flash.metal): causal GQA attention
// of T new rows over the resident KV cache [HKV, maxSeqLen, D], one dispatch.
typedef struct { uint32_t T, H, HKV, D, pos, maxSeqLen; float scaleLog2; } MetalFlashParams;
typedef struct { uint32_t T, HKV, D, pos, maxSeqLen; } MetalKvScatterParams;

int metal_attention_prefill_flash_enc(MetalBufferHandle q, MetalBufferHandle kAll, MetalBufferHandle vAll,
                                      MetalBufferHandle o, int T, int H, int HKV, int D, int pos,
                                      int maxSeqLen, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_prefill_flash_enc] metal_session_begin not called"); return 0; }
        if (T <= 0 || H <= 0 || HKV <= 0 || D <= 0 || D > 128 || pos < 0 || pos + T > maxSeqLen) return 0;
        const char *flashMode = getenv("BYBY_FLASH_ATTENTION");
        id<MTLComputePipelineState> pipeline =
            (flashMode && strcmp(flashMode, "scalar") == 0) ? nil :
            getPipeline(@"attention_prefill_flash_fp16_kernel");
        if (!pipeline) pipeline = getPipeline(@"attention_prefill_flash_kernel");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) return 0;
        MetalFlashParams params = { (uint32_t)T, (uint32_t)H, (uint32_t)HKV, (uint32_t)D,
                                    (uint32_t)pos, (uint32_t)maxSeqLen, scale * 1.4426950408889634f };
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)q offset:0 atIndex:0];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)kAll offset:0 atIndex:1];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)vAll offset:0 atIndex:2];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)o offset:0 atIndex:3];
        [gSessionEncoder setBytes:&params length:sizeof(params) atIndex:4];
        MTLSize gridSize = MTLSizeMake((NSUInteger)((T + 15) / 16), (NSUInteger)H, 1);
        MTLSize groupSize = MTLSizeMake(128, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_attention_prefill_flash_fp16_supported(void) {
    const char *flashMode = getenv("BYBY_FLASH_ATTENTION");
    return !(flashMode && strcmp(flashMode, "scalar") == 0) &&
        getPipeline(@"attention_prefill_flash_fp16_kernel") != nil;
}

int metal_kv_scatter_enc(MetalBufferHandle src, MetalBufferHandle dst, int T, int HKV, int D,
                         int pos, int maxSeqLen) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_kv_scatter_enc] metal_session_begin not called"); return 0; }
        if (T <= 0 || pos < 0 || pos + T > maxSeqLen) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"kv_cache_scatter_kernel");
        if (!pipeline) return 0;
        MetalKvScatterParams params = { (uint32_t)T, (uint32_t)HKV, (uint32_t)D, (uint32_t)pos,
                                        (uint32_t)maxSeqLen };
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [gSessionEncoder setBytes:&params length:sizeof(params) atIndex:2];
        NSUInteger total = (NSUInteger)T * (NSUInteger)HKV * (NSUInteger)D;
        MTLSize gridSize = MTLSizeMake((total + 255) / 256, 1, 1);
        MTLSize groupSize = MTLSizeMake(256, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_session_end(void) {
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf) { return 0; }  // CHANGED: no NSLog spam - the safety logic stays (returns 0 so the caller knows no session is open)
        [gSessionEncoder endEncoding];
        [gSessionCmdBuf commit];
        [gSessionCmdBuf waitUntilCompleted];
        BOOL ok = (gSessionCmdBuf.status == MTLCommandBufferStatusCompleted);
        if (!ok) {
            NSLog(@"[metal_session_end] command buffer FAILED, status=%ld, error=%@",
                  (long)gSessionCmdBuf.status, gSessionCmdBuf.error);
            // NEW: keeps the REAL description so metal_get_last_session_error()
            // can return it to Nim - before, this information was ONLY in NSLog and was
            // lost entirely when Nim only got "sessionOk=false" (a bare bool).
            NSString *statusStr = [NSString stringWithFormat:@"status=%ld", (long)gSessionCmdBuf.status];
            NSString *errStr = gSessionCmdBuf.error ? [gSessionCmdBuf.error description] : @"(cmdBuf.error la nil)";
            gLastSessionEndError = [NSString stringWithFormat:@"%@ %@", statusStr, errStr];
        } else {
            gLastSessionEndError = nil;   // this one SUCCEEDED - clear the old error so it is not misreported
        }
        gSessionEncoder = nil;
        gSessionCmdBuf = nil;
        return ok ? 1 : 0;
    }
}

// NEW - "commitAndContinue" (following Apple's MPSCommandBuffer and
// PyTorch MPS SyncType::COMMIT_ADAPTIVE, see MPSStream.mm):
// this is the RIGHT ROOT fix for GPU timeouts, not a "flush+wait" patch
// or "retry after an error" (2 approaches tried before: one too slow because
// each waitUntilCompleted is a REAL CPU<->GPU sync on an old iGPU, the other
// did not prevent the error in the first place).
//
// Idea: one MTLCommandQueue executes command buffers IN COMMIT ORDER
// (FIFO), AUTOMATICALLY, without the CPU waiting for each buffer before encoding
// the next. So instead of "one huge buffer for all 64 layers" (killed by the
// watchdog because it is too long) or "a REAL wait (waitUntilCompleted) every few layers"
// (slow because of GPU wake/sleep each time), the current buffer is COMMITTED
// WITHOUT WAITING and a new buffer is opened RIGHT AWAY to keep encoding - the GPU runs
// the committed buffer in the background while the CPU encodes the next (like
// a print queue), similar to MPSCommandBuffer's commitAndContinue().
//
// Result: (1) EVERY buffer is small, so the chance of a watchdog kill caused by one buffer is
// VERY low (no more "64 layers in one lump"); (2) NO
// CPU<->GPU sync cost (no waitUntilCompleted in between), so it is not slow
// like the earlier "flush"; (3) ONLY ONE REAL waitUntilCompleted - in
// the final metal_session_end(), when the logits must be read back to the CPU (a
// REQUIRED sync point because the logits are needed to choose the next token).
//
// If a committed (not waited) buffer later REALLY fails (e.g. still a watchdog although
// it was split) - since every buffer shares one gQueue, that error "poisons" the
// queue and CASCADES to the later buffers, so metal_session_end() finally
// returns 0 (because the LAST buffer is also "Ignored" like a domino) - then the Nim code
// (sessionRecoverQueue + retry, already there) is still the last safety net.
int metal_session_commit_continue(void) {
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf) {
            NSLog(@"[metal_session_commit_continue] no session is open");
            return 0;
        }
        [gSessionEncoder endEncoding];
        // NON-BLOCKING (async) log if this buffer eventually fails - only for debugging,
        // no speed impact because nothing waits for the handler.
        [gSessionCmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buf) {
            if (buf.status != MTLCommandBufferStatusCompleted) {
                NSLog(@"[metal_session_commit_continue] a buffer in the middle failed (will cascade to later buffers): status=%ld, error=%@",
                      (long)buf.status, buf.error);
            }
        }];
        [gSessionCmdBuf commit];               // NO waitUntilCompleted - this is the key difference
        gSessionCmdBuf = [gQueue commandBuffer];
        if (!gSessionCmdBuf) { NSLog(@"[metal_session_commit_continue] new commandBuffer returned nil"); gSessionEncoder = nil; return 0; }
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        if (!gSessionEncoder) { NSLog(@"[metal_session_commit_continue] new computeCommandEncoder returned nil"); gSessionCmdBuf = nil; return 0; }
        return 1;
    }
}



const char* metal_get_last_session_error(void) {
    return gLastSessionEndError ? gLastSessionEndError.UTF8String : "";
}

int metal_buffer_read(MetalBufferHandle h, float* outData, int n) {
    @autoreleasepool {
        if (!h) return 0;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)h;
        // CHANGED (latent crash): [buf contents] is ONLY valid for Shared/
        // Managed buffers - it returns nil (or invalid) for Private buffers, and a memcpy from
        // it crashes or reads garbage. The buffers suballocated from the resident KV heap
        // (see metal_alloc_heap) are now Private (fix for the "Shared
        // storage mode disallowed" crash on Intel) -> verifyKvResidentOnce reading
        // them back through this function NEEDS a Private path. Fix: if Private, blit into a
        // temporary Shared staging buffer (a separate blit command encoder, destroyed
        // right after reading), then memcpy from the staging buffer - slightly slower (its own
        // command buffer + waitUntilCompleted) but CORRECT, and this
        // path only runs for debugging/verification (not the hot inference
        // path of every token).
        if (buf.storageMode == MTLStorageModePrivate) {
            if (!gQueue) { NSLog(@"[metal_buffer_read] gQueue is not initialised"); return 0; }
            size_t bytes = (size_t)n * sizeof(float);
            id<MTLBuffer> staging = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!staging) { NSLog(@"[metal_buffer_read] tao staging buffer FAILED"); return 0; }
            id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromBuffer:buf sourceOffset:0 toBuffer:staging destinationOffset:0 size:bytes];
            [blit endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            if (cmdBuf.status == MTLCommandBufferStatusError) {
                NSLog(@"[metal_buffer_read] blit that bai: %@", cmdBuf.error);
                return 0;
            }
            memcpy(outData, [staging contents], bytes);
            return 1;
        }
        memcpy(outData, [buf contents], (size_t)n * sizeof(float));
        return 1;
    }
}

void metal_buffer_free(MetalBufferHandle h) {
    @autoreleasepool {
        if (!h) return;
        // __bridge_transfer hands ARC ownership back to a temporary id<MTLBuffer>
        // variable that releases it when leaving the scope - the counterpart of
        // the __bridge_retained used by metal_upload/metal_alloc_scratch to create the handle.
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)h;
        (void)buf;
    }
}

int metal_attention_fused_backward(const char* kernel_src, const float* q, const float* k, const float* v,
                                    const float* s_matrix, const float* dy,
                                    float* dq, float* dk, float* dv,
                                    int B, int H, int S, int D, float scale, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (S > 256) { NSLog(@"[metal_attention_fused_backward] S=%d exceeds the kernel limit of 256", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_backward_kernel");
        if (!pipeline) return 0;
        size_t bytesQKV = (size_t)B * (size_t)H * (size_t)S * (size_t)D * sizeof(float);
        size_t bytesS = (size_t)B * (size_t)H * (size_t)S * (size_t)S * sizeof(float);
        id<MTLBuffer> bufQ = [gDevice newBufferWithBytes:q length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDevice newBufferWithBytes:k length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDevice newBufferWithBytes:v length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [gDevice newBufferWithBytes:s_matrix length:bytesS options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytesQKV options:MTLResourceStorageModeShared];
        // dq/dk/dv: the kernel uses "+=" (dq) / atomic adds (dk, dv) -> the buffers MUST
        // be zero-initialised before launch (unlike bufO/bufS in the forward, which are
        // fully overwritten). newBufferWithLength does not guarantee
        // zeros per the Metal spec -> an explicit memset to be sure.
        id<MTLBuffer> bufDq = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDk = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDv = [gDevice newBufferWithLength:bytesQKV options:MTLResourceStorageModeShared];
        if (!bufQ || !bufK || !bufV || !bufS || !bufDy || !bufDq || !bufDk || !bufDv) return 0;
        memset([bufDq contents], 0, bytesQKV);
        memset([bufDk contents], 0, bytesQKV);
        memset([bufDv contents], 0, bytesQKV);
        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufV offset:0 atIndex:2];
        [encoder setBuffer:bufS offset:0 atIndex:3];
        [encoder setBuffer:bufDy offset:0 atIndex:4];
        [encoder setBuffer:bufDq offset:0 atIndex:5];
        [encoder setBuffer:bufDk offset:0 atIndex:6];
        [encoder setBuffer:bufDv offset:0 atIndex:7];
        [encoder setBytes:&B length:sizeof(int) atIndex:8];
        [encoder setBytes:&H length:sizeof(int) atIndex:9];
        [encoder setBytes:&S length:sizeof(int) atIndex:10];
        [encoder setBytes:&D length:sizeof(int) atIndex:11];
        [encoder setBytes:&scale length:sizeof(float) atIndex:12];
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(1, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_attention_fused_backward] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(dq, [bufDq contents], bytesQKV);
        memcpy(dk, [bufDk contents], bytesQKV);
        memcpy(dv, [bufDv contents], bytesQKV);
        return 1;
    }
}
// ─────────────────────────────────────────────────────────────
// REAL GPU PORT (round 2): gather / argmax (value+index) / compare (bitmask)
// / dropout RNG - the kernel sources are appended to the end of vecop_matmul.metal, so
// the kernel_src passed here is the combined kMetalKernelSrc (no separate
// library compiled) - the same per-name pipeline cache (getPipeline)
// as metal_vecop above.
// ─────────────────────────────────────────────────────────────

int metal_gather(const char* kernel_src, const float* src, int srcLen,
                  const int* idx, float* dst, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"gather_kernel");
        if (!pipeline) return 0;

        size_t srcBytes = (size_t)srcLen * sizeof(float);
        size_t idxBytes = (size_t)n * sizeof(int);
        size_t dstBytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufSrc = [gDevice newBufferWithBytes:src length:srcBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufIdx = [gDevice newBufferWithBytes:idx length:idxBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [gDevice newBufferWithLength:dstBytes options:MTLResourceStorageModeShared];
        if (!bufSrc || !bufIdx || !bufDst) { NSLog(@"[metal_gather] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufSrc offset:0 atIndex:0];
        [encoder setBuffer:bufIdx offset:0 atIndex:1];
        [encoder setBuffer:bufDst offset:0 atIndex:2];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_gather] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(dst, [bufDst contents], dstBytes);
        return 1;
    }
}

int metal_argmax(const char* kernel_src, const float* x, float* outVal,
                  int* outIdx, int outer, int dimSize, int inner, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"argmax_kernel");
        if (!pipeline) return 0;

        int pairs = outer * inner;
        size_t xBytes = (size_t)(outer * dimSize * inner) * sizeof(float);
        size_t pairBytesF = (size_t)pairs * sizeof(float);
        size_t pairBytesI = (size_t)pairs * sizeof(int);
        id<MTLBuffer> bufX = [gDevice newBufferWithBytes:x length:xBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufVal = [gDevice newBufferWithLength:pairBytesF options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufIdx = [gDevice newBufferWithLength:pairBytesI options:MTLResourceStorageModeShared];
        if (!bufX || !bufVal || !bufIdx) { NSLog(@"[metal_argmax] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufVal offset:0 atIndex:1];
        [encoder setBuffer:bufIdx offset:0 atIndex:2];
        [encoder setBytes:&outer length:sizeof(int) atIndex:3];
        [encoder setBytes:&dimSize length:sizeof(int) atIndex:4];
        [encoder setBytes:&inner length:sizeof(int) atIndex:5];
        // threadgroup memory: sVal (float) + sIdx (int), dimSize elements each,
        // matching the [[threadgroup]] buffer(0)/buffer(1) declared in argmax_kernel.
        NSUInteger tgSize = (NSUInteger)dimSize;
        [encoder setThreadgroupMemoryLength:tgSize * sizeof(float) atIndex:0];
        [encoder setThreadgroupMemoryLength:tgSize * sizeof(int) atIndex:1];
        MTLSize gridSize = MTLSizeMake((NSUInteger)pairs * tgSize, 1, 1);
        MTLSize groupSize = MTLSizeMake(tgSize, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_argmax] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(outVal, [bufVal contents], pairBytesF);
        memcpy(outIdx, [bufIdx contents], pairBytesI);
        return 1;
    }
}

int metal_compare(const char* kernel_src, const float* a, const float* b,
                   unsigned char* mask, int op, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"compare_kernel");
        if (!pipeline) return 0;

        size_t fBytes = (size_t)n * sizeof(float);
        size_t mBytes = (size_t)n * sizeof(unsigned char);
        id<MTLBuffer> bufA = [gDevice newBufferWithBytes:a length:fBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [gDevice newBufferWithBytes:b length:fBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMask = [gDevice newBufferWithLength:mBytes options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufMask) { NSLog(@"[metal_compare] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufMask offset:0 atIndex:2];
        [encoder setBytes:&op length:sizeof(int) atIndex:3];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_compare] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(mask, [bufMask contents], mBytes);
        return 1;
    }
}

int metal_dropout_mask(const char* kernel_src, float* mask,
                        unsigned int seed, float p, float scale, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"dropout_rng_kernel");
        if (!pipeline) return 0;

        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufMask = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufMask) { NSLog(@"[metal_dropout_mask] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufMask offset:0 atIndex:0];
        [encoder setBytes:&seed length:sizeof(unsigned int) atIndex:1];
        [encoder setBytes:&p length:sizeof(float) atIndex:2];
        [encoder setBytes:&scale length:sizeof(float) atIndex:3];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_dropout_mask] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(mask, [bufMask contents], bytes);
        return 1;
    }
}

// ─────────────────────────────────────────────────────────────
// REAL GPU PORT (round 3) - LIKE TORCH: permute/split/concat, the device
// decodes the indices from a small O(rank)/O(numTensors) table passed through setBytes,
// no O(n) index array built on the host.
// ─────────────────────────────────────────────────────────────

int metal_permute(const char* kernel_src, const float* src, float* dst,
                   const int* newStrides, const int* oldStrides,
                   const int* dimsMap, int ndim, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"permute_strided");
        if (!pipeline) return 0;

        size_t srcBytes = (size_t)n * sizeof(float); // n = numel, enough for a src of the same size
        size_t dstBytes = (size_t)n * sizeof(float);
        size_t tblBytes = (size_t)ndim * sizeof(int);
        id<MTLBuffer> bufSrc = [gDevice newBufferWithBytes:src length:srcBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [gDevice newBufferWithLength:dstBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNewStr = [gDevice newBufferWithBytes:newStrides length:tblBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOldStr = [gDevice newBufferWithBytes:oldStrides length:tblBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDimsMap = [gDevice newBufferWithBytes:dimsMap length:tblBytes options:MTLResourceStorageModeShared];
        if (!bufSrc || !bufDst || !bufNewStr || !bufOldStr || !bufDimsMap) { NSLog(@"[metal_permute] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufSrc offset:0 atIndex:0];
        [encoder setBuffer:bufDst offset:0 atIndex:1];
        [encoder setBuffer:bufNewStr offset:0 atIndex:2];
        [encoder setBuffer:bufOldStr offset:0 atIndex:3];
        [encoder setBuffer:bufDimsMap offset:0 atIndex:4];
        [encoder setBytes:&ndim length:sizeof(int) atIndex:5];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_permute] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(dst, [bufDst contents], dstBytes);
        return 1;
    }
}

int metal_split_slice(const char* kernel_src, const float* src, int srcLen, float* dst,
                       int dimTotal, int thisSize, int offset, int inner, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"split_slice");
        if (!pipeline) return 0;

        size_t srcBytes = (size_t)srcLen * sizeof(float);
        size_t dstBytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufSrc = [gDevice newBufferWithBytes:src length:srcBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [gDevice newBufferWithLength:dstBytes options:MTLResourceStorageModeShared];
        if (!bufSrc || !bufDst) { NSLog(@"[metal_split_slice] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufSrc offset:0 atIndex:0];
        [encoder setBuffer:bufDst offset:0 atIndex:1];
        [encoder setBytes:&dimTotal length:sizeof(int) atIndex:2];
        [encoder setBytes:&thisSize length:sizeof(int) atIndex:3];
        [encoder setBytes:&offset length:sizeof(int) atIndex:4];
        [encoder setBytes:&inner length:sizeof(int) atIndex:5];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_split_slice] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(dst, [bufDst contents], dstBytes);
        return 1;
    }
}

int metal_concat_gather(const char* kernel_src, const float* flatSrc, int flatLen,
                         const int* srcBase, const int* srcDimSize,
                         const int* srcDimOffset, int numTensors,
                         float* dst, int totalDim, int inner, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"concat_gather");
        if (!pipeline) return 0;

        size_t flatBytes = (size_t)flatLen * sizeof(float);
        size_t tblBytes = (size_t)numTensors * sizeof(int);
        size_t dstBytes = (size_t)n * sizeof(float);
        id<MTLBuffer> bufFlat = [gDevice newBufferWithBytes:flatSrc length:flatBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBase = [gDevice newBufferWithBytes:srcBase length:tblBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSize = [gDevice newBufferWithBytes:srcDimSize length:tblBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOff = [gDevice newBufferWithBytes:srcDimOffset length:tblBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [gDevice newBufferWithLength:dstBytes options:MTLResourceStorageModeShared];
        if (!bufFlat || !bufBase || !bufSize || !bufOff || !bufDst) { NSLog(@"[metal_concat_gather] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufFlat offset:0 atIndex:0];
        [encoder setBuffer:bufBase offset:0 atIndex:1];
        [encoder setBuffer:bufSize offset:0 atIndex:2];
        [encoder setBuffer:bufOff offset:0 atIndex:3];
        [encoder setBytes:&numTensors length:sizeof(int) atIndex:4];
        [encoder setBuffer:bufDst offset:0 atIndex:5];
        [encoder setBytes:&totalDim length:sizeof(int) atIndex:6];
        [encoder setBytes:&inner length:sizeof(int) atIndex:7];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_concat_gather] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }
        memcpy(dst, [bufDst contents], dstBytes);
        return 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// NEW: the remaining _enc functions for sessions - following 100% the idioms of
// metal_gather/metal_argmax/metal_compare/metal_dropout_mask/metal_permute/
// metal_split_slice/metal_concat_gather/metal_layernorm_backward/
// metal_attention_fused_backward above (same pipelines, same buffer indices),
// the only differences: the buffers are already resident (no newBufferWithBytes from the host), no
// separate commit/wait (the open gSessionEncoder is used, memoryBarrier instead of
// endEncoding/commit/wait). NOT YET RUN ON A REAL METAL GPU - must be
// built + tested on a real Mac before being used for real training.
// ─────────────────────────────────────────────────────────────────────────

int metal_layernorm_backward_enc(MetalBufferHandle dy, MetalBufferHandle x,
                                  MetalBufferHandle gamma, MetalBufferHandle beta,
                                  MetalBufferHandle dx, MetalBufferHandle dgamma, MetalBufferHandle dbeta,
                                  int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_layernorm_backward_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"layernorm_backward_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufDy = (__bridge id<MTLBuffer>)dy;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufGamma = (__bridge id<MTLBuffer>)gamma;
        id<MTLBuffer> bufBeta = (__bridge id<MTLBuffer>)beta;
        id<MTLBuffer> bufDx = (__bridge id<MTLBuffer>)dx;
        id<MTLBuffer> bufDgamma = (__bridge id<MTLBuffer>)dgamma;
        id<MTLBuffer> bufDbeta = (__bridge id<MTLBuffer>)dbeta;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufDy offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufGamma offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufBeta offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufDx offset:0 atIndex:4];
        [gSessionEncoder setBuffer:bufDgamma offset:0 atIndex:5];
        [gSessionEncoder setBuffer:bufDbeta offset:0 atIndex:6];
        [gSessionEncoder setBytes:&rows length:sizeof(int) atIndex:7];
        [gSessionEncoder setBytes:&cols length:sizeof(int) atIndex:8];
        [gSessionEncoder setBytes:&eps length:sizeof(float) atIndex:9];
        int gridN = rows > cols ? rows : cols;
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)gridN);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)gridN, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_attention_fused_backward_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                        MetalBufferHandle s_matrix, MetalBufferHandle dy,
                                        MetalBufferHandle dq, MetalBufferHandle dk, MetalBufferHandle dv,
                                        int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_fused_backward_enc] metal_session_begin not called"); return 0; }
        if (S > 256) { NSLog(@"[metal_attention_fused_backward_enc] S=%d exceeds the limit of 256", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_backward_kernel");
        if (!pipeline) return 0;
        size_t bytesQKV = (size_t)B * (size_t)H * (size_t)S * (size_t)D * sizeof(float);
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)q;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)k;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)v;
        id<MTLBuffer> bufS = (__bridge id<MTLBuffer>)s_matrix;
        id<MTLBuffer> bufDy = (__bridge id<MTLBuffer>)dy;
        id<MTLBuffer> bufDq = (__bridge id<MTLBuffer>)dq;
        id<MTLBuffer> bufDk = (__bridge id<MTLBuffer>)dk;
        id<MTLBuffer> bufDv = (__bridge id<MTLBuffer>)dv;
        // dq/dk/dv use "+=" / atomic adds in the kernel -> they MUST be zero-initialised
        // first (unlike the forward, which overwrites fully) - the caller (gpubackend.nim)
        // allocates the buffers with metalSessionAllocScratch, and a blit
        // encoder is NOT available here, so the memset goes directly through contents
        // (the buffer is MTLResourceStorageModeShared on most Intel/
        // Apple Silicon iGPU machines, so contents is safely accessible from the CPU
        // EVEN while the session command buffer is open, because it is not committed yet).
        memset([bufDq contents], 0, bytesQKV);
        memset([bufDk contents], 0, bytesQKV);
        memset([bufDv contents], 0, bytesQKV);
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufQ offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufK offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufV offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufS offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufDy offset:0 atIndex:4];
        [gSessionEncoder setBuffer:bufDq offset:0 atIndex:5];
        [gSessionEncoder setBuffer:bufDk offset:0 atIndex:6];
        [gSessionEncoder setBuffer:bufDv offset:0 atIndex:7];
        [gSessionEncoder setBytes:&B length:sizeof(int) atIndex:8];
        [gSessionEncoder setBytes:&H length:sizeof(int) atIndex:9];
        [gSessionEncoder setBytes:&S length:sizeof(int) atIndex:10];
        [gSessionEncoder setBytes:&D length:sizeof(int) atIndex:11];
        [gSessionEncoder setBytes:&scale length:sizeof(float) atIndex:12];
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(1, 1, 1);
        [gSessionEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_gather_enc(MetalBufferHandle src, MetalBufferHandle idx, MetalBufferHandle dst, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_gather_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"gather_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufSrc = (__bridge id<MTLBuffer>)src;
        id<MTLBuffer> bufIdx = (__bridge id<MTLBuffer>)idx;
        id<MTLBuffer> bufDst = (__bridge id<MTLBuffer>)dst;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufSrc offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufIdx offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufDst offset:0 atIndex:2];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_argmax_enc(MetalBufferHandle x, MetalBufferHandle outVal, MetalBufferHandle outIdx,
                      int outer, int dimSize, int inner) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_argmax_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"argmax_kernel");
        if (!pipeline) return 0;
        int pairs = outer * inner;
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x;
        id<MTLBuffer> bufVal = (__bridge id<MTLBuffer>)outVal;
        id<MTLBuffer> bufIdx = (__bridge id<MTLBuffer>)outIdx;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufX offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufVal offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufIdx offset:0 atIndex:2];
        [gSessionEncoder setBytes:&outer length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&dimSize length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&inner length:sizeof(int) atIndex:5];
        NSUInteger tgSize = (NSUInteger)dimSize;
        [gSessionEncoder setThreadgroupMemoryLength:tgSize * sizeof(float) atIndex:0];
        [gSessionEncoder setThreadgroupMemoryLength:tgSize * sizeof(int) atIndex:1];
        MTLSize gridSize = MTLSizeMake((NSUInteger)pairs * tgSize, 1, 1);
        MTLSize groupSize = MTLSizeMake(tgSize, 1, 1);
        [gSessionEncoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_compare_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle mask, int op, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_compare_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"compare_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> bufMask = (__bridge id<MTLBuffer>)mask;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufA offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufB offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufMask offset:0 atIndex:2];
        [gSessionEncoder setBytes:&op length:sizeof(int) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_dropout_mask_enc(MetalBufferHandle mask, unsigned int seed, float p, float scale, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_dropout_mask_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"dropout_rng_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufMask = (__bridge id<MTLBuffer>)mask;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufMask offset:0 atIndex:0];
        [gSessionEncoder setBytes:&seed length:sizeof(unsigned int) atIndex:1];
        [gSessionEncoder setBytes:&p length:sizeof(float) atIndex:2];
        [gSessionEncoder setBytes:&scale length:sizeof(float) atIndex:3];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_permute_enc(MetalBufferHandle src, MetalBufferHandle dst,
                       MetalBufferHandle newStrides, MetalBufferHandle oldStrides,
                       MetalBufferHandle dimsMap, int ndim, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_permute_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"permute_strided");
        if (!pipeline) return 0;
        id<MTLBuffer> bufSrc = (__bridge id<MTLBuffer>)src;
        id<MTLBuffer> bufDst = (__bridge id<MTLBuffer>)dst;
        id<MTLBuffer> bufNewStr = (__bridge id<MTLBuffer>)newStrides;
        id<MTLBuffer> bufOldStr = (__bridge id<MTLBuffer>)oldStrides;
        id<MTLBuffer> bufDimsMap = (__bridge id<MTLBuffer>)dimsMap;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufSrc offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufDst offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufNewStr offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufOldStr offset:0 atIndex:3];
        [gSessionEncoder setBuffer:bufDimsMap offset:0 atIndex:4];
        [gSessionEncoder setBytes:&ndim length:sizeof(int) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_split_slice_enc(MetalBufferHandle src, MetalBufferHandle dst,
                           int dimTotal, int thisSize, int offset, int inner, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_split_slice_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"split_slice");
        if (!pipeline) return 0;
        id<MTLBuffer> bufSrc = (__bridge id<MTLBuffer>)src;
        id<MTLBuffer> bufDst = (__bridge id<MTLBuffer>)dst;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufSrc offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufDst offset:0 atIndex:1];
        [gSessionEncoder setBytes:&dimTotal length:sizeof(int) atIndex:2];
        [gSessionEncoder setBytes:&thisSize length:sizeof(int) atIndex:3];
        [gSessionEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        [gSessionEncoder setBytes:&inner length:sizeof(int) atIndex:5];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_concat_gather_enc(MetalBufferHandle flatSrc, MetalBufferHandle srcBase,
                             MetalBufferHandle srcDimSize, MetalBufferHandle srcDimOffset,
                             int numTensors, MetalBufferHandle dst, int totalDim, int inner, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_concat_gather_enc] metal_session_begin not called"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"concat_gather");
        if (!pipeline) return 0;
        id<MTLBuffer> bufFlat = (__bridge id<MTLBuffer>)flatSrc;
        id<MTLBuffer> bufBase = (__bridge id<MTLBuffer>)srcBase;
        id<MTLBuffer> bufDimSize = (__bridge id<MTLBuffer>)srcDimSize;
        id<MTLBuffer> bufDimOffset = (__bridge id<MTLBuffer>)srcDimOffset;
        id<MTLBuffer> bufDst = (__bridge id<MTLBuffer>)dst;
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufFlat offset:0 atIndex:0];
        [gSessionEncoder setBuffer:bufBase offset:0 atIndex:1];
        [gSessionEncoder setBuffer:bufDimSize offset:0 atIndex:2];
        [gSessionEncoder setBuffer:bufDimOffset offset:0 atIndex:3];
        [gSessionEncoder setBytes:&numTensors length:sizeof(int) atIndex:4];
        [gSessionEncoder setBuffer:bufDst offset:0 atIndex:5];
        [gSessionEncoder setBytes:&totalDim length:sizeof(int) atIndex:6];
        [gSessionEncoder setBytes:&inner length:sizeof(int) atIndex:7];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

/* ============================================================
 * LOW-LEVEL session — implementation
 * Paste this WHOLE block at the END of metal_shim.m (after the last function).
 * Depends on: gSessionEncoder, gLibrary, gDevice, getPipeline() already present.
 * ============================================================ */

static id<MTLComputePipelineState> gEncCurrentPipeline = nil;

/* Cache pipeline tu metal_jit_compile (bytecode -> Metal source). */
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* gJitPipelines = nil;

int metal_enc_pipeline(const char* kernelName) {
    @autoreleasepool {
        if (!gSessionEncoder) {
            NSLog(@"[metal_enc_pipeline] no session_begin yet");
            return 0;
        }
        if (!kernelName) return 0;
        NSString *name = [NSString stringWithUTF8String:kernelName];
        id<MTLComputePipelineState> p = nil;
        if (gJitPipelines) p = gJitPipelines[name];
        if (!p) p = getPipeline(name);
        if (!p) {
            NSLog(@"[metal_enc_pipeline] kernel not found: %@", name);
            return 0;
        }
        gEncCurrentPipeline = p;
        [gSessionEncoder setComputePipelineState:p];
        return 1;
    }
}

int metal_enc_bind_buffer(int index, MetalBufferHandle buf, int offsetBytes) {
    @autoreleasepool {
        if (!gSessionEncoder || !buf) return 0;
        if (index < 0) return 0;
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
        [gSessionEncoder setBuffer:b
                            offset:(NSUInteger)MAX(offsetBytes, 0)
                           atIndex:(NSUInteger)index];
        return 1;
    }
}

int metal_enc_bind_int(int index, int v) {
    @autoreleasepool {
        if (!gSessionEncoder || index < 0) return 0;
        [gSessionEncoder setBytes:&v length:sizeof(int) atIndex:(NSUInteger)index];
        return 1;
    }
}

int metal_enc_bind_float(int index, float v) {
    @autoreleasepool {
        if (!gSessionEncoder || index < 0) return 0;
        [gSessionEncoder setBytes:&v length:sizeof(float) atIndex:(NSUInteger)index];
        return 1;
    }
}

int metal_enc_bind_bytes(int index, const void* data, int nBytes) {
    @autoreleasepool {
        if (!gSessionEncoder || !data || nBytes <= 0 || index < 0) return 0;
        [gSessionEncoder setBytes:data length:(NSUInteger)nBytes atIndex:(NSUInteger)index];
        return 1;
    }
}

int metal_enc_dispatch(int gridX, int gridY, int gridZ, int tgX, int tgY, int tgZ) {
    @autoreleasepool {
        if (!gSessionEncoder || !gEncCurrentPipeline) {
            NSLog(@"[metal_enc_dispatch] needs a pipeline first (metal_enc_pipeline)");
            return 0;
        }
        NSUInteger tgx = tgX > 0 ? (NSUInteger)tgX : gEncCurrentPipeline.threadExecutionWidth;
        NSUInteger tgy = tgY > 0 ? (NSUInteger)tgY : 1;
        NSUInteger tgz = tgZ > 0 ? (NSUInteger)tgZ : 1;
        NSUInteger maxTG = gEncCurrentPipeline.maxTotalThreadsPerThreadgroup;
        if (tgx * tgy * tgz > maxTG) {
            tgx = maxTG;
            tgy = 1;
            tgz = 1;
        }
        MTLSize tg = MTLSizeMake(tgx, tgy, tgz);
        MTLSize grid = MTLSizeMake(
            (NSUInteger)(gridX > 0 ? gridX : 1),
            (NSUInteger)(gridY > 0 ? gridY : 1),
            (NSUInteger)(gridZ > 0 ? gridZ : 1));
        [gSessionEncoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        return 1;
    }
}

int metal_enc_dispatch_threads(int threadsX, int threadsY, int threadsZ,
                               int tgX, int tgY, int tgZ) {
    @autoreleasepool {
        if (!gSessionEncoder || !gEncCurrentPipeline) return 0;
        NSUInteger tgx = tgX > 0 ? (NSUInteger)tgX : gEncCurrentPipeline.threadExecutionWidth;
        NSUInteger tgy = tgY > 0 ? (NSUInteger)tgY : 1;
        NSUInteger tgz = tgZ > 0 ? (NSUInteger)tgZ : 1;
        NSUInteger maxTG = gEncCurrentPipeline.maxTotalThreadsPerThreadgroup;
        if (tgx * tgy * tgz > maxTG) {
            tgx = maxTG; tgy = 1; tgz = 1;
        }
        MTLSize tg = MTLSizeMake(tgx, tgy, tgz);
        MTLSize threads = MTLSizeMake(
            (NSUInteger)(threadsX > 0 ? threadsX : 1),
            (NSUInteger)(threadsY > 0 ? threadsY : 1),
            (NSUInteger)(threadsZ > 0 ? threadsZ : 1));
        [gSessionEncoder dispatchThreads:threads threadsPerThreadgroup:tg];
        // The Nim session_vm_dispatch.sessionDispatch wrapper immediately
        // calls metal_enc_barrier() so dependent VM dispatches see these writes.
        // Keep this primitive fence-free: its other direct users may add a
        // barrier at a wider scheduling boundary.
        return 1;
    }
}

int metal_enc_barrier(void) {
    @autoreleasepool {
        if (!gSessionEncoder) return 0;
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}


int metal_enc_set_threadgroup_bytes(int index, int nBytes) {
    @autoreleasepool {
        if (!gSessionEncoder || index < 0 || nBytes <= 0) return 0;
        /* Metal requires setThreadgroupMemoryLength before dispatch when the kernel
         * declares threadgroup(N). nBytes=4 is enough for kernels without real TG memory. */
        [gSessionEncoder setThreadgroupMemoryLength:(NSUInteger)nBytes
                                            atIndex:(NSUInteger)index];
        return 1;
    }
}

int metal_enc_simd_width(void) {
    if (!gEncCurrentPipeline) return 32;
    return (int)gEncCurrentPipeline.threadExecutionWidth;
}

int metal_enc_max_tg_threads(void) {
    if (!gEncCurrentPipeline) return 256;
    return (int)gEncCurrentPipeline.maxTotalThreadsPerThreadgroup;
}



/* ============================================================
 * metal_jit_compile — JIT bytecode/source -> pipeline (giu kernel cu)
 * ============================================================ */
int metal_recover_queue(void) {
    // NEW: according to Apple's documentation + community experience (Ryujinx/MoltenVK
    // "resumeLostDevice", MLX issue #3390 confirms it): AFTER
    // an MTLCommandBuffer REALLY fails (Timeout/StackOverflow/PageFault...),
    // its MTLCommandQueue is "poisoned" - EVERY later commandBuffer
    // on THAT queue is rejected by the driver with the error
    // "Ignored (for causing prior/excessive GPU errors)
    // (kIOAccelCommandBufferCallbackErrorSubmissionsIgnored)", PERMANENTLY,
    // until a NEW MTLCommandQueue is created. This function recreates gQueue from gDevice
    // (gDevice is NOT recreated - a really lost device (DeviceRemoved) is a different,
    // rare case, not handled here).
    if (!gDevice) { NSLog(@"[metal_recover] gDevice nil, cannot recover"); return 0; }
    id<MTLCommandQueue> newQueue = [gDevice newCommandQueue];
    if (!newQueue) { NSLog(@"[metal_recover] newCommandQueue that bai"); return 0; }
    gQueue = newQueue;
    NSLog(@"[metal_recover] Created a NEW MTLCommandQueue (the old queue was poisoned by an earlier GPU error)");
    return 1;
}

/* ---- MTLBinaryArchive: keeps JIT-compiled pipelines between runs ----
   Each new micro-VM kernel costs ~0.2-0.5 s to compile (newLibraryWithSource +
   newComputePipelineStateWithDescriptor). Prefilling a short prompt uses ~20
   kernels, i.e. ~4 s of 12 s wall time - paid again from scratch in EVERY process.
   Vulkan already has an on-disk shader cache; this is the Metal equivalent. */
static id<MTLBinaryArchive> gJitArchive = nil;
static NSString* gJitArchivePath = nil;
static dispatch_once_t gJitArchiveOnce;

static BOOL metal_jit_info_logging_enabled(void) {
    static dispatch_once_t once;
    static BOOL enabled = YES;
    dispatch_once(&once, ^{
        const char* env = getenv("BYBY_METAL_JIT_LOG");
        if (env && env[0] == '0' && env[1] == '\0') enabled = NO;
    });
    return enabled;
}

static void metal_jit_archive_save(void) {
    if (!gJitArchive || !gJitArchivePath) return;
    /* Serializing the archive goes through the external Metal compiler and can
       take minutes on Intel GPUs (where the call ultimately reports that it is
       unsupported). Recordings and short runs may opt out explicitly. */
    const char* saveEnv = getenv("BYBY_METAL_JIT_ARCHIVE_SAVE");
    if (saveEnv && saveEnv[0] == '0' && saveEnv[1] == '\0') return;
    static dispatch_once_t warnOnce;
    @autoreleasepool {
        NSError* err = nil;
        if ([gJitArchive serializeToURL:[NSURL fileURLWithPath:gJitArchivePath] error:&err]) {
            if (metal_jit_info_logging_enabled()) NSLog(@"[metal_jit] archive saved %@", gJitArchivePath);
        } else {
            /* serializeToURL is Apple-silicon only; on Intel it returns NO + a nil
               error. Warn only once so every run is not spammed. */
            dispatch_once(&warnOnce, ^{
                NSLog(@"[metal_jit] archive serialization not supported on this machine (%@) - in-process cache only", err ?: @"Intel");
            });
        }
    }
}

static void metal_jit_archive_init(void) {
    if (!gDevice) return;
    @autoreleasepool {
        if (@available(macOS 11.0, *)) {
            const char* env = getenv("BYBY_METAL_JIT_ARCHIVE");
            NSString* path = nil;
            if (env && env[0]) {
                path = [NSString stringWithUTF8String:env];
            } else {
                NSArray* dirs = NSSearchPathForDirectoriesInDomains(
                    NSCachesDirectory, NSUserDomainMask, YES);
                NSString* base = dirs.count ? dirs[0] : NSTemporaryDirectory();
                NSString* dir = [base stringByAppendingPathComponent:@"nimformers"];
                [[NSFileManager defaultManager] createDirectoryAtPath:dir
                                         withIntermediateDirectories:YES
                                                          attributes:nil
                                                               error:NULL];
                if (![[NSFileManager defaultManager] fileExistsAtPath:dir]) {
                    dir = NSTemporaryDirectory();
                }
                path = [dir stringByAppendingPathComponent:@"metal-jit.binarchive"];
            }
            NSString* parent = [path stringByDeletingLastPathComponent];
            NSError* dirErr = nil;
            if (parent.length > 0 &&
                ![[NSFileManager defaultManager] createDirectoryAtPath:parent
                                          withIntermediateDirectories:YES
                                                           attributes:nil
                                                                error:&dirErr]) {
                NSLog(@"[metal_jit] cannot create the archive directory %@ (%@) - running without a cache", parent, dirErr);
                return;
            }
            MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
            BOOL loaded = [[NSFileManager defaultManager] fileExistsAtPath:path];
            if (loaded) ad.url = [NSURL fileURLWithPath:path];
            NSError* err = nil;
            gJitArchive = [gDevice newBinaryArchiveWithDescriptor:ad error:&err];
            if (!gJitArchive) {
                NSLog(@"[metal_jit] cannot open the archive (%@) - running without a cache", err);
                return;
            }
            gJitArchivePath = path;
            atexit(metal_jit_archive_save);
            if (metal_jit_info_logging_enabled()) NSLog(@"[metal_jit] archive %@ (%@)", loaded ? @"loaded" : @"new", path);
        }
    }
}

int metal_jit_compile(const char* source, const char* kernelName) {
    if (!source || !kernelName) return 0;
    @autoreleasepool {
        if (!gDevice) {
            NSLog(@"[metal_jit] gDevice nil");
            return 0;
        }
        if (!gJitPipelines) {
            gJitPipelines = [NSMutableDictionary dictionary];
        }
        dispatch_once(&gJitArchiveOnce, ^{ metal_jit_archive_init(); });
        NSString* name = [NSString stringWithUTF8String:kernelName];
        if (gJitPipelines[name]) {
            return 1; /* already cached */
        }
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        NSString* src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [gDevice newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            NSLog(@"[metal_jit] compile fail: %@", err);
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) {
            NSLog(@"[metal_jit] function '%@' not found in JIT library", name);
            return 0;
        }
        MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = fn;
        desc.maxCallStackDepth = 32; /* avoids the Stack Overflow Exception (Code=12) */
        if (gJitArchive) {
            NSError* archiveErr = nil;
            if ([gJitArchive addComputePipelineFunctionsWithDescriptor:desc error:&archiveErr]) {
                desc.binaryArchives = @[gJitArchive];
            } else {
                NSLog(@"[metal_jit] cannot add pipeline %@ to the archive (%@)", name, archiveErr);
            }
        }
        id<MTLComputePipelineState> pso =
            [gDevice newComputePipelineStateWithDescriptor:desc
                                                   options:MTLPipelineOptionNone
                                                reflection:nil
                                                     error:&err];
        if (!pso && gJitArchive) {
            desc.binaryArchives = nil;
            pso = [gDevice newComputePipelineStateWithDescriptor:desc
                                                         options:MTLPipelineOptionNone
                                                      reflection:nil
                                                           error:&err];
        }
        if (!pso) {
            NSLog(@"[metal_jit] pipeline fail: %@", err);
            return 0;
        }
        gJitPipelines[name] = pso;
        if (metal_jit_info_logging_enabled()) NSLog(@"[metal_jit] OK kernel=%@", name);
        return 1;
    }
}

int metal_device_max_tg_threads(void) {
    if (!gDevice) return 0;
    return (int)gDevice.maxThreadsPerThreadgroup.width;
}
int metal_device_max_tg_bytes(void) {
    if (!gDevice) return 0;
    return (int)gDevice.maxThreadgroupMemoryLength;
}
