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
// A/B TEST kernel matmul_q4 (THEM MOI): chi doi 2 dong duoi day de
// thu bien the khac, KHONG can sua rai rac o metal_matmul_q4() va
// metal_matmul_q4_enc() (2 ham, moi ham co ca ten kernel lan
// kMMQ4_NDST rieng - de quen doi 1 trong 4 cho -> rowBlocks tinh SAI
// -> ket qua SAI hoac crash, xem giai thich trong vecop_matmul.metal).
//
//   "matmul_q4_kernel"              NDST=8  <- MAC DINH, ban da do
//                                              ON DINH (0.86 tok/s,
//                                              1156.6 ms/token) tren
//                                              Intel Iris Plus 640.
//   "matmul_q4_kernel_ndst2_paired" NDST=2  <- thu nghiem, giong
//   "matmul_q4_kernel_ndst4_paired" NDST=4     N_R0_Q4_K=2 cua ggml,
//                                              CHUA CHAY THU TREN
//                                              GPU METAL THAT.
//
// Doi CA HAI dong duoi (ten kernel + so NDST) CUNG LUC khi thu:
#define MMQ4_KERNEL_NAME_ACTIVE @"matmul_q4_kernel_ndst2_paired"
#define MMQ4_NDST_ACTIVE 2

// ─────────────────────────────────────────────────────────────
// State toàn cục, khởi tạo 1 lần, sống tới khi process thoát.
// ─────────────────────────────────────────────────────────────
static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLLibrary> gLibrary = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *gPipelineCache = nil;
static dispatch_once_t gInitOnce;

// MTLCreateSystemDefaultDevice() chỉ trả về ĐÚNG MỘT GPU "mặc định" theo lựa
// chọn của hệ thống. Trên Mac Apple Silicon chỉ có 1 GPU nên luôn ổn, nhưng
// trên Mac Intel máy có thể có NHIỀU GPU (iGPU Intel + dGPU AMD, hoặc thêm
// eGPU rời) -- default device đôi khi không phải GPU thực sự đang "sống"
// (vd. dGPU đang ngủ do auto graphics-switching, hoặc app không có quyền
// power-on dGPU). Dùng MTLCopyAllDevices() (chỉ có trên macOS, không có trên
// iOS) để liệt kê toàn bộ GPU và tự chọn/thử lần lượt thay vì tin vào 1 default.
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

    // Ưu tiên: GPU rời/hiệu năng cao (không lowPower) và không headless.
    for (id<MTLDevice> d in all) {
        if (!d.isLowPower) {
            NSLog(@"[metal] Chon GPU hieu nang cao: %@", d.name);
            return d;
        }
    }
    // Không có GPU rời -> lấy GPU đầu tiên tìm được (vd. iGPU tích hợp).
    NSLog(@"[metal] Khong co GPU rieng, dung GPU dau tien: %@", all[0].name);
    return all[0];
}

// Chọn GPU MỘT lần, log MỘT lần (metal_available() và các nơi khác gọi hàng trăm lần).
static id<MTLDevice> pickMetalDevice(void) {
    static id<MTLDevice> cached = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ cached = pickMetalDeviceUncached(); });
    return cached;
}

// Khởi tạo device + queue + library MỘT LẦN DUY NHẤT cho toàn bộ process.
// Trả về NO nếu thất bại (vd. không có GPU, hoặc compile MSL lỗi).
static BOOL ensureMetalInit(const char* kernel_src) {
    __block BOOL ok = YES;
    dispatch_once(&gInitOnce, ^{
        gDevice = pickMetalDevice();
        if (!gDevice) { NSLog(@"[metal] khong tim thay GPU Metal nao"); ok = NO; return; }

        gQueue = [gDevice newCommandQueue];
        if (!gQueue) { NSLog(@"[metal] newCommandQueue tra ve nil"); ok = NO; return; }

        NSString *source = [NSString stringWithUTF8String:kernel_src];
        NSError *error = nil;
        gLibrary = [gDevice newLibraryWithSource:source options:nil error:&error];
        if (!gLibrary) { NSLog(@"[metal] newLibraryWithSource FAILED (compile 1 lan luc khoi dong): %@", error); ok = NO; return; }

        gPipelineCache = [NSMutableDictionary dictionary];
        NSLog(@"[metal] Khoi tao xong: device=%@, kernel library da compile 1 lan.", gDevice.name);
        // THEM MOI: in RO RANG kernel matmul_q4 nao dang active + NDST bao
        // nhieu - tranh tinh trang "khong nho da doi chua" khi A/B test
        // (xem MMQ4_KERNEL_NAME_ACTIVE/MMQ4_NDST_ACTIVE o dau file). Log
        // NAY LUON hien (khong can BYBY_DEBUG_PATH), giong dong log GPU o
        // tren - vi day la thong tin cau hinh tinh, khong phai debug runtime.
        NSLog(@"[metal] matmul_q4 dang active: kernel=%@ NDST=%d", MMQ4_KERNEL_NAME_ACTIVE, MMQ4_NDST_ACTIVE);
    });
    return ok && gDevice != nil && gLibrary != nil;
}

// Lấy pipeline state đã cache theo tên hàm; compile pipeline (không compile
// lại source) chỉ ở lần đầu gặp tên hàm đó, các lần sau lấy thẳng từ cache.
static id<MTLComputePipelineState> getPipeline(NSString *fname) {
    id<MTLComputePipelineState> cached = gPipelineCache[fname];
    if (cached) return cached;

    id<MTLFunction> fn = [gLibrary newFunctionWithName:fname];
    if (!fn) { NSLog(@"[metal] newFunctionWithName:%@ tra ve nil", fname); return nil; }

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

// Chọn biến thể matmul_kernel_t{8,16,32} lớn nhất mà GPU đang chạy CHO
// PHÉP THẬT SỰ, đo bằng pipeline.maxTotalThreadsPerThreadgroup của chính
// pipeline đó (không đoán theo tên GPU / lowPower flag - có thể sai giữa
// các driver/OS version). Cache lại 1 lần vì kết quả không đổi trong đời
// process (cùng device, cùng library).
static int gMatmulTile = 0;                    // 0 = chưa chọn
static id<MTLComputePipelineState> gMatmulPipeline = nil;

static id<MTLComputePipelineState> pickMatmulPipeline(int *outTile) {
    if (gMatmulPipeline != nil) { *outTile = gMatmulTile; return gMatmulPipeline; }
    // SUA: BỎ auto-chọn 16/32 - đo thực tế trên Iris Plus 640 không nhanh
    // hơn rõ rệt, và việc NSLog chỉ chạy ở LẦN GỌI ĐẦU TIÊN (dù đã có
    // warm-up) vẫn có nguy cơ trùng thời điểm generate nếu warm-up bị JIT/
    // trình biên dịch tối ưu bỏ qua với input quá nhỏ. Ép cứng về t8 - đã
    // biết ổn định, đơn giản, không phụ thuộc đo đạc runtime nữa.
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

// THÊM: matmul qua MPS thay vì kernel tay viết. Ưu điểm so với
// matmul_kernel_t{8,16,32}: MPS tự chọn thuật toán/tiling nội bộ theo
// đúng GPU đang chạy (kể cả simdgroup_matrix trên Apple Silicon mà kernel
// MSL tay viết ở trên KHÔNG dùng), không cần ta đoán tile size hay lo
// GPU Timeout do threadgroup vượt giới hạn - MPS tự đảm bảo dispatch hợp lệ
// cho GPU hiện tại. Nhược điểm: thêm 1 bước wrap MPSMatrix quanh MTLBuffer,
// và availability phụ thuộc GPU có hỗ trợ MPS hay không (kiểm tra qua
// MPSSupportsMTLDevice) - vì vậy hàm này PHẢI có fallback ở tầng gọi.
int metal_matmul_mps(const float* a, const float* b, float* c, int m, int k, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    // Hard policy: M=1 decode must use native kernels, even for direct callers.
    if (m <= 1) return 0;
    @autoreleasepool {
        // Dùng chung device/queue đã init sẵn bởi ensureMetalInit qua 1 lần
        // gọi metal_available()-style init nhẹ (không cần compile lib MSL
        // của mình vì MPS không cần kernel_src).
        if (gDevice == nil) {
            gDevice = pickMetalDevice();
            if (gDevice) gQueue = [gDevice newCommandQueue];
        }
        if (!gDevice || !gQueue) { NSLog(@"[metal_matmul_mps] khong co GPU/queue"); return 0; }
        if (!MPSSupportsMTLDevice(gDevice)) {
            NSLog(@"[metal_matmul_mps] GPU %@ khong ho tro MPS - fallback ve kernel tay viet", gDevice.name);
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

// SỬA BUG NGHIÊM TRỌNG: hàm này TRƯỚC ĐÂY thiếu hẳn dòng khai báo signature
// (chỉ còn "@autoreleasepool {" trơ trọi, không nằm trong bất kỳ hàm C nào)
// -> file .m này không compile được bằng clang, hoặc tệ hơn là bị gộp nhầm
// vào thân hàm metal_matmul_q4() phía trên tuỳ trình biên dịch/preprocessor,
// khiến metal_activation() (dùng cho sigmoid trong SwiGLU FFN) không tồn tại
// hoặc chạy sai hoàn toàn -> đây nhiều khả năng chính là nguyên nhân sinh
// token rác (SwiGLU sai ngay từ sigmoid thì toàn bộ FFN mỗi layer đều sai).
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
        // Kernel dùng 1 thread cho MỖI hàng (dx) VÀ, khi id < cols, cùng
        // thread đó gộp reduction dgamma/dbeta theo cột -> dispatch phải phủ
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
        if (S > 256) { NSLog(@"[metal_attention_fused] S=%d vuot gioi han 256 cua kernel (mang scores[256] cung)", S); return 0; }
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
        // Grid: x = bh (0..B*H-1), y = ti (0..S-1). Mỗi thread tự lo trọn 1
        // hàng causal - không cần threadgroup memory / barrier ở kernel này
        // (khác softmax/layernorm) nên threadgroup size 1 là an toàn và đơn
        // giản nhất, tránh phải tính chia hết cho grid.
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
// API resident (session): nhiều op mã hoá vào CÙNG 1 MTLCommandBuffer, chỉ
// commit+waitUntilCompleted MỘT LẦN ở metal_session_end(). Bám sát 100%
// idiom đã chạy được ở metal_matmul/metal_softmax phía trên (cùng cách tạo
// buffer, cùng cách lấy pipeline, cùng cách dispatch) - chỉ khác dùng
// chung 1 encoder/command buffer thay vì tạo mới mỗi lệnh.
static id<MTLCommandBuffer> gSessionCmdBuf = nil;
static id<MTLComputeCommandEncoder> gSessionEncoder = nil;
// THÊM MỚI: giữ mô tả NSError THẬT của lần metal_session_end() gần nhất bị
// fail - xem giải thích đầy đủ ở metal_shim.h. strong ref (static) nên sống
// qua lần gọi hàm getter, chỉ bị THAY THẾ (không phải mất) ở lần
// metal_session_end() kế tiếp.
static NSString *gLastSessionEndError = nil;

int metal_session_begin(const char* kernel_src, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        if (gSessionCmdBuf != nil) { NSLog(@"[metal_session_begin] session cu chua end, bo qua"); return 0; }
        gSessionCmdBuf = [gQueue commandBuffer];
        if (!gSessionCmdBuf) { NSLog(@"[metal_session_begin] commandBuffer tra ve nil"); return 0; }
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        if (!gSessionEncoder) { NSLog(@"[metal_session_begin] computeCommandEncoder tra ve nil"); gSessionCmdBuf = nil; return 0; }
        return 1;
    }
}

MetalBufferHandle metal_upload(const float* data, int n) {
    @autoreleasepool {
        // SUA: xem giai thich o metal_upload_bytes ngay ben duoi - cung thieu
        // ensureMetalInit truoc day, gDevice NULL neu goi truoc metal_session_begin/
        // metal_matmul lan dau (đúng case resident-upload luc load model).
        if (!gDevice) { NSLog(@"[metal_upload] gDevice chua init - phai goi metal_init() truoc"); return NULL; }
        size_t bytes = (size_t)n * sizeof(float);
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

MetalBufferHandle metal_upload_borrowed(const float* data, int n) {
    @autoreleasepool {
        if (!gDevice) {
            NSLog(@"[metal_upload_borrowed] gDevice chua init");
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
        if (!gDevice) { NSLog(@"[metal_upload_indices] gDevice chua init - phai goi metal_init() truoc"); return NULL; }
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

// THÊM MỚI: xem giải thích đầy đủ ở metal_shim.h. Heap cấp phát 1 lần
// (device-level, có phí cố định như metal_alloc_scratch, nhưng CHỈ 1 lần
// thay vì lặp lại hàng nghìn lần), sau đó suballocate rẻ từ heap đó.
void* metal_alloc_heap(long bytes) {
    @autoreleasepool {
        if (!gDevice) { NSLog(@"[metal_alloc_heap] gDevice chua init - phai goi metal_init() truoc"); return NULL; }
        MTLHeapDescriptor *desc = [[MTLHeapDescriptor alloc] init];
        desc.size = (NSUInteger)bytes;
        // SUA (theo yeu cau): chon storageMode theo KIEN TRUC GPU thay vi
        // hardcode 1 mode cho ca 2 loai may.
        // - Apple Silicon (gDevice.hasUnifiedMemory == YES): CPU/GPU dung
        //   chung 1 vung RAM vat ly -> Shared HOP LE cho heap va la lua chon
        //   TOT NHAT (zero-copy that su, khong ton blit khi verify doc ve,
        //   khong ton coherency traffic vi thuc chat khong co "copy" nao ca).
        // - Intel/AMD (hasUnifiedMemory == NO, VRAM rieng): heap TU CHOI
        //   Shared ("Heap Descriptor Validation: Shared storage mode
        //   disallowed", SIGABRT thuc te da gap) -> BAT BUOC Private.
        // CHAN DOAN + AN TOAN: da co log tan gay o day (SIGABRT ngay trong
        // newHeapWithDescriptor:, truoc khi kip chay NSLog "OK") -> nghia la
        // desc.storageMode luc runtime la Shared tren may Intel iGPU, tuc
        // gDevice.hasUnifiedMemory tra ve YES SAI ("Heap Descriptor
        // Validation: Shared storage mode disallowed" chi xay ra khi ho tro
        // Shared heap KHONG duoc dam bao that su, du property co the bao
        // sai tren mot so driver Intel/AMD cu). FIX AN TOAN: KHONG chi tin
        // hasUnifiedMemory - kiem tra CHEO qua device name (Intel/AMD/
        // "Iris"/"Radeon" -> ep Private du hasUnifiedMemory bao gi).
        BOOL unified = gDevice.hasUnifiedMemory;
        NSString *devName = gDevice.name ? gDevice.name : @"";
        BOOL looksNonAppleSilicon = ([devName rangeOfString:@"Intel" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"AMD" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"Radeon" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"Iris" options:NSCaseInsensitiveSearch].location != NSNotFound)
                                  || ([devName rangeOfString:@"NVIDIA" options:NSCaseInsensitiveSearch].location != NSNotFound);
        NSLog(@"[metal_alloc_heap] gDevice.name=%@ hasUnifiedMemory=%d looksNonAppleSilicon=%d", devName, unified, looksNonAppleSilicon);
        if (looksNonAppleSilicon) unified = NO;   // ep Private du property co bao sai
        desc.storageMode = unified ? MTLStorageModeShared : MTLStorageModePrivate;
        // SUA: type = MTLHeapTypePlacement (mac dinh Automatic cung duoc, nhung
        // Placement cho phep kiem soat lifetime buffer con ro rang hon, tranh GPU
        // giu resource lau hon can thiet do heap tu dong theo doi residency).
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
        // SUA: doc storageMode THAT SU cua heap (heap.storageMode) thay vi
        // hardcode Private - suballocate PHAI khop mode voi heap cha (Metal
        // khong cho phep lech mode giua buffer va heap chua no), va gio heap
        // co the la Shared (Apple Silicon) hoac Private (Intel/AMD) tuy
        // metal_alloc_heap quyet dinh o tren.
        MTLResourceOptions opts = (heap.storageMode == MTLStorageModeShared)
            ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
        id<MTLBuffer> buf = [heap newBufferWithLength:bytes options:opts];
        if (!buf) {
            // Suballocate that bai (het cho trong heap, vd tinh sai tong kich
            // thuoc luc metal_alloc_heap) - KHONG fallback am tham, tra NULL de
            // caller (Nim ensureKvResident) biet ma xu ly (vd bao loi ro rang
            // thay vi doc/ghi vao buffer khong ton tai).
            NSLog(@"[metal_alloc_from_heap] suballocate FAILED (n=%d) - heap co the da het cho", n);
            return NULL;
        }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

void metal_free_heap(void* heapPtr) {
    @autoreleasepool {
        if (!heapPtr) return;
        id<MTLHeap> heap = (__bridge_transfer id<MTLHeap>)heapPtr;
        (void)heap;   // ARC giai phong khi ra khoi scope autoreleasepool nay
    }
}

int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_enc] chua goi metal_session_begin"); return 0; }
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
        // SUA: tile động (8/16/32) theo GPU thật, xem pickMatmulPipeline().
        NSUInteger t = (NSUInteger)tile;
        MTLSize groupSize = MTLSizeMake(t, t, 1);
        MTLSize threadgroupCount = MTLSizeMake(((NSUInteger)n + t - 1) / t, ((NSUInteger)m + t - 1) / t, 1);
        [gSessionEncoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:groupSize];
        // SỬA BUG THẬT: 1 encoder dùng chung cho nhiều dispatch (matmul rồi
        // softmax rồi matmul...) KHÔNG tự động đảm bảo write của dispatch
        // này visible cho read của dispatch SAU trên GPU không phải Apple
        // Silicon (vd Intel Iris) - Metal chỉ đảm bảo THỨ TỰ thực thi, không
        // đảm bảo memory visibility giữa các pipeline khác nhau nếu thiếu
        // barrier tường minh. Đây đúng là nguyên nhân "kết quả sai trên
        // driver Intel Iris" mà comment cũ nghi ngờ nhưng chưa fix - insert
        // memoryBarrierWithScope: ngay sau MỖI dispatch ghi vào buffer mà
        // dispatch kế tiếp sẽ đọc, để ép GPU hoàn tất + flush cache trước khi
        // bước tiếp theo (matmul->softmax->matmul, hoặc mul->mul->add...) đọc.
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

int metal_matmul_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n) {
    return metal_matmul_nt_off_enc(a, b, 0, c, m, k, n);
}

int metal_matmul_nt_off_enc(MetalBufferHandle a, MetalBufferHandle b, int bOffsetBytes,
                            MetalBufferHandle c, int m, int k, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_nt_enc] chua goi metal_session_begin"); return 0; }
        // Dung co dinh tile=8 (matmul_nt_kernel), giong het lua chon on dinh
        // cua pickMatmulPipeline() cho ban thuong - khong can do lai
        // maxTotalThreadsPerThreadgroup rieng cho kernel nay.
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
        // Xem giai thich memoryBarrierWithScope o metal_matmul_enc - can bat
        // buoc tren driver khong phai Apple Silicon (vd Intel Iris).
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

MetalBufferHandle metal_upload_bytes(const unsigned char* data, int n) {
    @autoreleasepool {
        // Không nhận kernel_src (chữ ký hàm không đổi để khớp header) - CALLER
        // (Nim: metalUploadBytes) PHẢI gọi metal_init(kernel_src) 1 lần trước
        // (dispatch_once bên trong nên gọi nhiều lần vẫn rẻ) để gDevice sẵn sàng.
        if (!gDevice) { NSLog(@"[metal_upload_bytes] gDevice chua init - phai goi metal_init() truoc"); return NULL; }
        size_t bytes = (size_t)n;
        id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if (!buf) { NSLog(@"[metal_upload_bytes] tao buffer FAILED (n=%d)", n); return NULL; }
        return (__bridge_retained MetalBufferHandle)buf;
    }
}

// ============================================================
// UPLOAD DẠNG PRIVATE (GPU-only) - dùng cho RESIDENT WEIGHT (Q4: wq/scales/
// zeros, HOẶC fp32 resident dùng cho int8/int4-symmetric/tied-embedding qua
// beResidentUploadFp32) - loại buffer chỉ upload ĐÚNG 1 LẦN lúc load model,
// sau đó CPU KHÔNG BAO GIỜ đọc/ghi lại (chỉ GPU đọc lại mỗi token). Dùng
// .private để driver không phải giữ coherency với CPU cache mỗi lần GPU đọc
// (Shared bắt buộc phải giữ). Activation buffer (upload từ CPU mỗi token,
// hoặc phải sessionRead về CPU) VẪN dùng Shared như cũ (metal_upload/
// metal_upload_bytes) - đây là đúng nghĩa "hybrid" theo yêu cầu: chỉ GPU
// đụng -> Private, có CPU đụng (upload/đọc) -> Shared.
// Private buffer không thể newBufferWithBytes trực tiếp - phải qua staging
// Shared buffer rồi blit sang Private, tốn thêm 1 lần copy CHỈ lúc load
// model (không lặp lại mỗi token).
//
// SUA (fix OOM/Killed:9 - xem báo cáo debug 2026-09-14; CHƯA build/chạy thử
// trên máy thật, môi trường sửa lỗi này KHÔNG có macOS/Metal toolchain -
// build lại + theo dõi RSS trước khi tin tưởng hoàn toàn, revert về Private
// như cũ nếu thấy bất thường): GPU low-power/tích hợp (vd Intel Iris Plus
// 640, isLowPower=YES - đúng flag đã log ở pickMetalDevice, "lowPower=1")
// KHÔNG có VRAM riêng - Private trên loại này vẫn nằm CÙNG 1 vùng RAM vật
// lý với CPU, chỉ khác ở chỗ bị kernel GHIM (wired) - KHÔNG THỂ nén/swap
// được dù swap hệ thống còn trống bao nhiêu. Cộng dồn qua 129 tham số
// quant-resident (64 block x2 + lm_head) phần wired này là nguyên nhân
// chính Jetsam kill tiến trình. Dùng isLowPower (KHÁC hasUnifiedMemory -
// property đó đã biết không đáng tin cho MTLHeap, xem comment
// metal_alloc_heap, nhưng đó là bug riêng của validation Shared-heap, không
// áp dụng cho MTLBuffer thường dùng ở đây) để CHỈ đổi hành vi trên GPU
// tích hợp; GPU rời/hiệu năng cao vẫn giữ nguyên Private như thiết kế gốc.
static MetalBufferHandle metalUploadPrivateRaw(const void* data, size_t bytes) {
    if (!gDevice) { NSLog(@"[metalUploadPrivateRaw] gDevice chua init"); return NULL; }
    id<MTLBuffer> staging = [gDevice newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    if (!staging) { NSLog(@"[metalUploadPrivateRaw] tao staging buffer FAILED (bytes=%zu)", bytes); return NULL; }
    if (gDevice.isLowPower) {
        // UMA/tich hop: staging DA LA buffer dung duoc (cung 1 vung RAM,
        // Shared, pageable) - khong tao them ban Private + blit (chi ton
        // wired memory vo ich tren loai GPU nay). Tra thang staging.
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
        NSLog(@"[metalUploadPrivateRaw] blit FAILED, error=%@ - rơi về Shared để không mất dữ liệu", cmdBuf.error);
        return (__bridge_retained MetalBufferHandle)staging;   // an toàn: đúng dữ liệu, chỉ mất phần tăng tốc
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
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_q4_enc] chua goi metal_session_begin"); return 0; }
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
        // SUA BUG NGHIEM TRONG (nguyen nhan that cua "token rac" - xem giai
        // thich day du o metal_matmul_q4() ban khong-session va trong
        // vecop_matmul.metal): KHONG con hardcode 32 - doc THAT
        // pipeline.threadExecutionWidth, dispatch threadgroup CHINH XAC
        // bang so do -> threadgroup LUON = DUNG 1 simdgroup phan cung that.
        NSUInteger simdWidthNS = pipeline.threadExecutionWidth;
        int simdWidthI = (int)simdWidthNS;
        [gSessionEncoder setBytes:&simdWidthI length:sizeof(int) atIndex:10];
        [gSessionEncoder setThreadgroupMemoryLength:MMQ4_NDST_ACTIVE * simdWidthNS * sizeof(float) atIndex:0];

        const NSUInteger kMMQ4_NDST = MMQ4_NDST_ACTIVE;  // dong bo voi MMQ4_KERNEL_NAME_ACTIVE o dau file
        NSUInteger rowBlocks = ((NSUInteger)n + kMMQ4_NDST - 1) / kMMQ4_NDST;
        MTLSize groupSz = MTLSizeMake(simdWidthNS, 1, 1);
        MTLSize gridCount = MTLSizeMake(rowBlocks, (NSUInteger)m, 1);
        [gSessionEncoder dispatchThreadgroups:gridCount threadsPerThreadgroup:groupSz];
        // SUA: xem giai thich memoryBarrierWithScope o metal_matmul_enc - can
        // cho GPU khong phai Apple Silicon (Intel Iris) de dispatch ke tiep
        // (vd sessionActivation doc lai bufC) thay dung du lieu.
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// THÊM MỚI: matmul Q4 batch M thật (đọc weight 1 lần, dùng lại cho MBLOCK=4
// hàng) - CHỈ khác metal_matmul_q4_enc ở tên kernel + cách tính grid Y (chia
// cho MBLOCK thay vì = M trực tiếp). Xem matmul_q4_batched.metal.
int metal_matmul_q4_batched_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                                 MetalBufferHandle zeros, MetalBufferHandle c,
                                 int m, int k, int n, int groupSize, int nGroupsPerRow, int wqOffset) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_matmul_q4_batched_enc] chua goi metal_session_begin"); return 0; }
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

        const NSUInteger kBATCHED_NDST = 2;    // PHẢI khớp NDST_VAL trong BB_MATMUL_Q4_BATCHED(2,4,...)
        const NSUInteger kMBLOCK = 4;          // PHẢI khớp MBLOCK_VAL trong BB_MATMUL_Q4_BATCHED(2,4,...)
        NSUInteger rowBlocksB = ((NSUInteger)n + kBATCHED_NDST - 1) / kBATCHED_NDST;
        NSUInteger mBlocks = ((NSUInteger)m + kMBLOCK - 1) / kMBLOCK;   // KHÁC bản gốc: chia cho MBLOCK
        MTLSize groupSzB = MTLSizeMake(simdWidthNS, 1, 1);
        MTLSize gridCountB = MTLSizeMake(rowBlocksB, mBlocks, 1);
        [gSessionEncoder dispatchThreadgroups:gridCountB threadsPerThreadgroup:groupSzB];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_softmax_enc] chua goi metal_session_begin"); return 0; }
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
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n) {
    return metal_activation_enc_p(op, x, y, n, 0.0f);
}

// MOI THEM (BYBY_MERGE_QKV_ATTN_SESSION): nhan scalar tai cho trong session,
// dung scale_kernel co san trong vecop_matmul.metal (truoc gio chi duoc goi
// qua duong khong-session metal_scale_c, chua co ban _enc). Dung de nhan
// attn.scale vao Q NGAY TREN GPU, thay vi doc Q ve CPU nhan roi upload lai -
// day la buoc bat buoc de gop Session #1 (QKV+RoPE) voi Session #2
// (attention+oProj+FFN) thanh 1 command buffer duy nhat cho ca layer.
// An toan: mirror y het pattern cua metal_activation_enc_p (pipeline lookup,
// dispatchThreads, memoryBarrier) - khong doi kernel .metal, chi them glue.
int metal_scale_enc(MetalBufferHandle x, float scalar, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_scale_enc] chua goi metal_session_begin"); return 0; }
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

// MỚI THÊM: bản có tham số p (dùng cho op=7 "pow" - opcode khớp bảng mã
// metalActivation immediate ở metal_backend.nim: 0=relu,1=sigmoid,2=tanh,
// 3=apflu,4=exp,5=log,6=sqrt,7=pow). Các op khác (0-6) không đọc buffer(3)
// trong activation_kernel nên truyền p=0.0 vô hại, giữ metal_activation_enc
// cũ nguyên chữ ký để KHÔNG phá vỡ call site relu/sigmoid/tanh/apflu hiện tại.
int metal_activation_enc_p(int op, MetalBufferHandle x, MetalBufferHandle y, int n, float p) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_activation_enc_p] chua goi metal_session_begin"); return 0; }
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
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

// MỚI THÊM: activation_backward_kernel (dùng cho apflu_backward trong
// session - relu/sigmoid/tanh backward hiện chưa gọi qua session nên chỉ
// wire apflu, nhưng hàm này tổng quát cho cả 4 opcode nếu sau cần).
int metal_activation_backward_enc(int op, MetalBufferHandle x, MetalBufferHandle dy, MetalBufferHandle dx, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_activation_backward_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_rope_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_vecop_enc] chua goi metal_session_begin"); return 0; }
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
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

// THÊM MỚI: dst[dstOffset..dstOffset+n) = src[0..n). dstOffset tính bằng SỐ
// PHẦN TỬ float32 - nhân sizeof(float) khi setBuffer:offset: (offset đó là
// byte offset thật của Metal). Dùng để ghép GPU-side nhiều buffer con
// thành 1 buffer liên tục trong CÙNG session (xem giải thích ở
// copy_into_kernel trong vecop_matmul.metal) - gỡ bỏ điểm chặn cuối cùng
// khiến attention-decode-session và oProj-session phải tách rời.
// LƯU Ý: offset dùng cho setBuffer PHẢI là bội số của
// device.minimumLinearTextureAlignmentForPixelFormat hoặc đơn giản hơn -
// theo tài liệu Metal, offset cho buffer THƯỜNG (không phải texture) chỉ
// cần là bội số của 4 byte trên hầu hết GPU (Apple/AMD/Intel) vì đây là
// buffer of floats (4-byte aligned tự nhiên) - không cần căn chỉnh đặc biệt
// thêm, khác texture. Nếu 1 driver cụ thể nào đó đòi hỏi alignment lớn hơn
// (hiếm, chưa gặp trong bản này), hàm sẽ trả 0 qua adjustedOffset check.
int metal_copy_into_enc(MetalBufferHandle src, MetalBufferHandle dst, int srcOffsetElems, int dstOffsetElems, int n) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_copy_into_enc] chua goi metal_session_begin"); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"copy_into_kernel");
        if (!pipeline) return 0;
        id<MTLBuffer> bufSrc = (__bridge id<MTLBuffer>)src;
        id<MTLBuffer> bufDst = (__bridge id<MTLBuffer>)dst;
        NSUInteger srcByteOffset = (NSUInteger)srcOffsetElems * sizeof(float);
        NSUInteger dstByteOffset = (NSUInteger)dstOffsetElems * sizeof(float);
        // MỚI THÊM: srcOffsetElems - dùng ĐÚNG cơ chế setBuffer:offset: mà
        // dstOffsetElems đã dùng từ trước (đã chạy đúng, không phải toán học
        // mới) - chỉ áp dụng offset đó cho buffer NGUỒN thay vì luôn cố định 0.
        // Mục đích: gộp nhiều upload nhỏ thành 1 upload lớn rồi trích từng
        // đoạn ra bằng copy_into, thay vì upload riêng từng đoạn nhỏ.
        if (srcByteOffset % 4 != 0) { NSLog(@"[metal_copy_into_enc] src offset khong chia het 4 byte"); return 0; }
        if (dstByteOffset % 4 != 0) { NSLog(@"[metal_copy_into_enc] dst offset khong chia het 4 byte"); return 0; }
        if (dstByteOffset >= bufDst.length) { NSLog(@"[metal_copy_into_enc] dst offset vuot qua kich thuoc buffer dich"); return 0; }
        if (srcByteOffset >= bufSrc.length) { NSLog(@"[metal_copy_into_enc] src offset vuot qua kich thuoc buffer nguon"); return 0; }
        [gSessionEncoder setComputePipelineState:pipeline];
        [gSessionEncoder setBuffer:bufSrc offset:srcByteOffset atIndex:0];
        [gSessionEncoder setBuffer:bufDst offset:dstByteOffset atIndex:1];
        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        [gSessionEncoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_rmsnorm_enc(MetalBufferHandle x, MetalBufferHandle gamma,
                       MetalBufferHandle y, int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_rmsnorm_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_layernorm_enc] chua goi metal_session_begin"); return 0; }
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
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_embedding_lookup_enc] chua goi metal_session_begin"); return 0; }
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
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

// Bản session của metal_attention_fused ở trên: KHÔNG tự tạo buffer/command
// buffer riêng, dùng thẳng buffer đã resident (q/k/v upload sẵn, o/s_matrix
// alloc_scratch sẵn) và gSessionEncoder đang mở - encode xong KHÔNG commit,
// để metal_session_end() commit+wait chung 1 lần với mọi op khác trong layer
// (add/layernorm/matmul/apflu). Cùng giới hạn S<=256 như bản non-session vì
// dùng chung 1 kernel attention_fused_kernel.
int metal_attention_fused_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                               MetalBufferHandle o, MetalBufferHandle s_matrix,
                               int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_fused_enc] chua goi metal_session_begin"); return 0; }
        if (S > 256) { NSLog(@"[metal_attention_fused_enc] S=%d vuot gioi han 256 cua kernel", S); return 0; }
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
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];  // SỬA: xem giải thích ở metal_matmul_enc
        return 1;
    }
}

// MOI THEM: ban session cua attention_fused_flash_kernel (xem giai thich
// thuat toan trong .metal). Khong con buffer s_matrix (index 4) nua -> B
// chuyen xuong index 4, H->5, S->6, D->7, scale->8 (lui 1 so voi ham
// metal_attention_fused_enc cu o tren). Dung cho INFERENCE (prefill) -
// KHONG dung cho training (backward can s_matrix, van phai goi
// metal_attention_fused_enc cu o tren cho duong training).
int metal_attention_fused_flash_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                     MetalBufferHandle o,
                                     int B, int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_fused_flash_enc] chua goi metal_session_begin"); return 0; }
        if (D > 128) { NSLog(@"[metal_attention_fused_flash_enc] D=%d vuot gioi han 128 cua kernel flash", D); return 0; }
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
        // SUA: kernel can THREADGROUP THAT (64 thread hop tac / hang query,
        // xem giai thich trong .metal) cho phan reduce (redBuf) va cac bien
        // scalar dung chung (sM/sL/sAOld/sP) - phai dung dispatchThreadgroups,
        // KHONG con dispatchThreads groupSize=(1,1,1) nua. LUU Y: acc[] (bo
        // dem output tich luy) DA CHUYEN VE private array/thread (thanh ghi)
        // trong .metal - KHONG con la threadgroup memory nua, nhung dieu nay
        // KHONG anh huong gi toi phia shim: acc[] khai bao ben trong than
        // kernel (khong phai tham so threadgroup con tro truyen tu host qua
        // setThreadgroupMemoryLength), nen dispatch/buffer-binding o day GIU
        // NGUYEN, khong can sua gi them.
        const NSUInteger kAttnFlashTg = 64;   // PHAI khop voi #define ATTN_FLASH_TG trong vecop_matmul.metal
        MTLSize gridSize = MTLSizeMake((NSUInteger)(B * H), (NSUInteger)S, 1);
        MTLSize groupSize = MTLSizeMake(kAttnFlashTg, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// MỚI THÊM (v2 - sửa lỗi song song): bản session của attention_decode_gqa_kernel.
// Bản đầu dispatch 1 thread/head (chỉ 4-8 thread tổng) - lãng phí gần hết
// GPU, đo thực tế chậm hơn ~4x so với đường cũ. Giờ dispatch ĐÚNG 1
// THREADGROUP (128 thread, khớp ATTN_DECODE_TG trong .metal) / query head -
// threadgroup tự chia nhau lặp S và D bên trong kernel bằng tree-reduction.
int metal_attention_decode_gqa_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                    MetalBufferHandle o, MetalBufferHandle scores,
                                    int H, int S, int D, float scale) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_decode_gqa_enc] chua goi metal_session_begin"); return 0; }
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
        // Grid = H threadgroup, MỖI threadgroup 128 thread (khớp ATTN_DECODE_TG
        // trong .metal) - tổng thread thực tế = H*128 (vd H=8 -> 1024 thread),
        // KHÔNG phải chỉ H thread như bản v1 lỗi trước đây.
        const NSUInteger TG = 128;
        MTLSize gridSize = MTLSizeMake((NSUInteger)H, 1, 1);
        MTLSize groupSize = MTLSizeMake(TG, 1, 1);
        [gSessionEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];
        [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        return 1;
    }
}

// THÊM MỚI: attention decode FUSED - TOÀN BỘ H head trong ĐÚNG 1 dispatch.
// Xem giải thích đầy đủ ở metal_shim.h và attention_decode_all_heads.metal.
// Viết mới hoàn toàn, KHÔNG dùng chung pipeline/logic với hàm GQA phía trên
// (kernel đó đã bị ghi nhận sinh sai trước đây).
int metal_attention_decode_all_heads_enc(MetalBufferHandle q, MetalBufferHandle kAll, MetalBufferHandle vAll,
                                          MetalBufferHandle o,
                                          int H, int HKV, int D, int S, int maxSeqLen) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_attention_decode_all_heads_enc] chua goi metal_session_begin"); return 0; }
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
        // Threadgroup memory "scores" (threadgroup(0) trong kernel) cấp phát
        // ĐỘNG theo S (khác Kv resident, S thay đổi MỖI TOKEN) - PHẢI gọi
        // setThreadgroupMemoryLength TRƯỚC dispatch, kích thước = S floats.
        // (shared_reduce trong kernel là mảng CỐ ĐỊNH ATTN_ALL_TG=128, khai
        // báo ngay trong kernel, KHÔNG cần host cấp phát riêng.)
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

int metal_session_end(void) {
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf) { return 0; }  // SUA: bo NSLog spam theo yeu cau - van giu logic an toan (tra ve 0 de caller biet session chua mo)
        [gSessionEncoder endEncoding];
        [gSessionCmdBuf commit];
        [gSessionCmdBuf waitUntilCompleted];
        BOOL ok = (gSessionCmdBuf.status == MTLCommandBufferStatusCompleted);
        if (!ok) {
            NSLog(@"[metal_session_end] command buffer FAILED, status=%ld, error=%@",
                  (long)gSessionCmdBuf.status, gSessionCmdBuf.error);
            // THÊM MỚI: lưu lại description THẬT để metal_get_last_session_error()
            // trả về cho Nim - trước đây thông tin này CHỈ có trong NSLog, mất
            // hẳn khi Nim chỉ nhận được "sessionOk=false" (bool trần trụi).
            NSString *statusStr = [NSString stringWithFormat:@"status=%ld", (long)gSessionCmdBuf.status];
            NSString *errStr = gSessionCmdBuf.error ? [gSessionCmdBuf.error description] : @"(cmdBuf.error la nil)";
            gLastSessionEndError = [NSString stringWithFormat:@"%@ %@", statusStr, errStr];
        } else {
            gLastSessionEndError = nil;   // lan nay THANH CONG - xoa loi cu de khong bao nham
        }
        gSessionEncoder = nil;
        gSessionCmdBuf = nil;
        return ok ? 1 : 0;
    }
}

// MOI - "commitAndContinue" (hoc theo MPSCommandBuffer cua Apple va
// SyncType::COMMIT_ADAPTIVE cua PyTorch MPS backend, xem MPSStream.mm):
// đây mới là fix ĐÚNG GỐC cho GPU Timeout, không phải chỗ vá "flush+wait"
// hay "retry sau khi loi" (2 huong da thu truoc, 1 huong qua cham vi
// waitUntilCompleted moi lan la 1 lan CPU<->GPU sync THAT trên iGPU cu, 1
// huong khong ngan duoc loi xay ra tu dau).
//
// Y tuong: 1 MTLCommandQueue thuc thi cac command buffer THEO DUNG THU TU
// commit (FIFO), TU DONG, khong can CPU cho tung buffer xong roi moi encode
// buffer tiep theo. Vay nen thay vi "1 buffer khong lo cho ca 64 layer" (dut
// watchdog vi qua lau) hay "cho THAT (waitUntilCompleted) sau moi vai layer"
// (cham vi wake/sleep GPU tung lan), ta CAM KET (commit) buffer hien tai
// KHONG CHO, roi mo NGAY buffer moi de tiep tuc encode - GPU tu chay ngam
// buffer vua commit trong luc CPU dang encode buffer ke tiep (giong hang
// doi lenh in), tuong tu commitAndContinue() cua MPSCommandBuffer.
//
// Ket qua: (1) MOI buffer nho, bi chan boi watchdog do 1 buffer gay ra la
// RAT thap (khong con "64 layer don 1 cuc"); (2) KHONG tra gia
// CPU<->GPU sync (khong waitUntilCompleted giua chung) nen khong bi cham
// nhu lan "flush" truoc; (3) CHI 1 lan waitUntilCompleted THAT SU - luc
// metal_session_end() cuoi cung, khi can doc logits ve CPU (day la diem
// dong bo BAT BUOC vi phai co logit moi chon duoc token tiep theo).
//
// Neu buffer da commit (khong cho) sau nay LOI that (vi du van watchdog du
// da chia nho) - vi tat ca buffer chung 1 gQueue, loi do se lam "poison"
// queue va CASCADE xuong cac buffer sau, cuoi cung metal_session_end() se
// tra ve 0 (vi buffer CUOI CUNG cung bi "Ignored" domino) - luc do code Nim
// (sessionRecoverQueue + retry, da co san) van la luoi an toan cuoi cung.
int metal_session_commit_continue(void) {
    @autoreleasepool {
        if (!gSessionEncoder || !gSessionCmdBuf) {
            NSLog(@"[metal_session_commit_continue] chua co session dang mo");
            return 0;
        }
        [gSessionEncoder endEncoding];
        // Log KHONG CHAN (async) neu buffer nay cuoi cung loi - chi de debug,
        // khong anh huong toc do vi khong doi handler chay.
        [gSessionCmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buf) {
            if (buf.status != MTLCommandBufferStatusCompleted) {
                NSLog(@"[metal_session_commit_continue] 1 buffer giua chung loi (se cascade xuong buffer sau): status=%ld, error=%@",
                      (long)buf.status, buf.error);
            }
        }];
        [gSessionCmdBuf commit];               // KHONG waitUntilCompleted - day la diem khac biet mau chot
        gSessionCmdBuf = [gQueue commandBuffer];
        if (!gSessionCmdBuf) { NSLog(@"[metal_session_commit_continue] commandBuffer moi tra ve nil"); gSessionEncoder = nil; return 0; }
        gSessionEncoder = [gSessionCmdBuf computeCommandEncoder];
        if (!gSessionEncoder) { NSLog(@"[metal_session_commit_continue] computeCommandEncoder moi tra ve nil"); gSessionCmdBuf = nil; return 0; }
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
        // SUA (crash tiem an): [buf contents] CHI hop le voi buffer Shared/
        // Managed - tra ve nil (hoac invalid) voi buffer Private, memcpy tu
        // do se crash hoac doc rac. Cac buffer suballocate tu heap resident
        // KV (xem metal_alloc_heap) gio la Private (fix crash "Shared
        // storage mode disallowed" tren Intel) -> verifyKvResidentOnce doc
        // ve qua ham nay CAN duong Private. Fix: neu Private, blit qua 1
        // staging buffer Shared tam thoi (blit command encoder rieng, doc
        // xong huy ngay), roi memcpy tu staging - cham hon 1 chut (co
        // command buffer + waitUntilCompleted rieng) nhung ĐÚNG, va con
        // duong nay chi chay khi debug/verify (khong phai hot path suy
        // luan moi token).
        if (buf.storageMode == MTLStorageModePrivate) {
            if (!gQueue) { NSLog(@"[metal_buffer_read] gQueue chua init"); return 0; }
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
        // __bridge_transfer trả quyền sở hữu ARC lại cho biến id<MTLBuffer>
        // tạm rồi để nó tự release khi ra khỏi scope - đúng cặp với
        // __bridge_retained lúc metal_upload/metal_alloc_scratch tạo ra handle.
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
        if (S > 256) { NSLog(@"[metal_attention_fused_backward] S=%d vuot gioi han 256 cua kernel", S); return 0; }
        id<MTLComputePipelineState> pipeline = getPipeline(@"attention_fused_backward_kernel");
        if (!pipeline) return 0;
        size_t bytesQKV = (size_t)B * (size_t)H * (size_t)S * (size_t)D * sizeof(float);
        size_t bytesS = (size_t)B * (size_t)H * (size_t)S * (size_t)S * sizeof(float);
        id<MTLBuffer> bufQ = [gDevice newBufferWithBytes:q length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [gDevice newBufferWithBytes:k length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [gDevice newBufferWithBytes:v length:bytesQKV options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [gDevice newBufferWithBytes:s_matrix length:bytesS options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDy = [gDevice newBufferWithBytes:dy length:bytesQKV options:MTLResourceStorageModeShared];
        // dq/dk/dv: kernel dùng "+=" (dq) / atomic-add (dk, dv) -> PHẢI
        // zero-init buffer trước khi launch (khác bufO/bufS ở forward, ghi
        // đè hoàn toàn nên không cần). newBufferWithLength không đảm bảo
        // zero theo spec Metal -> memset tường minh cho chắc.
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
// GPU HOÁ THẬT (vòng 2): gather / argmax(giá trị+chỉ số) / compare(bitmask)
// / dropout RNG - kernel source đã nối vào cuối vecop_matmul.metal, nên
// kernel_src truyền vào đây là kMetalKernelSrc đã gộp sẵn (không compile
// library riêng) - cùng cơ chế cache pipeline theo tên hàm (getPipeline)
// như metal_vecop ở trên.
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
        // threadgroup memory: sVal (float) + sIdx (int) mỗi cái dimSize phần tử,
        // khớp buffer(0)/buffer(1) [[threadgroup]] khai báo trong argmax_kernel.
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
// GPU HOÁ THẬT (vòng 3) - GIỐNG TORCH: permute/split/concat, device tự
// giải mã chỉ số từ bảng nhỏ O(rank)/O(numTensors) truyền qua setBytes,
// không có mảng chỉ số O(n) build trên host.
// ─────────────────────────────────────────────────────────────

int metal_permute(const char* kernel_src, const float* src, float* dst,
                   const int* newStrides, const int* oldStrides,
                   const int* dimsMap, int ndim, int n, char** error_out) {
    if (error_out) *error_out = NULL;
    @autoreleasepool {
        if (!ensureMetalInit(kernel_src)) return 0;
        id<MTLComputePipelineState> pipeline = getPipeline(@"permute_strided");
        if (!pipeline) return 0;

        size_t srcBytes = (size_t)n * sizeof(float); // n = numel, đủ cho src cùng kích thước
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
// MỚI THÊM: các hàm _enc còn lại cho session - bám sát 100% idiom của
// metal_gather/metal_argmax/metal_compare/metal_dropout_mask/metal_permute/
// metal_split_slice/metal_concat_gather/metal_layernorm_backward/
// metal_attention_fused_backward ở trên (cùng pipeline, cùng buffer index),
// chỉ khác: buffer đã resident (không newBufferWithBytes từ host), không
// commit/wait riêng (dùng gSessionEncoder đang mở, memoryBarrier thay vì
// endEncoding/commit/wait). CHƯA CHẠY THỬ TRÊN GPU METAL THẬT - bắt buộc
// build + test trên máy Mac thật trước khi dùng cho train thật.
// ─────────────────────────────────────────────────────────────────────────

int metal_layernorm_backward_enc(MetalBufferHandle dy, MetalBufferHandle x,
                                  MetalBufferHandle gamma, MetalBufferHandle beta,
                                  MetalBufferHandle dx, MetalBufferHandle dgamma, MetalBufferHandle dbeta,
                                  int rows, int cols, float eps) {
    @autoreleasepool {
        if (!gSessionEncoder) { NSLog(@"[metal_layernorm_backward_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_attention_fused_backward_enc] chua goi metal_session_begin"); return 0; }
        if (S > 256) { NSLog(@"[metal_attention_fused_backward_enc] S=%d vuot gioi han 256", S); return 0; }
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
        // dq/dk/dv dùng "+=" / atomic-add trong kernel -> PHẢI zero-init
        // trước (khác forward ghi đè hoàn toàn) - caller (gpubackend.nim)
        // cấp buffer bằng metalSessionAllocScratch rồi memset qua blit
        // encoder KHÔNG có sẵn ở đây, nên memset trực tiếp qua contents
        // (buffer là MTLResourceStorageModeShared trên hầu hết máy Intel/
        // Apple Silicon iGPU nên contents truy cập được từ CPU an toàn
        // NGAY CẢ khi command buffer session đang mở, vì chưa commit).
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
        if (!gSessionEncoder) { NSLog(@"[metal_gather_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_argmax_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_compare_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_dropout_mask_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_permute_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_split_slice_enc] chua goi metal_session_begin"); return 0; }
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
        if (!gSessionEncoder) { NSLog(@"[metal_concat_gather_enc] chua goi metal_session_begin"); return 0; }
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
 * Session DƯỚI CHÂN — implement
 * Dán TOÀN BỘ file này vào CUỐI metal_shim.m (sau hàm cuối cùng).
 * Phụ thuộc: gSessionEncoder, gLibrary, gDevice, getPipeline() đã có.
 * ============================================================ */

static id<MTLComputePipelineState> gEncCurrentPipeline = nil;

/* Cache pipeline tu metal_jit_compile (bytecode -> Metal source). */
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* gJitPipelines = nil;

int metal_enc_pipeline(const char* kernelName) {
    @autoreleasepool {
        if (!gSessionEncoder) {
            NSLog(@"[metal_enc_pipeline] chua session_begin");
            return 0;
        }
        if (!kernelName) return 0;
        NSString *name = [NSString stringWithUTF8String:kernelName];
        id<MTLComputePipelineState> p = nil;
        if (gJitPipelines) p = gJitPipelines[name];
        if (!p) p = getPipeline(name);
        if (!p) {
            NSLog(@"[metal_enc_pipeline] khong tim thay kernel: %@", name);
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
            NSLog(@"[metal_enc_dispatch] can pipeline truoc (metal_enc_pipeline)");
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
        /* Metal yêu cầu setThreadgroupMemoryLength trước dispatch khi kernel
         * khai báo threadgroup(N). nBytes=4 đủ cho kernel không dùng TG thật. */
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
    // MOI: theo tai lieu Apple + thuc te cong dong (Ryujinx/MoltenVK
    // "resumeLostDevice", MLX issue #3390 xac nhan hien tuong nay): SAU KHI
    // 1 MTLCommandBuffer THAT SU loi (Timeout/StackOverflow/PageFault...),
    // MTLCommandQueue chua no bi "poisoned" - MOI commandBuffer tiep theo
    // tren CUNG queue do se bi driver tu dong tu choi voi loi
    // "Ignored (for causing prior/excessive GPU errors)
    // (kIOAccelCommandBufferCallbackErrorSubmissionsIgnored)", VINH VIEN,
    // cho den khi tao MTLCommandQueue MOI. Ham nay tao lai gQueue tu gDevice
    // (KHONG tao lai gDevice - device that su mat (DeviceRemoved) la truong
    // hop khac, hiem, khong xu ly o day).
    if (!gDevice) { NSLog(@"[metal_recover] gDevice nil, khong the phuc hoi"); return 0; }
    id<MTLCommandQueue> newQueue = [gDevice newCommandQueue];
    if (!newQueue) { NSLog(@"[metal_recover] newCommandQueue that bai"); return 0; }
    gQueue = newQueue;
    NSLog(@"[metal_recover] Da tao MTLCommandQueue MOI (queue cu bi poisoned sau loi GPU truoc do)");
    return 1;
}

/* ---- MTLBinaryArchive: giu pipeline JIT da compile giua cac lan chay ----
   Moi kernel micro-VM moi ton ~0.2-0.5 s de dich (newLibraryWithSource +
   newComputePipelineStateWithDescriptor). Prefill mot prompt ngan dung ~20
   kernel, tuc ~4 s trong 12 s wall - phai tra lai tu dau o MOI tien trinh.
   Vulkan da co shader cache tren dia; day la ban tuong duong cua Metal. */
static id<MTLBinaryArchive> gJitArchive = nil;
static NSString* gJitArchivePath = nil;
static dispatch_once_t gJitArchiveOnce;

static void metal_jit_archive_save(void) {
    if (!gJitArchive || !gJitArchivePath) return;
    static dispatch_once_t warnOnce;
    @autoreleasepool {
        NSError* err = nil;
        if ([gJitArchive serializeToURL:[NSURL fileURLWithPath:gJitArchivePath] error:&err]) {
            NSLog(@"[metal_jit] archive saved %@", gJitArchivePath);
        } else {
            /* serializeToURL la Apple-silicon only; tren Intel no tra NO + nil
               error. Chi canh bao mot lan de khoi spam moi lan chay. */
            dispatch_once(&warnOnce, ^{
                NSLog(@"[metal_jit] archive serialize khong ho tro tren may nay (%@) - chi cache trong tien trinh", err ?: @"Intel");
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
                NSLog(@"[metal_jit] khong tao duoc thu muc archive %@ (%@) - chay khong cache", parent, dirErr);
                return;
            }
            MTLBinaryArchiveDescriptor* ad = [MTLBinaryArchiveDescriptor new];
            BOOL loaded = [[NSFileManager defaultManager] fileExistsAtPath:path];
            if (loaded) ad.url = [NSURL fileURLWithPath:path];
            NSError* err = nil;
            gJitArchive = [gDevice newBinaryArchiveWithDescriptor:ad error:&err];
            if (!gJitArchive) {
                NSLog(@"[metal_jit] archive khong mo duoc (%@) - chay khong cache", err);
                return;
            }
            gJitArchivePath = path;
            atexit(metal_jit_archive_save);
            NSLog(@"[metal_jit] archive %@ (%@)", loaded ? @"loaded" : @"new", path);
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
        desc.maxCallStackDepth = 32; /* tránh Stack Overflow Exception (Code=12) */
        if (gJitArchive) {
            NSError* archiveErr = nil;
            if ([gJitArchive addComputePipelineFunctionsWithDescriptor:desc error:&archiveErr]) {
                desc.binaryArchives = @[gJitArchive];
            } else {
                NSLog(@"[metal_jit] khong them duoc pipeline %@ vao archive (%@)", name, archiveErr);
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
        NSLog(@"[metal_jit] OK kernel=%@", name);
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
