// metal_shim.m - Metal compute backend implementation (macOS only).
// Biên dịch Metal Shading Language từ chuỗi nguồn ngay lúc chạy (newLibraryWithSource),
// không cần build-time .metallib -> chạy trực tiếp trên bất kỳ Mac nào hỗ trợ Metal.
// Nguồn MSL không còn hardcode ở đây nữa: nó nằm ở kernels/vecop_matmul.metal
// (file .metal thật) và được Nim đọc lúc compile-time (staticRead) rồi truyền
// xuống qua tham số kernel_src.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "metal_shim.h"

// MTLCreateSystemDefaultDevice() chỉ trả về ĐÚNG MỘT GPU "mặc định" theo lựa
// chọn của hệ thống. Trên Mac Apple Silicon chỉ có 1 GPU nên luôn ổn, nhưng
// trên Mac Intel máy có thể có NHIỀU GPU (iGPU Intel + dGPU AMD, hoặc thêm
// eGPU rời) -- default device đôi khi không phải GPU thực sự đang "sống"
// (vd. dGPU đang ngủ do auto graphics-switching, hoặc app không có quyền
// power-on dGPU). Dùng MTLCopyAllDevices() (chỉ có trên macOS, không có trên
// iOS) để liệt kê toàn bộ GPU và tự chọn/thử lần lượt thay vì tin vào 1 default.
static id<MTLDevice> pickMetalDevice(void) {
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

int metal_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = pickMetalDevice();
        return device != nil ? 1 : 0;
    }
}

int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n) {
    @autoreleasepool {
        id<MTLDevice> device = pickMetalDevice();
        if (!device) { NSLog(@"[metal_vecop] khong tim thay GPU Metal nao (MTLCopyAllDevices rong)"); return 0; }
        NSLog(@"[metal_vecop] device = %@ (n=%d, op=%d)", device.name, n, op);

        NSString *source = [NSString stringWithUTF8String:kernel_src];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) { NSLog(@"[metal_vecop] newLibraryWithSource FAILED: %@", error); return 0; }

        NSString *fname = nil;
        switch (op) {
            case 0: fname = @"vecop_add"; break;
            case 1: fname = @"vecop_sub"; break;
            case 2: fname = @"vecop_mul"; break;
            case 3: fname = @"vecop_div"; break;
            default: fname = @"vecop_add"; break;
        }
        id<MTLFunction> fn = [library newFunctionWithName:fname];
        if (!fn) { NSLog(@"[metal_vecop] newFunctionWithName:%@ tra ve nil", fname); return 0; }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) { NSLog(@"[metal_vecop] newComputePipelineStateWithFunction FAILED: %@", error); return 0; }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { NSLog(@"[metal_vecop] newCommandQueue tra ve nil"); return 0; }
        size_t bytes = (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) { NSLog(@"[metal_vecop] tao buffer FAILED (bytes=%zu, n=%d)", bytes, n); return 0; }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];

        NSUInteger threadsPerGroup = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
        if (threadsPerGroup == 0) threadsPerGroup = 1;
        MTLSize gridSize = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_vecop] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c, [bufC contents], bytes);
        return 1;
    }
}

// Matmul naive (grid 2 chiều). Trên Apple Silicon, muốn tận dụng khối ma trận
// phần cứng (tương đương Tensor Core) thì thay lời gọi encoder thủ công này
// bằng MPSMatrixMultiplication (framework MetalPerformanceShaders) với
// MPSMatrixDescriptor cho A/B/C -> MPS tự chọn kernel tối ưu theo GPU đang chạy.
// Giữ bản naive ở đây để không phụ thuộc thêm framework khi chỉ cần đúng kết quả.
int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n) {
    @autoreleasepool {
        id<MTLDevice> device = pickMetalDevice();
        if (!device) { NSLog(@"[metal_matmul] khong tim thay GPU Metal nao"); return 0; }

        NSString *source = [NSString stringWithUTF8String:kernel_src];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) { NSLog(@"[metal_matmul] newLibraryWithSource FAILED: %@", error); return 0; }

        id<MTLFunction> fn = [library newFunctionWithName:@"matmul_naive"];
        if (!fn) { NSLog(@"[metal_matmul] newFunctionWithName:matmul_naive tra ve nil"); return 0; }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) { NSLog(@"[metal_matmul] newComputePipelineStateWithFunction FAILED: %@", error); return 0; }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        size_t bytesA = (size_t)m * (size_t)k * sizeof(float);
        size_t bytesB = (size_t)k * (size_t)n * sizeof(float);
        size_t bytesC = (size_t)m * (size_t)n * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithBytes:a length:bytesA options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b length:bytesB options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufC) { NSLog(@"[metal_matmul] tao buffer FAILED"); return 0; }

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&m length:sizeof(int) atIndex:3];
        [encoder setBytes:&k length:sizeof(int) atIndex:4];
        [encoder setBytes:&n length:sizeof(int) atIndex:5];

        MTLSize gridSize = MTLSizeMake((NSUInteger)n, (NSUInteger)m, 1);
        NSUInteger tw = MIN((NSUInteger)16, (NSUInteger)n);
        NSUInteger th = MIN((NSUInteger)16, (NSUInteger)m);
        if (tw == 0) tw = 1;
        if (th == 0) th = 1;
        MTLSize groupSize = MTLSizeMake(tw, th, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            NSLog(@"[metal_matmul] command buffer FAILED, status=%ld, error=%@", (long)cmdBuf.status, cmdBuf.error);
            return 0;
        }

        memcpy(c, [bufC contents], bytesC);
        return 1;
    }
}
