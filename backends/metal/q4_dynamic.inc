// Included after session state/weight upload declarations in metal_shim.m.
@interface BBQ4Plan : NSObject
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic) NSUInteger threads, rows, batch;
@property(nonatomic) BOOL legacy, simd, mm;
@property(nonatomic) NSUInteger sgs, simdRows;      // simd plans: simdgroups per threadgroup, rows per simdgroup
@property(nonatomic) uint32_t packs, load, pre;     // simd plans: weight read variant (q4v_* in the kernel)
@property(nonatomic, copy) NSString *label;
@end
@implementation BBQ4Plan
@end

static NSMutableDictionary<NSString *, BBQ4Plan *> *gQ4Plans;
static NSMutableDictionary<NSString *, BBQ4Plan *> *gQ4Kernels;
static BOOL gQ4Tuning = NO;

static NSString *q4PlanKey(int k, int n, int group, int groups, int m) {
    return [NSString stringWithFormat:@"%d_%d_g%d_%d_m%d", k, n, group, groups, m == 1 ? 1 : 8];
}

static BOOL q4TunedEnabled(void) {
    const char *v = getenv("BYBY_Q4_TUNED");
    return !v || strcmp(v, "0") != 0;
}

static BBQ4Plan *q4MakePlan(NSUInteger tg, NSUInteger rows, NSUInteger batch) {
    if (!gQ4Kernels) gQ4Kernels = [NSMutableDictionary new];
    NSString *key = [NSString stringWithFormat:@"%lu_%lu_%lu", (unsigned long)tg,
        (unsigned long)rows, (unsigned long)batch];
    if (gQ4Kernels[key]) return gQ4Kernels[key];
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    uint32_t t = (uint32_t)tg, r = (uint32_t)rows, b = (uint32_t)batch;
    [constants setConstantValue:&t type:MTLDataTypeUInt atIndex:100];
    [constants setConstantValue:&r type:MTLDataTypeUInt atIndex:101];
    [constants setConstantValue:&b type:MTLDataTypeUInt atIndex:102];
    NSError *error = nil;
    id<MTLFunction> fn = [gLibrary newFunctionWithName:@"matmul_q4_dynamic"
        constantValues:constants error:&error];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!pso || tg > pso.maxTotalThreadsPerThreadgroup ||
        pso.staticThreadgroupMemoryLength + tg * rows * batch * sizeof(float) >
            gDevice.maxThreadgroupMemoryLength) return nil;
    BBQ4Plan *plan = [BBQ4Plan new];
    plan.pipeline = pso; plan.threads = tg; plan.rows = rows; plan.batch = batch;
    plan.label = [NSString stringWithFormat:@"wide%lux%lu-tg%lu",
        (unsigned long)rows, (unsigned long)batch, (unsigned long)tg];
    gQ4Kernels[key] = plan;
    return plan;
}

// simdgroup_matrix Q4 GEMM (kernels/gemm_q4_sg.metal) for M > 1. The source is
// registered from Nim and compiled once into its own library, only on an Apple
// GPU (BYBY_Q4_SG_COMPILE_CHECK=1 compiles it anywhere to catch syntax errors).
static const char *gQ4SgSource = NULL;
void metal_register_q4_sg_source(const char *source) { gQ4SgSource = source; }

static BBQ4Plan *q4MakeMmPlan(void) {
    static int state = 0;   // 0 untried, 1 ready, -1 unavailable
    static BBQ4Plan *plan = nil;
    if (state != 0) return state > 0 ? plan : nil;
    state = -1;
    if (!gQ4SgSource || !gDevice || getenv("BYBY_Q4_NO_SG")) return nil;
    BOOL apple = NO;
    if (@available(macOS 10.15, *)) apple = [gDevice supportsFamily:MTLGPUFamilyApple7];
    const BOOL compileOnly = !apple && getenv("BYBY_Q4_SG_COMPILE_CHECK") != NULL;
    if (!apple && !compileOnly) return nil;
    NSError *error = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:[NSString stringWithUTF8String:gQ4SgSource]
        options:nil error:&error];
    if (!lib) { NSLog(@"[q4-sg] library compile failed: %@", error); return nil; }
    if (compileOnly) { NSLog(@"[q4-sg] library compiled (compile check only)"); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_q4_mm_sg"];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!pso || pso.maxTotalThreadsPerThreadgroup < 128) return nil;
    plan = [BBQ4Plan new];
    plan.pipeline = pso; plan.threads = 128; plan.rows = 32; plan.batch = 32; plan.mm = YES;
    plan.label = @"mm-sg32x32";
    state = 1;
    return plan;
}

// Configurable-tile variant of the prefill GEMM (matmul_q4_mm_sg2): bn output columns and
// bk k-values per step.
static BBQ4Plan *q4MakeMm2Plan(uint32_t bn, uint32_t bk) {
    static id<MTLLibrary> lib = nil;
    static BOOL tried = NO;
    static NSMutableDictionary<NSString *, BBQ4Plan *> *cache = nil;
    if (!tried) {
        tried = YES;
        BOOL apple = NO;
        if (@available(macOS 10.15, *)) apple = gDevice && [gDevice supportsFamily:MTLGPUFamilyApple7];
        if (gQ4SgSource && apple && !getenv("BYBY_Q4_NO_SG") && !getenv("BYBY_Q4_NO_MM2")) {
            NSError *error = nil;
            lib = [gDevice newLibraryWithSource:[NSString stringWithUTF8String:gQ4SgSource]
                options:nil error:&error];
            if (!lib) NSLog(@"[q4-sg] library compile failed: %@", error);
        }
        cache = [NSMutableDictionary new];
        if (getenv("BYBY_Q4_TUNE_LOG"))
            NSLog(@"[q4-sg] mm2 library %s (source %s, apple %d)", lib ? "ready" : "missing",
                  gQ4SgSource ? "set" : "NULL", (int)apple);
    }
    if (!lib) return nil;
    NSString *key = [NSString stringWithFormat:@"mm2_%u_%u", bn, bk];
    if (cache[key]) return cache[key] == (id)[NSNull null] ? nil : cache[key];
    cache[key] = (BBQ4Plan *)[NSNull null];
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&bn type:MTLDataTypeUInt atIndex:130];
    [constants setConstantValue:&bk type:MTLDataTypeUInt atIndex:131];
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_q4_mm_sg2" constantValues:constants error:&error];
    if (!fn) {
        if (getenv("BYBY_Q4_TUNE_LOG")) NSLog(@"[q4-sg] mm2 %ux%u function failed: %@", bn, bk, error);
        return nil;
    }
    id<MTLComputePipelineState> pso = [gDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!pso || pso.maxTotalThreadsPerThreadgroup < bn * 4) {
        if (getenv("BYBY_Q4_TUNE_LOG")) NSLog(@"[q4-sg] mm2 %ux%u pipeline failed: %@", bn, bk, error);
        return nil;
    }
    BBQ4Plan *plan = [BBQ4Plan new];
    plan.pipeline = pso; plan.threads = bn * 4; plan.rows = bn; plan.batch = 32; plan.mm = YES;
    plan.label = [NSString stringWithFormat:@"mm-sg32x%u-k%u", bn, bk];
    cache[key] = plan;
    return plan;
}

// One simdgroup per `rows` output rows, `sgs` simdgroups per threadgroup (M=1);
// packs/load/pre pick how a thread reads its weights (see q4v_* in the kernel).
static BBQ4Plan *q4MakeSimdPlan(NSUInteger sgs, NSUInteger rows, uint32_t packs,
        uint32_t load, uint32_t pre) {
    if (!gQ4Kernels) gQ4Kernels = [NSMutableDictionary new];
    NSString *key = [NSString stringWithFormat:@"simd_%lu_%lu_%u_%u_%u", (unsigned long)sgs,
        (unsigned long)rows, packs, load, pre];
    if (gQ4Kernels[key]) return gQ4Kernels[key];
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    uint32_t r = (uint32_t)rows, s = (uint32_t)sgs;
    [constants setConstantValue:&r type:MTLDataTypeUInt atIndex:110];
    [constants setConstantValue:&s type:MTLDataTypeUInt atIndex:111];
    [constants setConstantValue:&packs type:MTLDataTypeUInt atIndex:112];
    [constants setConstantValue:&load type:MTLDataTypeUInt atIndex:113];
    [constants setConstantValue:&pre type:MTLDataTypeUInt atIndex:114];
    NSError *error = nil;
    id<MTLFunction> fn = [gLibrary newFunctionWithName:@"matmul_q4_simd"
        constantValues:constants error:&error];
    if (!fn) return nil;
    id<MTLComputePipelineState> pso = [gDevice newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) return nil;
    NSUInteger threads = sgs * pso.threadExecutionWidth;
    if (threads > pso.maxTotalThreadsPerThreadgroup) return nil;
    BBQ4Plan *plan = [BBQ4Plan new];
    plan.pipeline = pso; plan.threads = threads; plan.rows = sgs * rows; plan.batch = 1;
    plan.simd = YES; plan.sgs = sgs; plan.simdRows = rows;
    plan.packs = packs; plan.load = load; plan.pre = pre;
    plan.label = (packs == 1 && load == 0 && pre == 0) ?
        [NSString stringWithFormat:@"simd%lux%lu", (unsigned long)sgs, (unsigned long)rows] :
        [NSString stringWithFormat:@"simd%lux%lu-p%u-l%u-s%u", (unsigned long)sgs,
            (unsigned long)rows, packs, load, pre];
    gQ4Kernels[key] = plan;
    return plan;
}

static int q4EncodePlan(BBQ4Plan *plan, MetalBufferHandle a, MetalBufferHandle w,
        MetalBufferHandle scale, MetalBufferHandle zero, MetalBufferHandle out,
        int m, int k, int n, int group, int groups, int wqOffset) {
    if (!gSessionEncoder || !plan || plan.legacy) return 0;
    // The kernel loads aligned words: bind at the word below wqOffset and
    // let it shift the remaining 0-3 bytes in.
    uint32_t wqShift = (uint32_t)(wqOffset & 3);
    [gSessionEncoder setComputePipelineState:plan.pipeline];
    if (!plan.simd && !plan.mm)
        [gSessionEncoder setThreadgroupMemoryLength:plan.threads * plan.rows * plan.batch * sizeof(float) atIndex:0];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)w offset:(NSUInteger)(wqOffset & ~3) atIndex:1];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:2];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)zero offset:0 atIndex:3];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:4];
    [gSessionEncoder setBytes:&m length:4 atIndex:5];
    [gSessionEncoder setBytes:&k length:4 atIndex:6];
    [gSessionEncoder setBytes:&n length:4 atIndex:7];
    [gSessionEncoder setBytes:&group length:4 atIndex:8];
    [gSessionEncoder setBytes:&groups length:4 atIndex:9];
    [gSessionEncoder setBytes:&wqShift length:4 atIndex:10];
    [gSessionEncoder dispatchThreadgroups:MTLSizeMake((n + plan.rows - 1) / plan.rows,
        (m + plan.batch - 1) / plan.batch, 1)
        threadsPerThreadgroup:MTLSizeMake(plan.threads, 1, 1)];
    sessionBarrier();
    return 1;
}

static int q4EncodeSelected(MetalBufferHandle a, MetalBufferHandle w,
        MetalBufferHandle scale, MetalBufferHandle zero, MetalBufferHandle out,
        int m, int k, int n, int group, int groups, int wqOffset) {
    if (gQ4Tuning || !q4TunedEnabled()) return 0;
    BBQ4Plan *plan = gQ4Plans[q4PlanKey(k, n, group, groups, m)];
    return q4EncodePlan(plan, a, w, scale, zero, out, m, k, n, group, groups, wqOffset);
}

int metal_prepare_q4_tuned(MetalBufferHandle w, MetalBufferHandle scale,
        MetalBufferHandle zero, int k, int n, int group, int groups, int mode, int wqOffset) {
    @autoreleasepool {
        const char *autoTune = getenv("BYBY_GPU_AUTOTUNE");
        if (!q4TunedEnabled() || (autoTune && strcmp(autoTune, "0") == 0)) return 0;
        if (!gDevice || !gLibrary || gSessionCmdBuf || k <= 0 || k % 8 || n <= 0 ||
            group < 0 || (group > 0 && group % 8) || groups <= 0 || wqOffset < 0 ||
            groups < (group > 0 ? (k + group - 1) / group : 1)) return 0;
        id<MTLBuffer> weight = (__bridge id<MTLBuffer>)w;
        id<MTLBuffer> scales = (__bridge id<MTLBuffer>)scale;
        id<MTLBuffer> zeros = (__bridge id<MTLBuffer>)zero;
        if (weight.length < (NSUInteger)wqOffset + (NSUInteger)n * (k / 2) ||
            scales.length < (NSUInteger)n * groups * 4 ||
            zeros.length < (NSUInteger)n * groups * 4) return 0;
        if (!gQ4Plans) gQ4Plans = [NSMutableDictionary new];
        const int m = mode == 1 ? 1 : 32;
        NSString *key = q4PlanKey(k, n, group, groups, m);
        if (gQ4Plans[key]) return 1;
        BBQ4Plan *baseline = [BBQ4Plan new];
        baseline.legacy = YES; baseline.label = @"legacy";
        NSMutableArray<BBQ4Plan *> *plans = [NSMutableArray arrayWithObject:baseline];
        for (NSUInteger tg = 32; tg <= 128; tg *= 2) {
            BBQ4Plan *p = q4MakePlan(tg, m == 1 ? 4 : 1, m);
            if (p) [plans addObject:p];
        }
        if (m == 1 && !getenv("BYBY_Q4_NO_SIMD")) {
            static const NSUInteger cfg[][2] = {{1, 4}, {2, 2}, {2, 4}, {4, 2}, {4, 4}, {8, 2}};
            for (NSUInteger c = 0; c < sizeof(cfg) / sizeof(cfg[0]); ++c) {
                BBQ4Plan *p = q4MakeSimdPlan(cfg[c][0], cfg[c][1], 1, 0, 0);
                if (p) [plans addObject:p];
            }
        }
        if (m > 1 && group > 0 && group % 32 == 0) {
            BBQ4Plan *p = q4MakeMmPlan();
            if (p) [plans addObject:p];
            for (uint32_t bn = 32; bn <= 64; bn *= 2)
                for (uint32_t bk = 32; bk <= 64; bk *= 2) {
                    if (group % bk) continue;
                    BBQ4Plan *q = q4MakeMm2Plan(bn, bk);
                    if (q) [plans addObject:q];
                }
        }
        if (m > 1) {
            for (NSUInteger tg = 32; tg <= 64; tg *= 2) {
                BBQ4Plan *p = q4MakePlan(tg, 2, 4);
                if (p) [plans addObject:p];
            }
        }
        if (plans.count == 1) { gQ4Plans[key] = baseline; return 1; }
        id<MTLBuffer> input = [gDevice newBufferWithLength:(NSUInteger)m * k * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [gDevice newBufferWithLength:(NSUInteger)m * n * 4 options:MTLResourceStorageModeShared];
        if (!input || !output) return 0;
        for (NSUInteger i = 0; i < (NSUInteger)m * k; ++i)
            ((float*)input.contents)[i] = ((int)((i * 17) % 101) - 50) / 64.0f;
        MetalBufferHandle hIn = (__bridge MetalBufferHandle)input;
        MetalBufferHandle hOut = (__bridge MetalBufferHandle)output;
        gQ4Tuning = YES;
        @try {
            double (^run)(BBQ4Plan *) = ^double(BBQ4Plan *p) {
                const double start = [NSProcessInfo processInfo].systemUptime;
                if (!metal_session_begin(NULL, NULL)) return INFINITY;
                int encoded = p.legacy ? (m == 1 ?
                    metal_matmul_q4_enc(hIn, w, scale, zero, hOut, m, k, n, group, groups, wqOffset) :
                    metal_matmul_q4_batched_enc(hIn, w, scale, zero, hOut, m, k, n, group, groups, wqOffset)) :
                    q4EncodePlan(p, hIn, w, scale, zero, hOut, m, k, n, group, groups, wqOffset);
                const double gpu0 = metal_session_gpu_seconds();
                int ended = metal_session_end();
                if (!encoded || !ended) return INFINITY;
                // GPU execution time when the driver reports it: the wall time of a
                // sub-millisecond kernel is mostly command buffer round trip.
                const double gpu = (metal_session_gpu_seconds() - gpu0) * 1000;
                return gpu > 0 ? gpu : ([NSProcessInfo processInfo].systemUptime - start) * 1000;
            };
            if (!isfinite(run(baseline))) return 0;
            NSData *reference = [NSData dataWithBytes:output.contents length:(NSUInteger)m * n * 4];
            const float *ref = reference.bytes;
            NSMutableArray<BBQ4Plan *> *valid = [NSMutableArray arrayWithObject:baseline];
            for (NSUInteger p = 1; p < plans.count; ++p) {
                if (!isfinite(run(plans[p]))) {
                    if (getenv("BYBY_Q4_TUNE_LOG")) NSLog(@"[q4-tune]   %@ failed to run", plans[p].label);
                    continue;
                }
                float err = 0, mag = 0;
                BOOL finite = YES;
                const float *v = output.contents;
                for (NSUInteger i = 0; i < (NSUInteger)m * n; ++i) {
                    if (!isfinite(v[i])) { finite = NO; break; }
                    err = fmaxf(err, fabsf(v[i] - ref[i])); mag = fmaxf(mag, fabsf(ref[i]));
                }
                if (finite && err <= fmaxf(1e-5f, mag * 1e-4f)) [valid addObject:plans[p]];
                else if (getenv("BYBY_Q4_TUNE_LOG")) NSLog(@"[q4-tune]   %@ rejected: err %g mag %g", plans[p].label, err, mag);
            }
            double times[24][3];
            for (NSUInteger round = 0; round < 3; ++round)
                for (NSUInteger j = 0; j < valid.count; ++j) {
                    NSUInteger i = round % 2 ? valid.count - 1 - j : j;
                    times[i][round] = run(valid[i]);
                }
            double medians[24];
            for (NSUInteger i = 0; i < valid.count; ++i) {
                double a = times[i][0], b = times[i][1], c = times[i][2];
                medians[i] = fmin(a, fmin(b, c));   // fastest run: a stall only ever adds time
            }
            if (getenv("BYBY_Q4_TUNE_LOG"))
                for (NSUInteger i = 0; i < valid.count; ++i)
                    NSLog(@"[q4-tune]   %@ %.3fms", valid[i].label, medians[i]);
            const char *setting = getenv("BYBY_Q4_TUNED");
            NSUInteger best = setting && strcmp(setting, "wide") == 0 && valid.count > 1 ? 1 : 0;
            for (NSUInteger i = best == 0 ? 1 : 2; i < valid.count; ++i)
                if (medians[i] < medians[best] * 0.97) best = i;
            BBQ4Plan *chosen = valid[best];
            double chosenMs = medians[best];
            // Second stage for a simdgroup plan: the weight read variants of its
            // shape, each parity checked, timed by the fastest of five runs.
            // BYBY_Q4_VARIANT=packs,load,pre forces one; BYBY_Q4_VARIANT=0 skips the stage.
            const char *forced = getenv("BYBY_Q4_VARIANT");
            if (chosen.simd && !(forced && strcmp(forced, "0") == 0)) {
                NSMutableArray<BBQ4Plan *> *variants = [NSMutableArray array];
                unsigned fp = 0, fl = 0, fs = 0;
                const BOOL haveForced = forced && sscanf(forced, "%u,%u,%u", &fp, &fl, &fs) == 3;
                for (uint32_t packs = 1; packs <= 4; packs *= 2) {
                    if (k % (8 * packs) || (group > 0 && group % (8 * packs))) continue;
                    for (uint32_t load = 0; load < 3; ++load)
                        for (uint32_t pre = 0; pre < 2; ++pre) {
                            if (packs == 1 && load == 0 && pre == 0) continue;
                            if (haveForced && (packs != fp || load != fl || pre != fs)) continue;
                            BBQ4Plan *p = q4MakeSimdPlan(chosen.sgs, chosen.simdRows, packs, load, pre);
                            if (p) [variants addObject:p];
                        }
                }
                double bestMs = INFINITY;
                for (int rep = 0; rep < 5; ++rep) bestMs = fmin(bestMs, run(chosen));
                if (haveForced) bestMs = INFINITY;
                chosenMs = isfinite(bestMs) ? bestMs : chosenMs;
                for (BBQ4Plan *p in variants) {
                    if (!isfinite(run(p))) continue;
                    float err = 0, mag = 0;
                    BOOL finite = YES;
                    const float *v = output.contents;
                    for (NSUInteger i = 0; i < (NSUInteger)m * n; ++i) {
                        if (!isfinite(v[i])) { finite = NO; break; }
                        err = fmaxf(err, fabsf(v[i] - ref[i])); mag = fmaxf(mag, fabsf(ref[i]));
                    }
                    if (!finite || err > fmaxf(1e-5f, mag * 1e-4f)) {
                        if (getenv("BYBY_Q4_TUNE_LOG")) NSLog(@"[q4-tune]   %@ rejected: err %g mag %g", p.label, err, mag);
                        continue;
                    }
                    double ms = INFINITY;
                    for (int rep = 0; rep < 5; ++rep) ms = fmin(ms, run(p));
                    if (getenv("BYBY_Q4_TUNE_LOG")) NSLog(@"[q4-tune]   %@ %.3fms", p.label, ms);
                    if (ms < bestMs * 0.97) { bestMs = ms; chosen = p; chosenMs = ms; }
                }
            }
            gQ4Plans[key] = chosen;
            // Effective read rate: packed weights plus fp32 scale and zero per group.
            const double bytes = (double)n * (k / 2.0 + 8.0 * (group > 0 ? (double)k / group : 1.0));
            NSLog(@"[q4-tune] %dx%d M=%d shift %d: %@ %.3fms = %.1f GB/s (legacy %.3fms; GPU parity checked)",
                k, n, m, wqOffset & 3, chosen.label, chosenMs, bytes / (chosenMs * 1e-3) / 1e9, medians[0]);
            return 1;
        } @finally { gQ4Tuning = NO; }
    }
}
