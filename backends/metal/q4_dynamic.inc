// Included after session state/weight upload declarations in metal_shim.m.
@interface BBQ4Plan : NSObject
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic) NSUInteger threads, rows, batch;
@property(nonatomic) BOOL legacy;
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

static int q4EncodePlan(BBQ4Plan *plan, MetalBufferHandle a, MetalBufferHandle w,
        MetalBufferHandle scale, MetalBufferHandle zero, MetalBufferHandle out,
        int m, int k, int n, int group, int groups) {
    if (!gSessionEncoder || !plan || plan.legacy) return 0;
    [gSessionEncoder setComputePipelineState:plan.pipeline];
    [gSessionEncoder setThreadgroupMemoryLength:plan.threads * plan.rows * plan.batch * sizeof(float) atIndex:0];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:1];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:2];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)zero offset:0 atIndex:3];
    [gSessionEncoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:4];
    [gSessionEncoder setBytes:&m length:4 atIndex:5];
    [gSessionEncoder setBytes:&k length:4 atIndex:6];
    [gSessionEncoder setBytes:&n length:4 atIndex:7];
    [gSessionEncoder setBytes:&group length:4 atIndex:8];
    [gSessionEncoder setBytes:&groups length:4 atIndex:9];
    [gSessionEncoder dispatchThreadgroups:MTLSizeMake((n + plan.rows - 1) / plan.rows,
        (m + plan.batch - 1) / plan.batch, 1)
        threadsPerThreadgroup:MTLSizeMake(plan.threads, 1, 1)];
    [gSessionEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return 1;
}

static int q4EncodeSelected(MetalBufferHandle a, MetalBufferHandle w,
        MetalBufferHandle scale, MetalBufferHandle zero, MetalBufferHandle out,
        int m, int k, int n, int group, int groups) {
    if (gQ4Tuning || !q4TunedEnabled()) return 0;
    BBQ4Plan *plan = gQ4Plans[q4PlanKey(k, n, group, groups, m)];
    return q4EncodePlan(plan, a, w, scale, zero, out, m, k, n, group, groups);
}

int metal_prepare_q4_tuned(MetalBufferHandle w, MetalBufferHandle scale,
        MetalBufferHandle zero, int k, int n, int group, int groups, int mode) {
    @autoreleasepool {
        const char *autoTune = getenv("BYBY_GPU_AUTOTUNE");
        if (!q4TunedEnabled() || (autoTune && strcmp(autoTune, "0") == 0)) return 0;
        if (!gDevice || !gLibrary || gSessionCmdBuf || k <= 0 || k % 8 || n <= 0 ||
            group < 0 || (group > 0 && group % 8) || groups <= 0 ||
            groups < (group > 0 ? (k + group - 1) / group : 1)) return 0;
        id<MTLBuffer> weight = (__bridge id<MTLBuffer>)w;
        id<MTLBuffer> scales = (__bridge id<MTLBuffer>)scale;
        id<MTLBuffer> zeros = (__bridge id<MTLBuffer>)zero;
        if (weight.length < (NSUInteger)n * (k / 2) ||
            scales.length < (NSUInteger)n * groups * 4 ||
            zeros.length < (NSUInteger)n * groups * 4) return 0;
        if (!gQ4Plans) gQ4Plans = [NSMutableDictionary new];
        const int m = mode == 1 ? 1 : 8;
        NSString *key = q4PlanKey(k, n, group, groups, m);
        if (gQ4Plans[key]) return 1;
        BBQ4Plan *baseline = [BBQ4Plan new];
        baseline.legacy = YES; baseline.label = @"legacy";
        NSMutableArray<BBQ4Plan *> *plans = [NSMutableArray arrayWithObject:baseline];
        for (NSUInteger tg = 32; tg <= 128; tg *= 2) {
            BBQ4Plan *p = q4MakePlan(tg, m == 1 ? 4 : 1, m);
            if (p) [plans addObject:p];
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
                    metal_matmul_q4_enc(hIn, w, scale, zero, hOut, m, k, n, group, groups) :
                    metal_matmul_q4_batched_enc(hIn, w, scale, zero, hOut, m, k, n, group, groups)) :
                    q4EncodePlan(p, hIn, w, scale, zero, hOut, m, k, n, group, groups);
                int ended = metal_session_end();
                return encoded && ended ? ([NSProcessInfo processInfo].systemUptime - start) * 1000 : INFINITY;
            };
            if (!isfinite(run(baseline))) return 0;
            NSData *reference = [NSData dataWithBytes:output.contents length:(NSUInteger)m * n * 4];
            const float *ref = reference.bytes;
            NSMutableArray<BBQ4Plan *> *valid = [NSMutableArray arrayWithObject:baseline];
            for (NSUInteger p = 1; p < plans.count; ++p) {
                if (!isfinite(run(plans[p]))) continue;
                float err = 0, mag = 0;
                BOOL finite = YES;
                const float *v = output.contents;
                for (NSUInteger i = 0; i < (NSUInteger)m * n; ++i) {
                    if (!isfinite(v[i])) { finite = NO; break; }
                    err = fmaxf(err, fabsf(v[i] - ref[i])); mag = fmaxf(mag, fabsf(ref[i]));
                }
                if (finite && err <= fmaxf(1e-5f, mag * 1e-4f)) [valid addObject:plans[p]];
            }
            double times[6][3];
            for (NSUInteger round = 0; round < 3; ++round)
                for (NSUInteger j = 0; j < valid.count; ++j) {
                    NSUInteger i = round % 2 ? valid.count - 1 - j : j;
                    times[i][round] = run(valid[i]);
                }
            double medians[6];
            for (NSUInteger i = 0; i < valid.count; ++i) {
                double a = times[i][0], b = times[i][1], c = times[i][2];
                medians[i] = fmax(fmin(a, b), fmin(fmax(a, b), c));
            }
            const char *setting = getenv("BYBY_Q4_TUNED");
            NSUInteger best = setting && strcmp(setting, "wide") == 0 && valid.count > 1 ? 1 : 0;
            for (NSUInteger i = best == 0 ? 1 : 2; i < valid.count; ++i)
                if (medians[i] < medians[best] * 0.97) best = i;
            gQ4Plans[key] = valid[best];
            NSLog(@"[q4-tune] %dx%d M=%d: %@ %.3fms (legacy %.3fms; GPU parity checked)",
                k, n, m, valid[best].label, medians[best], medians[0]);
            return 1;
        } @finally { gQ4Tuning = NO; }
    }
}
