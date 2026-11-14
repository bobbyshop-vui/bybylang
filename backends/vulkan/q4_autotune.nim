## Included by vulkan_driver: Q4 plans share its device/session/pipeline cache.
type
  VkQ4PlanKind = enum q4Reference, q4Wide, q4Cooperative, q4Subgroup, q4CoopTiled, q4Gemm
  VkQ4Plan = object
    kind: VkQ4PlanKind
    tg, rows, batch, vec: int
    name, spv: string

var gVkQ4Plans = initTable[string, VkQ4Plan]()
var gVkQ4WideDispatches = 0
var gVkQ4TuneCacheLoaded = false
var gVkQ4TuneCache = initTable[string, JsonNode]()

const Q4TuneCacheVersion = 1
const Q4TuneSourceSignature = staticRead("kernels/matmul_q4.comp") &
  staticRead("kernels/matmul_q4_dynamic.glsl") &
  staticRead("kernels/matmul_q4_sg.glsl") &
  staticRead("kernels/matmul_q4_gemm.glsl") &
  staticRead("kernels/matmul_coop.glsl") &
  staticRead("kernels/matmul_fp4km_tiled.glsl")

proc q4Compile(tg, rows, batch: int): VkQ4Plan
proc q4CompileSubgroup(tg, rows, vec: int; sharedReduce: bool): VkQ4Plan
proc q4CompileGemm(bm, bn, tm, tn: int): VkQ4Plan

proc q4StableHash(s: string): uint64 =
  ## Stable FNV-1a: std/hash is not a persistent on-disk cache key.
  result = 14695981039346656037'u64
  for c in s:
    result = (result xor uint64(ord(c))) * 1099511628211'u64

proc q4TuneCacheSignature(): string =
  let props = gDeviceCaps.properties
  var uuid = ""
  for b in props.pipelineCacheUUID: uuid.add(toHex(int(b), 2))
  let sourceHash = toHex(q4StableHash(Q4TuneSourceSignature), 16)
  result = "v" & $Q4TuneCacheVersion & "|" & deviceName(gDeviceCaps) &
    "|vendor=" & $props.vendorID & "|device=" & $props.deviceID &
    "|driver=" & $props.driverVersion & "|api=" & $props.apiVersion & "|uuid=" & uuid &
    "|subgroup=" & $gDeviceCaps.subgroupSize & "|threads=" & $gDeviceCaps.maxThreads &
    "|shared=" & $gDeviceCaps.maxSharedBytes & "|coop=" & $gCoopEnabled &
    "|coopRequired=" & $gCoopRequired & "|fp8=" & $gFp8CoopEnabled &
    "|tuned=" & getEnv("BYBY_Q4_TUNED", "1") &
    "|subgroupMode=" & getEnv("BYBY_Q4_SUBGROUP", "1") &
    "|tensorCores=" & getEnv("BYBY_VULKAN_TENSOR_CORES", "1") &
    "|source=" & sourceHash

proc q4TuneCachePath(): string =
  let sigHash = toHex(q4StableHash(q4TuneCacheSignature()), 16)
  vulkanJitCacheDir() / ("q4-plans-" & sigHash & ".json")

proc q4LoadTuneCache() =
  if gVkQ4TuneCacheLoaded: return
  gVkQ4TuneCacheLoaded = true
  let path = q4TuneCachePath()
  if not fileExists(path): return
  try:
    let root = parseJson(readFile(path))
    if not root.hasKey("signature") or root["signature"].getStr() != q4TuneCacheSignature() or
        not root.hasKey("plans") or root["plans"].kind != JObject:
      return
    for key, plan in root["plans"].pairs:
      gVkQ4TuneCache[key] = plan
  except CatchableError:
    gVkQ4TuneCache.clear()

proc q4SaveTuneCache(key: string, plan: VkQ4Plan) =
  q4LoadTuneCache()
  var entry = newJObject()
  entry["kind"] = newJString($plan.kind)
  entry["name"] = newJString(plan.name)
  entry["tg"] = newJInt(plan.tg)
  entry["rows"] = newJInt(plan.rows)
  entry["batch"] = newJInt(plan.batch)
  entry["vec"] = newJInt(plan.vec)
  entry["sharedReduce"] = newJBool(plan.name.endsWith("_shm"))
  gVkQ4TuneCache[key] = entry
  var root = newJObject()
  root["signature"] = newJString(q4TuneCacheSignature())
  var plans = newJObject()
  for cacheKey, cachedPlan in gVkQ4TuneCache.pairs:
    plans[cacheKey] = cachedPlan
  root["plans"] = plans
  try:
    writeFile(q4TuneCachePath(), $root)
  except CatchableError:
    discard

proc q4CachedPlan(key: string): tuple[plan: VkQ4Plan, found: bool] =
  q4LoadTuneCache()
  if key notin gVkQ4TuneCache: return
  let entry = gVkQ4TuneCache[key]
  try:
    let kind = entry["kind"].getStr()
    let name = entry["name"].getStr()
    let tg = entry["tg"].getInt()
    let rows = entry["rows"].getInt()
    let batch = entry["batch"].getInt()
    let vec = entry["vec"].getInt()
    case kind
    of "q4Reference": result.plan = VkQ4Plan(kind: q4Reference, name: name)
    of "q4Wide": result.plan = q4Compile(tg, rows, batch)
    of "q4Subgroup": result.plan = q4CompileSubgroup(tg, rows, vec, entry["sharedReduce"].getBool())
    of "q4Gemm":
      if batch == 32 and rows == 64: result.plan = q4CompileGemm(32, 64, 4, 4)
      elif batch == 64 and rows == 64: result.plan = q4CompileGemm(64, 64, 8, 4)
      elif batch == 32 and rows == 128: result.plan = q4CompileGemm(32, 128, 4, 8)
      elif batch == 64 and rows == 128: result.plan = q4CompileGemm(64, 128, 8, 8)
      else: return
    of "q4Cooperative": result.plan = VkQ4Plan(kind: q4Cooperative, name: name)
    of "q4CoopTiled": result.plan = VkQ4Plan(kind: q4CoopTiled, name: name)
    else: return
    result.found = result.plan.name == name
  except CatchableError:
    result.found = false
  if not result.found:
    gVkQ4TuneCache.del(key)

proc q4Key(K, N, group, groups, M: int): string =
  $K & "_" & $N & "_g" & $group & "_" & $groups & "_m" & $(if M == 1: 1 else: 64)

proc q4ReferenceEnc(a, w, scales, zeros, c: VulkanBufferHandle;
                    M, K, N, group, groups, wqOffset: int): bool =
  var pc = [int32(M), int32(K), int32(N), int32(group), int32(groups), int32(wqOffset)]
  dispatchGeneric("matmul_q4", [a, w, scales, zeros, c], addr pc[0], 24,
    uint32((N + MMQ4_NDST - 1) div MMQ4_NDST), uint32(M), 1)

proc q4Encode(plan: VkQ4Plan; a, w, scales, zeros, c: VulkanBufferHandle;
              M, K, N, group, groups, wqOffset: int; countDispatch = false): bool =
  case plan.kind
  of q4Reference:
    result = q4ReferenceEnc(a, w, scales, zeros, c, M, K, N, group, groups, wqOffset)
  of q4Cooperative:
    result = tryCooperativeMatmul(a, w, c, scales, zeros, M, K, N, 2, group, groups,
      wqOffset)
  of q4CoopTiled:
    result = tryTiledCooperative(a, w, c, scales, M, K, N, 2, zeros, group, groups,
      wqOffset)
  of q4Gemm:
    var pc = [int32(M), int32(K), int32(N), int32(group), int32(groups), int32(wqOffset)]
    result = dispatchGeneric(plan.name, [a, w, scales, zeros, c], addr pc[0], 24,
      uint32((N + plan.rows - 1) div plan.rows),
      uint32((M + plan.batch - 1) div plan.batch), 1, 128, plan.spv)
  of q4Wide:
    var pc = [int32(M), int32(K), int32(N), int32(group), int32(groups), int32(wqOffset)]
    result = dispatchGeneric(plan.name, [a, w, scales, zeros, c], addr pc[0], 24,
      uint32((N + plan.rows - 1) div plan.rows),
      uint32((M + plan.batch - 1) div plan.batch), 1, plan.tg, plan.spv)
    if result and countDispatch: inc gVkQ4WideDispatches
  of q4Subgroup:
    # Rows per workgroup: one subgroup per plan.rows rows.
    let rowsPerGroup = plan.rows * plan.batch
    if wqOffset mod (4 * plan.vec) != 0: return false
    var pc = [int32(M), int32(K), int32(N), int32(group), int32(groups), int32(wqOffset)]
    result = dispatchGeneric(plan.name, [a, w, scales, zeros, c], addr pc[0], 24,
      uint32((N + rowsPerGroup - 1) div rowsPerGroup), uint32(M), 1, plan.tg, plan.spv)
    if result and countDispatch: inc gVkQ4WideDispatches

proc q4Compile(tg, rows, batch: int): VkQ4Plan =
  const sourceTemplate = staticRead("kernels/matmul_q4_dynamic.glsl")
  result = VkQ4Plan(kind: q4Wide, tg: tg, rows: rows, batch: batch,
    name: "q4_wide_r" & $rows & "_b" & $batch & "_tg" & $tg)
  let source = "#version 450\n#define WG_SIZE " & $tg &
    "\n#define OUT_ROWS " & $rows & "\n#define BATCH_ROWS " & $batch & "\n" &
    sourceTemplate.replace("#version 450\n", "")
  result.spv = compileVulkanJitShader(result.name, source)
  # Create the PSO before timing or opening an inference command buffer.
  discard getOrCreatePipeline(result.name, 5, 24, tg, result.spv)

proc q4CompileSubgroup(tg, rows, vec: int; sharedReduce: bool): VkQ4Plan =
  const sourceTemplate = staticRead("kernels/matmul_q4_sg.glsl")
  # batch holds the subgroups per workgroup the grid assumes (SG_HINT).
  let sg = gDeviceCaps.subgroupSize
  result = VkQ4Plan(kind: q4Subgroup, tg: tg, rows: rows, batch: tg div sg, vec: vec,
    name: "q4_sg_r" & $rows & "_v" & $vec & "_tg" & $tg & "_s" & $sg &
      (if sharedReduce: "_shm" else: ""))
  let source = "#version 450\n#define WG_SIZE " & $tg & "\n#define ROWS " & $rows &
    "\n#define VEC " & $vec & "\n#define SG_HINT " & $sg &
    "\n#define SHARED_REDUCE " & $int(sharedReduce) & "\n" &
    sourceTemplate.replace("#version 450\n", "")
  result.spv = compileVulkanJitShader(result.name, source)
  discard getOrCreatePipeline(result.name, 5, 24, tg, result.spv)

proc q4CompileGemm(bm, bn, tm, tn: int): VkQ4Plan =
  const sourceTemplate = staticRead("kernels/matmul_q4_gemm.glsl")
  result = VkQ4Plan(kind: q4Gemm, rows: bn, batch: bm,
    name: "q4_gemm_" & $bm & "x" & $bn)
  let source = "#version 450\n#define BM " & $bm & "\n#define BN " & $bn &
    "\n#define TM " & $tm & "\n#define TN " & $tn & "\n" &
    sourceTemplate.replace("#version 450\n", "")
  result.spv = compileVulkanJitShader(result.name, source)
  discard getOrCreatePipeline(result.name, 5, 24, 128, result.spv)

const Q4FusedShader = (name: "matmul_q4_fused", source: """#version 450
#define WG_SIZE 128
#define OUT_ROWS 4
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Input { float a[]; };
layout(std430, binding = 1) readonly buffer Weight0 { uint w0[]; };
layout(std430, binding = 2) readonly buffer Weight1 { uint w1[]; };
layout(std430, binding = 3) readonly buffer Weight2 { uint w2[]; };
layout(std430, binding = 4) readonly buffer Scale0 { float sc0[]; };
layout(std430, binding = 5) readonly buffer Scale1 { float sc1[]; };
layout(std430, binding = 6) readonly buffer Scale2 { float sc2[]; };
layout(std430, binding = 7) readonly buffer Zero0 { float zr0[]; };
layout(std430, binding = 8) readonly buffer Zero1 { float zr1[]; };
layout(std430, binding = 9) readonly buffer Zero2 { float zr2[]; };
layout(std430, binding = 10) writeonly buffer Output0 { float out0[]; };
layout(std430, binding = 11) writeonly buffer Output1 { float out1[]; };
layout(std430, binding = 12) writeonly buffer Output2 { float out2[]; };
layout(std430, binding = 13) readonly buffer Residual { float residual[]; };
layout(std430, binding = 14) readonly buffer Gamma { float gamma[]; };
layout(push_constant) uniform Params {
    uint K, groupSize, groupsPerRow, mode, flags;
    uint N0, N1, N2, off0, off1, off2;
    float eps;
} p;
shared float partialA[OUT_ROWS * WG_SIZE];
shared float partialB[OUT_ROWS * WG_SIZE];
shared float partialSq[WG_SIZE];

uint packedWord(uint matrix, uint row, uint k) {
    uint off = matrix == 0u ? p.off0 : (matrix == 1u ? p.off1 : p.off2);
    uint byteIndex = off + row * (p.K / 2u) + k / 2u;
    uint wordIndex = byteIndex >> 2u;
    uint shift = (byteIndex & 3u) * 8u;
    uint lo = matrix == 0u ? w0[wordIndex] : (matrix == 1u ? w1[wordIndex] : w2[wordIndex]);
    if (shift == 0u) return lo;
    uint hi = matrix == 0u ? w0[wordIndex + 1u] :
              (matrix == 1u ? w1[wordIndex + 1u] : w2[wordIndex + 1u]);
    return (lo >> shift) | (hi << (32u - shift));
}
float scaleAt(uint matrix, uint index) {
    return matrix == 0u ? sc0[index] : (matrix == 1u ? sc1[index] : sc2[index]);
}
float zeroAt(uint matrix, uint index) {
    return matrix == 0u ? zr0[index] : (matrix == 1u ? zr1[index] : zr2[index]);
}
void main() {
    uint lane = gl_LocalInvocationID.x;
    uint row0 = gl_WorkGroupID.x * OUT_ROWS;
    uint totalN = p.mode == 0u ? p.N0 + p.N1 + p.N2 : p.N0;
    if (row0 >= totalN) return;

    uint matrix = 0u, base = 0u, matrixN = p.N0;
    if (p.mode == 0u && row0 >= p.N0) {
        matrix = 1u; base = p.N0; matrixN = p.N1;
        if (row0 >= p.N0 + p.N1) {
            matrix = 2u; base = p.N0 + p.N1; matrixN = p.N2;
        }
    }

    float accA[OUT_ROWS];
    float accB[OUT_ROWS];
    for (uint r = 0u; r < OUT_ROWS; ++r) { accA[r] = 0.0; accB[r] = 0.0; }
    float ssq = 0.0;
    uint groupCount = (p.K + 7u) / 8u;
    for (uint k = lane * 8u; k < p.K; k += WG_SIZE * 8u) {
        uint g = p.groupSize > 0u ? k / p.groupSize : 0u;
        float x[8];
        for (uint t = 0u; t < 8u; ++t) {
            float raw = a[k + t];
            if ((p.flags & 2u) != 0u) {
                ssq = fma(raw, raw, ssq);
                x[t] = raw * gamma[k + t];
            } else x[t] = raw;
        }
        for (uint r = 0u; r < OUT_ROWS; ++r) {
            uint row = row0 + r - (p.mode == 0u ? base : 0u);
            if (row >= matrixN) continue;
            uint gi = row * p.groupsPerRow + g;
            if (p.mode == 0u) {
                uint packed = packedWord(matrix, row, k);
                float scale = scaleAt(matrix, gi), zero = zeroAt(matrix, gi);
                for (uint t = 0u; t < 8u; ++t) {
                    float weight = (float((packed >> (4u * t)) & 15u) - zero) * scale;
                    accA[r] = fma(weight, x[t], accA[r]);
                }
            } else {
                uint packedG = packedWord(0u, row, k);
                uint packedU = packedWord(1u, row, k);
                float scaleG = sc0[gi], zeroG = zr0[gi];
                float scaleU = sc1[gi], zeroU = zr1[gi];
                for (uint t = 0u; t < 8u; ++t) {
                    float weightG = (float((packedG >> (4u * t)) & 15u) - zeroG) * scaleG;
                    float weightU = (float((packedU >> (4u * t)) & 15u) - zeroU) * scaleU;
                    accA[r] = fma(weightG, x[t], accA[r]);
                    accB[r] = fma(weightU, x[t], accB[r]);
                }
            }
        }
    }
    for (uint r = 0u; r < OUT_ROWS; ++r) {
        partialA[r * WG_SIZE + lane] = accA[r];
        partialB[r * WG_SIZE + lane] = accB[r];
    }
    partialSq[lane] = ssq;
    barrier();
    for (uint stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            for (uint r = 0u; r < OUT_ROWS; ++r) {
                partialA[r * WG_SIZE + lane] += partialA[r * WG_SIZE + lane + stride];
                if (p.mode == 1u)
                    partialB[r * WG_SIZE + lane] += partialB[r * WG_SIZE + lane + stride];
            }
            partialSq[lane] += partialSq[lane + stride];
        }
        barrier();
    }
    if (lane == 0u) {
        float inv = (p.flags & 2u) != 0u ? inversesqrt(partialSq[0] / float(p.K) + p.eps) : 1.0;
        for (uint r = 0u; r < OUT_ROWS; ++r) {
            uint flatRow = row0 + r;
            if (flatRow >= totalN) continue;
            uint row = flatRow - (p.mode == 0u ? base : 0u);
            if (row >= matrixN) continue;
            float value = partialA[r * WG_SIZE] * inv;
            if (p.mode == 1u) {
                float up = partialB[r * WG_SIZE] * inv;
                value = value / (1.0 + exp(-value)) * up;
            }
            if ((p.flags & 1u) != 0u) value += residual[row];
            if (p.mode == 1u || matrix == 0u) out0[row] = value;
            else if (matrix == 1u) out1[row] = value;
            else out2[row] = value;
        }
    }
}
""")
const Q4FusedOutRows = 4

proc vulkanSessionMatmulQ4FusedEnc*(mode: int; a: VulkanBufferHandle;
    wq, scales, zeros, outputs: openArray[VulkanBufferHandle]; residual, gamma: VulkanBufferHandle;
    K, groupSize, groupsPerRow: int; Ns, offsets: openArray[int]; eps: float32;
    hasNorm, hasRes: bool): bool =
  ## Fused M=1 Q4 path matching Metal's shared-input Q/K/V and gate/up kernels.
  ## MODE 0 writes up to three projection vectors; MODE 1 writes silu(gate)*up.
  if not gSessionActive or a.buf == nil or K <= 0 or K mod 8 != 0 or
      groupSize < 0 or groupsPerRow <= 0 or wq.len < 1 or wq.len > 3 or
      scales.len < wq.len or zeros.len < wq.len or offsets.len < wq.len or
      outputs.len < wq.len or Ns.len < wq.len or mode notin [0, 1]: return false
  if mode == 1 and (wq.len != 2 or Ns[0] != Ns[1]): return false
  let count = wq.len
  for i in 0 ..< count:
    if wq[i].buf == nil or scales[i].buf == nil or zeros[i].buf == nil or outputs[i].buf == nil or
        offsets[i] < 0 or Ns[i] <= 0 or Ns[i] mod Q4FusedOutRows != 0: return false
    if wq[i].sizeBytes < uint64(offsets[i]) + uint64(Ns[i]) * uint64(K div 2) or
        scales[i].sizeBytes < uint64(Ns[i] * groupsPerRow) * 4 or
        zeros[i].sizeBytes < uint64(Ns[i] * groupsPerRow) * 4 or
        outputs[i].sizeBytes < uint64(Ns[i]) * 4: return false
  if a.sizeBytes < uint64(K) * 4: return false
  if hasNorm and (gamma.buf == nil or gamma.sizeBytes < uint64(K) * 4): return false
  if hasRes and (residual.buf == nil or residual.sizeBytes < uint64(Ns[0]) * 4): return false

  var spv: string
  try:
    spv = compileVulkanJitShader(Q4FusedShader.name, Q4FusedShader.source)
    discard getOrCreatePipeline(Q4FusedShader.name, 15, 48, 128, spv)
  except CatchableError as e:
    stderr.writeLine "[vulkan] fused Q4 unavailable: " & e.msg.splitLines()[0]
    return false

  var bindings: array[15, VulkanBufferHandle]
  bindings[0] = a
  for i in 0 ..< 3:
    let sourceIndex = min(i, count - 1)
    bindings[1 + i] = wq[sourceIndex]
    bindings[4 + i] = scales[sourceIndex]
    bindings[7 + i] = zeros[sourceIndex]
    bindings[10 + i] = outputs[min(i, outputs.len - 1)]
  bindings[13] = if hasRes: residual else: outputs[0]
  bindings[14] = if hasNorm: gamma else: scales[0]
  let n0 = uint32(Ns[0])
  let n1 = if count > 1: uint32(Ns[1]) else: 0'u32
  let n2 = if count > 2: uint32(Ns[2]) else: 0'u32
  let off0 = uint32(offsets[0])
  let off1 = if count > 1: uint32(offsets[1]) else: off0
  let off2 = if count > 2: uint32(offsets[2]) else: off1
  let flags = (if hasRes: 1'u32 else: 0'u32) or (if hasNorm: 2'u32 else: 0'u32)
  var pc = (K: uint32(K), groupSize: uint32(groupSize), groupsPerRow: uint32(groupsPerRow),
            mode: uint32(mode), flags: flags, N0: n0, N1: n1, N2: n2,
            off0: off0, off1: off1, off2: off2, eps: eps)
  let totalN = if mode == 0: int(n0 + n1 + n2) else: int(n0)
  dispatchGeneric(Q4FusedShader.name, bindings, addr pc, 48,
    uint32((totalN + Q4FusedOutRows - 1) div Q4FusedOutRows), 1, 1, 128, spv)

proc vulkanPrepareQ4Tuned*(w, scales, zeros: VulkanBufferHandle;
                           K, N, group, groups, mode: int; wqOffset = 0): bool =
  ensureInit()
  if gSessionActive or getEnv("BYBY_Q4_TUNED", "1") == "0" or
     getEnv("BYBY_GPU_AUTOTUNE", "1") == "0" or findExe("glslc").len == 0: return false
  if K <= 0 or K mod 8 != 0 or N <= 0 or group < 0 or
     (group > 0 and group mod 8 != 0) or groups <= 0: return false
  if groups < (if group > 0: (K + group - 1) div group else: 1): return false
  if w.buf == nil or scales.buf == nil or zeros.buf == nil or
     wqOffset < 0 or w.sizeBytes < uint64(wqOffset) + uint64(N) * uint64(K div 2) or
     scales.sizeBytes < uint64(N) * uint64(groups) * 4 or
     zeros.sizeBytes < uint64(N) * uint64(groups) * 4: return false
  # Every prefill M>1 shares one plan. Tune it at M=64: the flash-prefill path
  # processes 64-row chunks, and tiled Q4 GEMM needs a representative tile to
  # compete with the reference. Tuning at M=8 systematically penalized it.
  let coopCandidate = mode != 1 and gCoopEnabled and gDeviceCaps.cooperativeMatrix and
    K >= 64 and N >= 16
  # Tensor cores required (dGPU): prefill only chooses between tensor-core
  # kernels. The reference stays first as the parity baseline and is used
  # only if no tensor-core kernel passes parity.
  let coopOnly = coopCandidate and gCoopRequired
  let M = if mode == 1: 1 else: 64
  let key = q4Key(K, N, group, groups, M)
  if key in gVkQ4Plans: return true
  let cached = q4CachedPlan(key)
  if cached.found:
    gVkQ4Plans[key] = cached.plan
    stderr.writeLine "[q4-tune:vulkan] " & $K & "x" & $N & " M=" & $M &
      ": cache hit " & cached.plan.name
    return true
  let tuneStart = getMonoTime()
  var plans = @[VkQ4Plan(kind: q4Reference, name: "reference-tg128")]
  let geometries = if M == 1: @[(4, 1)] elif coopOnly: newSeq[(int, int)]() else: @[(1, 8), (2, 4)]
  for (rows, batch) in geometries:
    for tg in [32, 64, 128, 256, 512, 1024]:
      if tg > gDeviceCaps.maxThreads or rows * batch * tg * 4 > gDeviceCaps.maxSharedBytes: continue
      try:
        plans.add q4Compile(tg, rows, batch)
      except CatchableError as e:
        stderr.writeLine "[q4-tune:vulkan] skip candidate: " & e.msg
  # Decode GEMV: lanes stream each weight row with coalesced uint/uvec4 loads
  # and sum through subgroupAdd where the driver reports subgroup arithmetic
  # (T4), else through shared memory; both variants are timed where both run.
  # MoltenVK on Intel reports the builtins without arithmetic and gets them
  # wrong there, which parity rejects (BYBY_Q4_SUBGROUP=force still tries).
  let sg = gDeviceCaps.subgroupSize
  let sgMode = getEnv("BYBY_Q4_SUBGROUP", "1")
  var reduceModes = @[true]
  if sgMode == "force" or gDeviceCaps.subgroupArithmetic: reduceModes.add false
  if M == 1 and sg > 0 and sgMode != "0":
    for sharedReduce in reduceModes:
      for vec in [1, 4]:
        if K mod (8 * vec) != 0 or (group > 0 and group mod (8 * vec) != 0): continue
        for rows in [1, 2, 4]:
          for tg in [64, 128, 256]:
            if tg < sg or tg mod sg != 0 or tg > gDeviceCaps.maxThreads: continue
            try:
              plans.add q4CompileSubgroup(tg, rows, vec, sharedReduce)
            except CatchableError as e:
              stderr.writeLine "[q4-tune:vulkan] skip candidate: " & e.msg
  if coopCandidate:
    if tiledCoopSupported() and K mod 64 == 0 and group mod 2 == 0 and wqOffset mod 16 == 0:
      plans.add VkQ4Plan(kind: q4CoopTiled, name: "KHR-cooperative-matrix-tiled64")
    plans.add VkQ4Plan(kind: q4Cooperative, name: "KHR-cooperative-matrix")
  # Match Metal's four tiled GEMM geometries for prefill. The 64x128 tile is
  # the default there; autotuning can select a smaller tile for constrained GPUs.
  if M > 1 and K mod 32 == 0 and (group == 0 or group mod 32 == 0) and
      wqOffset >= 0:
    for (bm, bn, tm, tn) in [(32, 64, 4, 4), (64, 64, 8, 4),
                              (32, 128, 4, 8), (64, 128, 8, 8)]:
      let sharedBytes = 4 * (32 * (bm + 4) + 32 * (bn + 4))
      if 128 <= gDeviceCaps.maxThreads and sharedBytes <= gDeviceCaps.maxSharedBytes:
        try:
          plans.add q4CompileGemm(bm, bn, tm, tn)
        except CatchableError as e:
          stderr.writeLine "[q4-tune:vulkan] skip Q4 GEMM candidate: " & e.msg
  if plans.len == 1: return false
  var data = newSeq[float32](M * K)
  for i in 0 ..< data.len: data[i] = float32((i * 17 mod 101) - 50) / 64
  var input = vulkanSessionUpload(data)
  var output = vulkanSessionAllocScratch(M * N)
  defer:
    vulkanSessionFree(input); vulkanSessionFree(output)
  proc run(plan: VkQ4Plan; reps = 1): tuple[ok: bool, ms: float] =
    ## reps dispatches in one submit: one submit + fence wait (~1 ms on a
    ## dGPU) otherwise swamps a GEMV of a few tens of microseconds and every
    ## candidate times the same.
    let start = getMonoTime()
    if not vulkanSessionBegin(): return (false, 0.0)
    var encoded = true
    try:
      for i in 0 ..< reps:
        encoded = encoded and q4Encode(plan, input, w, scales, zeros, output,
          M, K, N, group, groups, wqOffset)
        if not encoded: break
    except CatchableError as e:
      encoded = false
      stderr.writeLine "[q4-tune:vulkan] reject " & plan.name & ": " & e.msg
    if not vulkanSessionEnd():
      raise newException(CatchableError, "Q4 tuning GPU failed: " & vulkanGetLastError())
    (encoded, float((getMonoTime() - start).inNanoseconds) / 1e6 / float(reps))
  var reference: seq[float32]
  var valid: seq[VkQ4Plan]
  for i, plan in plans:
    if not run(plan).ok:
      if i == 0: return false
      continue
    let values = vulkanSessionRead(output, M * N)
    var finite = true
    for v in values:
      if classify(v) in {fcNan, fcInf, fcNegInf}: finite = false
    if i == 0:
      if not finite: return false
      reference = values
      valid.add plan
    else:
      var err, mag: float32
      for j, v in values:
        err = max(err, abs(v - reference[j])); mag = max(mag, abs(reference[j]))
      let tolerance = if plan.kind in {q4Cooperative, q4CoopTiled}: 3e-3'f32 else: 1e-4'f32
      if finite and err <= max(1e-5'f32, mag * tolerance): valid.add plan
      elif getEnv("BYBY_Q4_TUNE_VERBOSE", "0") == "1":
        stderr.writeLine "[q4-tune:vulkan] parity reject " & plan.name & ": err=" & $err &
          " max=" & $mag & " finite=" & $finite
  var samples = newSeq[seq[float]](valid.len)
  let reps = if M == 1: 32 else: 4
  for round in 0 ..< 3:
    for j in 0 ..< valid.len:
      let i = if round mod 2 == 0: j else: valid.high - j
      let measured = run(valid[i], reps)
      if not measured.ok: raise newException(CatchableError, "Q4 timed dispatch failed")
      samples[i].add measured.ms
  for times in samples.mitems: times.sort()
  var best = 0
  if coopOnly:
    best = -1
    for i in 1 ..< valid.len:
      if valid[i].kind in {q4Cooperative, q4CoopTiled} and
          (best < 0 or samples[i][1] < samples[best][1]): best = i
    if best < 0:
      stderr.writeLine "[q4-tune:vulkan] WARNING " & $K & "x" & $N &
        ": no tensor-core kernel passed parity, prefill falls back to the scalar reference"
      best = 0
  elif getEnv("BYBY_Q4_TUNED", "1") == "wide":
    for i in 1 ..< valid.len:
      if valid[i].kind == q4Wide and (best == 0 or samples[i][1] < samples[best][1]): best = i
  else:
    for i in 1 ..< valid.len:
      if samples[i][1] < samples[best][1] * 0.97: best = i
  gVkQ4Plans[key] = valid[best]
  q4SaveTuneCache(key, valid[best])
  stderr.writeLine "[q4-tune:vulkan] " & $K & "x" & $N & " M=" & $M & ": " &
    valid[best].name & " " & formatFloat(samples[best][1], ffDecimal, 3) &
    "ms (reference " & formatFloat(samples[0][1], ffDecimal, 3) &
    "ms; " & $valid.len & " validated candidates; tuneWall=" &
    formatFloat(float((getMonoTime() - tuneStart).inNanoseconds) / 1e9, ffDecimal, 3) & "s)"
  true

proc vulkanQ4TuningStats*(): tuple[preparedPlans, wideDispatches: int] =
  (gVkQ4Plans.len, gVkQ4WideDispatches)

proc vulkanSessionMatmulQ4Enc*(a: VulkanBufferHandle, wq, scales, zeros: VulkanBufferHandle,
                                c: var VulkanBufferHandle, M, K, N, groupSize, nGroupsPerRow: int;
                                wqOffset = 0): bool =
  ## wqOffset: byte position of the packed weights inside wq.
  if getEnv("BYBY_Q4_TUNED", "1") != "0":
    let key = q4Key(K, N, groupSize, nGroupsPerRow, M)
    if key in gVkQ4Plans:
      if q4Encode(gVkQ4Plans[key], a, wq, scales, zeros, c,
          M, K, N, groupSize, nGroupsPerRow, wqOffset, countDispatch = true): return true
    elif M > 1 and tryCooperativeMatmul(a, wq, c, scales, zeros, M, K, N, 2,
        groupSize, nGroupsPerRow, wqOffset): return true
  q4ReferenceEnc(a, wq, scales, zeros, c, M, K, N, groupSize, nGroupsPerRow, wqOffset)
