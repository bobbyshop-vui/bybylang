## Included by vulkan_driver: Q4 plans share its device/session/pipeline cache.
type
  VkQ4PlanKind = enum q4Reference, q4Wide, q4Cooperative
  VkQ4Plan = object
    kind: VkQ4PlanKind
    tg, rows, batch: int
    name, spv: string

var gVkQ4Plans = initTable[string, VkQ4Plan]()
var gVkQ4WideDispatches = 0

proc q4Key(K, N, group, groups, M: int): string =
  $K & "_" & $N & "_g" & $group & "_" & $groups & "_m" & $(if M == 1: 1 else: 8)

proc q4ReferenceEnc(a, w, scales, zeros, c: VulkanBufferHandle;
                    M, K, N, group, groups: int): bool =
  var pc = [int32(M), int32(K), int32(N), int32(group), int32(groups)]
  dispatchGeneric("matmul_q4", [a, w, scales, zeros, c], addr pc[0], 20,
    uint32((N + MMQ4_NDST - 1) div MMQ4_NDST), uint32(M), 1)

proc q4Encode(plan: VkQ4Plan; a, w, scales, zeros, c: VulkanBufferHandle;
              M, K, N, group, groups: int; countDispatch = false): bool =
  case plan.kind
  of q4Reference:
    result = q4ReferenceEnc(a, w, scales, zeros, c, M, K, N, group, groups)
  of q4Cooperative:
    result = tryCooperativeMatmul(a, w, c, scales, zeros, M, K, N, 2, group, groups)
  of q4Wide:
    var pc = [int32(M), int32(K), int32(N), int32(group), int32(groups)]
    result = dispatchGeneric(plan.name, [a, w, scales, zeros, c], addr pc[0], 20,
      uint32((N + plan.rows - 1) div plan.rows),
      uint32((M + plan.batch - 1) div plan.batch), 1, plan.tg, plan.spv)
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
  discard getOrCreatePipeline(result.name, 5, 20, tg, result.spv)

proc vulkanPrepareQ4Tuned*(w, scales, zeros: VulkanBufferHandle;
                           K, N, group, groups, mode: int): bool =
  ensureInit()
  if gSessionActive or getEnv("BYBY_Q4_TUNED", "1") == "0" or
     getEnv("BYBY_GPU_AUTOTUNE", "1") == "0" or findExe("glslc").len == 0: return false
  if K <= 0 or K mod 8 != 0 or N <= 0 or group < 0 or
     (group > 0 and group mod 8 != 0) or groups <= 0: return false
  if groups < (if group > 0: (K + group - 1) div group else: 1): return false
  if w.buf == nil or scales.buf == nil or zeros.buf == nil or
     w.sizeBytes < uint64(N) * uint64(K div 2) or
     scales.sizeBytes < uint64(N) * uint64(groups) * 4 or
     zeros.sizeBytes < uint64(N) * uint64(groups) * 4: return false
  let M = if mode == 1: 1 else: 8
  let key = q4Key(K, N, group, groups, M)
  if key in gVkQ4Plans: return true
  var plans = @[VkQ4Plan(kind: q4Reference, name: "reference-tg128")]
  let geometries = if M == 1: @[(4, 1)] else: @[(1, 8), (2, 4)]
  for (rows, batch) in geometries:
    for tg in [32, 64, 128, 256, 512, 1024]:
      if tg > gDeviceCaps.maxThreads or rows * batch * tg * 4 > gDeviceCaps.maxSharedBytes: continue
      try:
        plans.add q4Compile(tg, rows, batch)
      except CatchableError as e:
        stderr.writeLine "[q4-tune:vulkan] skip candidate: " & e.msg
  if M > 1 and gCoopEnabled and K >= 64 and N >= 16:
    plans.add VkQ4Plan(kind: q4Cooperative, name: "KHR-cooperative-matrix")
  if plans.len == 1: return false
  var data = newSeq[float32](M * K)
  for i in 0 ..< data.len: data[i] = float32((i * 17 mod 101) - 50) / 64
  var input = vulkanSessionUpload(data)
  var output = vulkanSessionAllocScratch(M * N)
  defer:
    vulkanSessionFree(input); vulkanSessionFree(output)
  proc run(plan: VkQ4Plan): tuple[ok: bool, ms: float] =
    let start = getMonoTime()
    if not vulkanSessionBegin(): return (false, 0.0)
    var encoded = false
    try:
      encoded = q4Encode(plan, input, w, scales, zeros, output, M, K, N, group, groups)
    except CatchableError as e:
      stderr.writeLine "[q4-tune:vulkan] reject " & plan.name & ": " & e.msg
    if not vulkanSessionEnd():
      raise newException(CatchableError, "Q4 tuning GPU failed: " & vulkanGetLastError())
    (encoded, float((getMonoTime() - start).inNanoseconds) / 1e6)
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
      let tolerance = if plan.kind == q4Cooperative: 3e-3'f32 else: 1e-4'f32
      if finite and err <= max(1e-5'f32, mag * tolerance): valid.add plan
  var samples = newSeq[seq[float]](valid.len)
  for round in 0 ..< 3:
    for j in 0 ..< valid.len:
      let i = if round mod 2 == 0: j else: valid.high - j
      let measured = run(valid[i])
      if not measured.ok: raise newException(CatchableError, "Q4 timed dispatch failed")
      samples[i].add measured.ms
  for times in samples.mitems: times.sort()
  var best = 0
  if getEnv("BYBY_Q4_TUNED", "1") == "wide":
    for i in 1 ..< valid.len:
      if valid[i].kind == q4Wide and (best == 0 or samples[i][1] < samples[best][1]): best = i
  else:
    for i in 1 ..< valid.len:
      if samples[i][1] < samples[best][1] * 0.97: best = i
  gVkQ4Plans[key] = valid[best]
  stderr.writeLine "[q4-tune:vulkan] " & $K & "x" & $N & " M=" & $M & ": " &
    valid[best].name & " " & formatFloat(samples[best][1], ffDecimal, 3) &
    "ms (reference " & formatFloat(samples[0][1], ffDecimal, 3) &
    "ms; " & $valid.len & " validated candidates)"
  true

proc vulkanQ4TuningStats*(): tuple[preparedPlans, wideDispatches: int] =
  (gVkQ4Plans.len, gVkQ4WideDispatches)

proc vulkanSessionMatmulQ4Enc*(a: VulkanBufferHandle, wq, scales, zeros: VulkanBufferHandle,
                                c: var VulkanBufferHandle, M, K, N, groupSize, nGroupsPerRow: int): bool =
  if getEnv("BYBY_Q4_TUNED", "1") != "0":
    let key = q4Key(K, N, groupSize, nGroupsPerRow, M)
    if key in gVkQ4Plans:
      if q4Encode(gVkQ4Plans[key], a, wq, scales, zeros, c,
          M, K, N, groupSize, nGroupsPerRow, countDispatch = true): return true
    elif tryCooperativeMatmul(a, wq, c, scales, zeros, M, K, N, 2,
        groupSize, nGroupsPerRow): return true
  q4ReferenceEnc(a, wq, scales, zeros, c, M, K, N, groupSize, nGroupsPerRow)
