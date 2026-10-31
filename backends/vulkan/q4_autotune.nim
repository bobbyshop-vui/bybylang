## Included by vulkan_driver: Q4 plans share its device/session/pipeline cache.
type
  VkQ4PlanKind = enum q4Reference, q4Wide, q4Cooperative, q4Subgroup
  VkQ4Plan = object
    kind: VkQ4PlanKind
    tg, rows, batch, vec: int
    name, spv: string

var gVkQ4Plans = initTable[string, VkQ4Plan]()
var gVkQ4WideDispatches = 0

proc q4Key(K, N, group, groups, M: int): string =
  $K & "_" & $N & "_g" & $group & "_" & $groups & "_m" & $(if M == 1: 1 else: 8)

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
    # The cooperative kernel reads weights from the start of the buffer.
    result = wqOffset == 0 and
      tryCooperativeMatmul(a, w, c, scales, zeros, M, K, N, 2, group, groups)
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
  # Every M>1 shares the "m8" plan. With tensor cores, time the candidates at
  # a prompt-sized M: the cooperative kernel's 16x16 tiles are mostly idle at
  # M=8 and would lose to the scalar kernels that prefill then outgrows.
  let coopCandidate = mode != 1 and gCoopEnabled and gDeviceCaps.cooperativeMatrix and
    K >= 64 and N >= 16 and wqOffset == 0
  let M = if mode == 1: 1 elif coopCandidate: 64 else: 8
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
    plans.add VkQ4Plan(kind: q4Cooperative, name: "KHR-cooperative-matrix")
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
      let tolerance = if plan.kind == q4Cooperative: 3e-3'f32 else: 1e-4'f32
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
                                c: var VulkanBufferHandle, M, K, N, groupSize, nGroupsPerRow: int;
                                wqOffset = 0): bool =
  ## wqOffset: byte position of the packed weights inside wq.
  if getEnv("BYBY_Q4_TUNED", "1") != "0":
    let key = q4Key(K, N, groupSize, nGroupsPerRow, M)
    if key in gVkQ4Plans:
      if q4Encode(gVkQ4Plans[key], a, wq, scales, zeros, c,
          M, K, N, groupSize, nGroupsPerRow, wqOffset, countDispatch = true): return true
    elif wqOffset == 0 and tryCooperativeMatmul(a, wq, c, scales, zeros, M, K, N, 2,
        groupSize, nGroupsPerRow): return true
  q4ReferenceEnc(a, wq, scales, zeros, c, M, K, N, groupSize, nGroupsPerRow, wqOffset)
