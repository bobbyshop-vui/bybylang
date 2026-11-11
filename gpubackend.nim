# gpubackend.nim - GPU dispatch layer for BybyLang.
# Syntax in .bybylang:
#   gpu backend is "auto"     # or "cuda" / "metal" / "cpu"
#   gpu array A = [1, 2, 3, 4]
#   gpu add A, B -> C size 4
#   gpu sub A, B -> C
#   gpu mul A, B -> C
#   gpu div A, B -> C
#
# "auto" probes: CUDA (NVIDIA) -> Metal (macOS) -> Vulkan -> plain CPU.
# If an explicitly chosen backend is unavailable or fails, it falls back
# to the CPU so the program always gives a correct result instead of crashing.
import std/[strutils, math]
import ./probe
export probe
from std/os import getEnv
import ./backends/cpu/cpu_backend as cpu
import backends/metal/metal_backend
import backends/vulkan/vulkan_driver

# ── CUDA removed: stub types so SessionHandle.cudaHandle still type-checks.
# All gbCuda paths redirect to Vulkan at runtime.
type
  CudaResidentTensor* = object
    dptr*: uint64
    n*: int
  CudaTensor* = CudaResidentTensor

proc cuFree*(t: var CudaResidentTensor) = discard
proc cuUpload*(data: seq[float32]): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")


when defined(macosx):
  # metal_backend.nim already has Begin/Upload/AllocScratch/MatmulEnc/SoftmaxEnc/
  # End/Read/Free. VecOp and Activation encode-in-session are implemented
  # directly in the ObjC shim (metal_shim.m/.h) and declared here (they MUST come before
  # their use - sessionVecOp/sessionActivation below - because Nim does not allow
  # forward references to top-level procs).
  proc metalSessionVecOpEnc*(op: cint, a, b, c: MetalBufferHandle, n: int): cint {.
    importc: "metal_vecop_enc", header: "metal_shim.h".}

  proc metalSessionActivationEnc*(op: cint, x, y: MetalBufferHandle, n: int): cint {.
    importc: "metal_activation_enc", header: "metal_shim.h".}

  proc metalSessionRopeEnc*(x, cosT, sinT: MetalBufferHandle, T, Hn, D: cint): cint {.
    importc: "metal_rope_enc", header: "metal_shim.h".}

type
  GpuBackend* = enum
    gbAuto, gbCpu, gbCuda, gbMetal, gbVulkan

# ─────────────────────────────────────────────────────────────
# GPU FAILURE HANDLING: when a GPU kernel fails (OOM, driver error, etc.),
# the framework will automatically fall back to CPU for that operation,
# log a warning, and then attempt GPU again on the next operation.
# This allows training to continue on memory-constrained GPUs without
# manual intervention.
#
# Set gForbidCpuFallback = true ONLY for benchmarking / CI where you want
# a hard failure instead of silent CPU fallback.
var gForbidCpuFallback* = false

proc setForbidCpuFallback*(forbid: bool) =
  gForbidCpuFallback = forbid

proc handleGpuFailure(backend: GpuBackend, opName: string, e: ref CatchableError) =
  ## Called when a GPU backend fails during an operation.
  ## Always prints a clear traceback (Metal/Vulkan) so OOM / driver errors are diagnosable.
  ## If CPU fallback is allowed, log and return so the caller can retry on CPU.
  let fullMsg = "[GPU] backend '" & $backend & "' failed during '" & opName & "' (" & e.msg & ")"
  stderr.writeLine(fullMsg)
  try:
    let st = getStackTrace(e)
    if st.len > 0:
      stderr.writeLine("[GPU] exception traceback:")
      stderr.writeLine(st)
  except:
    discard
  try:
    let st2 = getStackTrace()
    if st2.len > 0:
      stderr.writeLine("[GPU] call-site traceback:")
      stderr.writeLine(st2)
  except:
    discard
  if gForbidCpuFallback:
    raise newException(CatchableError,
      fullMsg & ". " &
      "CPU fallback is forbidden (gForbidCpuFallback=true). " &
      "Call setForbidCpuFallback(false) to allow automatic CPU fallback.")
  stderr.writeLine("[GPU] falling back to CPU for this operation. Will retry GPU on next operation.")

proc parseBackend*(s: string): GpuBackend =
  case s.strip().toLowerAscii()
  of "cuda": gbVulkan  # CUDA was removed - redirected to Vulkan
  of "metal": gbMetal
  of "cpu": gbCpu
  of "vulkan": gbVulkan
  of "auto": gbAuto
  else: raise newException(ValueError, "unknown BybyLang backend: " & s)

proc normalizeBackend*(b: GpuBackend): GpuBackend =
  ## CUDA is a legacy spelling only. Vulkan does not implicitly use Tensor
  ## Cores; that requires an enabled cooperative-matrix feature and a matching
  ## shader, queried per physical device.
  if b == gbCuda: return gbVulkan
  return b

proc `$`*(b: GpuBackend): string =
  case b
  of gbAuto: "auto"
  of gbCpu: "cpu"
  of gbCuda: "cuda"  # legacy spelling; runtime redirects to Vulkan
  of gbMetal: "metal"
  of gbVulkan: "vulkan"

proc detectBackend*(): GpuBackend =
  ## Detects the best GPU backend available on this machine.
  ## CUDA was REMOVED from the framework - every NVIDIA device now runs through
  ## Vulkan (the NVIDIA driver supports Vulkan + tensor cores). Probe order:
  ## Metal (macOS) -> Vulkan (NVIDIA/AMD/Intel, any OS).
  when defined(macosx):
    if metalAvailable():
      return gbMetal
  if vulkanAvailable():
    return gbVulkan
  return gbCpu

# --- CPU implementations of activation and layers ---
proc cpuRelu(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    result[i] = max(x[i], 0'f32)

proc cpuSigmoid(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    result[i] = 1'f32 / (1'f32 + exp(-x[i]))

proc gpuSqrt*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  ## CPU fallback FORBIDDEN: only real GPUs (CUDA/Metal/Vulkan). If
  ## backend=cpu is chosen explicitly, or a GPU dispatch fails, RAISE
  ## instead of silently computing on the CPU and pretending it ran on the GPU.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("sqrt", x)
  of gbMetal: return metalActivation("sqrt", x)
  of gbVulkan: return vulkanActivation("sqrt", x)
  of gbCpu, gbAuto: return cpu.sqrt(x)

proc gpuExp*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  ## CPU fallback BANNED - see gpuSqrt() above.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("exp", x)
  of gbMetal: return metalActivation("exp", x)
  of gbVulkan: return vulkanActivation("exp", x)
  of gbCpu, gbAuto: return cpu.exp(x)

proc gpuLog*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  ## CPU fallback BANNED - see gpuSqrt() above.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("log", x)
  of gbMetal: return metalActivation("log", x)
  of gbVulkan: return vulkanActivation("log", x)
  of gbCpu, gbAuto: return cpu.log(x)

proc gpuPow*(backend: GpuBackend, x: seq[float32], p: float32): seq[float32] =
  ## CPU fallback BANNED - see gpuSqrt() above.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("pow", x, p)
  of gbMetal: return metalActivation("pow", x, p)
  of gbVulkan: return vulkanActivation("pow", x, p)
  of gbCpu, gbAuto: return cpu.pow(x, p)

proc cpuTanh(x: seq[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    result[i] = tanh(x[i])

proc cpuApflu(x: seq[float32], alpha, beta: float32): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    let val = x[i]
    result[i] = if val > 0'f32: val * (1'f32 + alpha * val) else: beta * val * exp(val)

proc cpuApfluBackward(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
  result = newSeq[float32](x.len)
  for i in 0..<x.len:
    let val = x[i]
    let d = dy[i]
    result[i] = if val > 0'f32: d * (1'f32 + 2'f32 * alpha * val) else: d * beta * exp(val) * (1'f32 + val)

proc cpuLayernormBackward(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  var dx = newSeq[float32](rows * cols)
  var dgamma = newSeq[float32](cols)
  var dbeta = newSeq[float32](cols)
  for r in 0..<rows:
    let off = r * cols
    var mean: float32 = 0
    for c in 0..<cols: mean += x[off + c]
    mean /= float32(cols)
    var varr: float32 = 0
    for c in 0..<cols:
      let diff = x[off + c] - mean
      varr += diff * diff
    varr /= float32(cols)
    let invStd = 1'f32 / sqrt(varr + eps)

    # Calculate parameter gradients
    for c in 0..<cols:
      let norm = (x[off + c] - mean) * invStd
      dgamma[c] += dy[off + c] * norm
      dbeta[c] += dy[off + c]

    # Calculate input gradients (dx)
    var sum1: float32 = 0
    var sum2: float32 = 0
    for c in 0..<cols:
      let grad = dy[off + c] * gamma[c] * invStd
      sum1 += grad
      sum2 += grad * (x[off + c] - mean)
    sum2 *= (-invStd * invStd / float32(cols))
    for c in 0..<cols:
      let term1 = dy[off + c] * gamma[c] * invStd
      let term2 = sum1 / float32(cols)
      # sum2 is already divided by cols (see above): no extra 2/cols - matches layernorm_backward_kernel (Metal)
      let term3 = (x[off + c] - mean) * sum2
      dx[off + c] = term1 - term2 + term3
  return (dx, dgamma, dbeta)

proc cpuAttentionFused(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  var o = newSeq[float32](B * H * S * D)
  var s_matrix = newSeq[float32](B * H * S * S)
  for b in 0..<B:
    for h in 0..<H:
      let base_idx = (b * H + h) * S * D
      let base_s = (b * H + h) * S * S
      for ti in 0..<S:
        var scores = newSeq[float32](S)
        var mx = -1e30'f32
        for tj in 0..ti:
          var dot: float32 = 0
          for d in 0..<D:
            dot += q[base_idx + ti * D + d] * k[base_idx + tj * D + d]
          scores[tj] = dot * scale
          if scores[tj] > mx: mx = scores[tj]
        var sum_exp = 0'f32
        for tj in 0..ti:
          scores[tj] = exp(scores[tj] - mx)
          sum_exp += scores[tj]
        for tj in 0..ti:
          scores[tj] /= sum_exp
          s_matrix[base_s + ti * S + tj] = scores[tj]

        # Output = S @ V
        for d in 0..<D:
          var acc = 0'f32
          for tj in 0..ti:
            acc += scores[tj] * v[base_idx + tj * D + d]
          o[base_idx + ti * D + d] = acc
  return (o, s_matrix)

proc cpuMatmulQ4Ref*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                      M, K, N, groupSize, nGroupsPerRow: int): seq[float32] =
  ## CPU REFERENCE (source of truth) for matmul_q4_kernel - an EXACT
  ## mirror of the matmul_q4_naive algorithm (kernels/vecop_matmul.cl, known
  ## correct, used as the reference). Used to verify GPU kernels (Metal/CUDA/
  ## including the new multi-simdgroup version) before trusting them - especially
  ## important because matmul_q4 is THE weight-reading kernel used in EVERY layer,
  ## so an error here matters far more than in any other kernel of the codebase.
  result = newSeq[float32](M * N)
  let bytesPerRow = (K + 1) div 2
  for m in 0 ..< M:
    for n in 0 ..< N:
      let rowByteOff = n * bytesPerRow
      let rowGroupOff = n * nGroupsPerRow
      var sum = 0'f32
      for k in 0 ..< K:
        let g = if groupSize > 0: k div groupSize else: 0
        let sc = scales[rowGroupOff + g]
        let zp = zeros[rowGroupOff + g]
        let byteVal = wq[rowByteOff + (k shr 1)]
        let nibble = if (k and 1) == 0: (byteVal and 0x0F'u8) else: ((byteVal shr 4) and 0x0F'u8)
        let wval = (float32(nibble) - zp) * sc
        sum += a[m * K + k] * wval
      result[m * N + n] = sum

proc cpuAttentionDecodeGqa*(q, k, v: seq[float32], H, S, D: int, scale: float32): seq[float32] =
  ## CPU REFERENCE (source of truth) for attention_decode_gqa_kernel -
  ## the ALGORITHM MUST match the GPU versions (Metal/CUDA) of the same name 1-1.
  ## Used to:
  ##   (1) fall back when no GPU supports the new kernel (cpu backend, or
  ##       a GPU not rebuilt with the new kernel)
  ##   (2) verify: compare the new GPU kernel with this function on the first tokens,
  ##       see the BYBY_VERIFY_ATTN_DECODE flag in nim_inference.nim
  ## q: [H, D], k/v: [S, D] (shared by H heads), returns o: [H, D]
  result = newSeq[float32](H * D)
  var scores = newSeq[float32](S)
  for h in 0..<H:
    let qBase = h * D
    var mx = -1e30'f32
    for tj in 0..<S:
      var dot = 0'f32
      let kBase = tj * D
      for d in 0..<D:
        dot += q[qBase + d] * k[kBase + d]
      let sc = dot * scale
      scores[tj] = sc
      if sc > mx: mx = sc
    var sumExp = 0'f32
    for tj in 0..<S:
      let e = exp(scores[tj] - mx)
      scores[tj] = e
      sumExp += e
    let invSum = 1'f32 / sumExp
    for tj in 0..<S:
      scores[tj] *= invSum
    for d in 0..<D:
      var acc = 0'f32
      for tj in 0..<S:
        acc += scores[tj] * v[tj * D + d]
      result[qBase + d] = acc

proc cpuAttentionFusedBackward(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  var dq = newSeq[float32](B * H * S * D)
  var dk = newSeq[float32](B * H * S * D)
  var dv = newSeq[float32](B * H * S * D)
  for b in 0..<B:
    for h in 0..<H:
      let base_idx = (b * H + h) * S * D
      let base_s = (b * H + h) * S * S
      for ti in 0..<S:
        var softmaxW = newSeq[float32](ti + 1)
        for tj in 0..ti:
          softmaxW[tj] = s_matrix[base_s + ti * S + tj]

        var dSoftmax = newSeq[float32](ti + 1)
        for tj in 0..ti:
          var dotVal = 0'f32
          for d in 0..<D:
            let dyVal = dy[base_idx + ti * D + d]
            dv[base_idx + tj * D + d] += softmaxW[tj] * dyVal
            dotVal += dyVal * v[base_idx + tj * D + d]
          dSoftmax[tj] = dotVal

        var dotSum = 0'f32
        for tj in 0..ti:
          dotSum += softmaxW[tj] * dSoftmax[tj]

        for tj in 0..ti:
          let dScore = softmaxW[tj] * (dSoftmax[tj] - dotSum) * scale
          for d in 0..<D:
            dq[base_idx + ti * D + d] += dScore * k[base_idx + tj * D + d]
            dk[base_idx + tj * D + d] += dScore * q[base_idx + ti * D + d]
  return (dq, dk, dv)

proc cpuSoftmax(x: seq[float32], rows, cols: int): seq[float32] =
  result = newSeq[float32](rows * cols)
  for r in 0..<rows:
    let off = r * cols
    var maxVal = x[off]
    for c in 1..<cols:
      if x[off + c] > maxVal:
        maxVal = x[off + c]
    var sum: float32 = 0'f32
    for c in 0..<cols:
      let e = exp(x[off + c] - maxVal)
      result[off + c] = e
      sum += e
    for c in 0..<cols:
      result[off + c] /= sum

proc cpuLayernorm(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  result = newSeq[float32](rows * cols)
  for r in 0..<rows:
    let off = r * cols
    var mean: float32 = 0'f32
    for c in 0..<cols:
      mean += x[off + c]
    mean /= float32(cols)
    var varr: float32 = 0'f32
    for c in 0..<cols:
      let diff = x[off + c] - mean
      varr += diff * diff
    varr /= float32(cols)
    let invStd = 1'f32 / sqrt(varr + eps)
    for c in 0..<cols:
      result[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c]

proc cpuEmbeddingLookup(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  let num = indices.len
  result = newSeq[float32](num * dim)
  for i in 0..<num:
    let idx = indices[i]
    if idx >= 0 and idx < vocab:
      for j in 0..<dim:
        result[i * dim + j] = table[idx * dim + j]
    else:
      for j in 0..<dim:
        result[i * dim + j] = 0'f32

proc cpuVecOp(op: string, a, b: seq[float32]): seq[float32] =
  let n = min(a.len, b.len)
  result = newSeq[float32](n)
  case op
  of "add":
    for i in 0..<n: result[i] = a[i] + b[i]
  of "sub":
    for i in 0..<n: result[i] = a[i] - b[i]
  of "mul":
    for i in 0..<n: result[i] = a[i] * b[i]
  of "div":
    for i in 0..<n: result[i] = a[i] / b[i]
  else:
    raise newException(ValueError, "Unknown gpu op: " & op)

proc cpuFusedAddAct*(a, b: seq[float32], act: string): seq[float32] =
  let n = min(a.len, b.len)
  result = newSeq[float32](n)
  case act
  of "relu":
    for i in 0..<n:
      let s = a[i] + b[i]
      result[i] = max(s, 0'f32)
  of "sigmoid":
    for i in 0..<n:
      let s = a[i] + b[i]
      result[i] = 1'f32 / (1'f32 + exp(-s))
  of "tanh":
    for i in 0..<n:
      let s = a[i] + b[i]
      result[i] = tanh(s)
  of "none":
    for i in 0..<n:
      result[i] = a[i] + b[i]
  else:
    raise newException(ValueError, "cpuFusedAddAct: unknown activation: " & act)

proc gpuOp*(op: string, backend: GpuBackend, a, b: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbCuda:
      return vulkanVecOp(op, a, b)
    of gbMetal:
      return metalVecOp(op, a, b)
    of gbVulkan:
      return vulkanVecOp(op, a, b)
    of gbCpu, gbAuto:
      case op
      of "add": return cpu.add(a, b)
      of "sub": return cpu.sub(a, b)
      of "mul": return cpu.mul(a, b)
      of "div": return cpu.divide(a, b)
      else: return cpuVecOp(op, a, b)
  except CatchableError as e:
    handleGpuFailure(chosen, "vecop:" & op, e)
    return cpuVecOp(op, a, b)

# --- Unified GPU entrance points for custom layers/activations ---
proc gpuRelu*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanActivation("relu", x)
    of gbMetal: return metalActivation("relu", x)
    of gbVulkan: return vulkanActivation("relu", x)
    of gbCpu, gbAuto: return cpu.relu(x)
  except CatchableError as e:
    handleGpuFailure(chosen, "relu", e)
    return cpuRelu(x)

proc gpuSigmoid*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanActivation("sigmoid", x)
    of gbMetal: return metalActivation("sigmoid", x)
    of gbVulkan: return vulkanActivation("sigmoid", x)
    of gbCpu, gbAuto: return cpu.sigmoid(x)
  except CatchableError as e:
    handleGpuFailure(chosen, "sigmoid", e)
    return cpuSigmoid(x)

proc gpuTanh*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanActivation("tanh", x)
    of gbMetal: return metalActivation("tanh", x)
    of gbVulkan: return vulkanActivation("tanh", x)
    of gbCpu, gbAuto: return cpu.tanh(x)
  except CatchableError as e:
    handleGpuFailure(chosen, "tanh", e)
    return cpuTanh(x)

# ============================================================
# SỬA: GPU APFLU - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuApflu*(backend: GpuBackend, x: seq[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda:
      return vulkanApflu(x, alpha, beta)
    of gbMetal:
      return metalApflu(x, alpha, beta)
    of gbVulkan:
      return vulkanApflu(x, alpha, beta)
    of gbCpu, gbAuto:
      return cpu.apflu(x, alpha, beta)
  except CatchableError as e:
    handleGpuFailure(chosen, "apflu", e)
    return cpuApflu(x, alpha, beta)

# ============================================================
# SỬA: GPU APFLU BACKWARD - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuApfluBackward*(backend: GpuBackend, x, dy: seq[float32], alpha: float32 = 0.1'f32, beta: float32 = 0.1'f32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda:
      return vulkanApfluBackward(x, dy, alpha, beta)
    of gbMetal:
      return metalApfluBackward(x, dy, alpha, beta)
    of gbVulkan:
      return vulkanApfluBackward(x, dy, alpha, beta)
    of gbCpu, gbAuto:
      return cpu.apfluBackward(x, dy, alpha, beta)
  except CatchableError as e:
    handleGpuFailure(chosen, "apflu_backward", e)
    return cpuApfluBackward(x, dy, alpha, beta)

# ============================================================
# SỬA: GPU LAYERNORM BACKWARD - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuLayernormBackward*(backend: GpuBackend, dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda:
      return vulkanLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbMetal:
      return metalLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbVulkan:
      return vulkanLayernormBackward(dy, x, gamma, beta, rows, cols, eps)
    of gbCpu, gbAuto:
      return cpu.layernormBackward(dy, x, gamma, beta, rows, cols, eps)
  except CatchableError as e:
    handleGpuFailure(chosen, "layernorm_backward", e)
    return cpuLayernormBackward(dy, x, gamma, beta, rows, cols, eps)

# ============================================================
# SỬA: GPU ATTENTION FUSED - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuAttentionFused*(backend: GpuBackend, q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda:
      return vulkanAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbMetal:
      return metalAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbVulkan:
      return vulkanAttentionFused(q, k, v, mask, B, H, S, D, scale)
    of gbCpu, gbAuto:
      let r = cpu.attention(q, k, v, mask, mask.len, B, H, S, D, scale)
      return (r.o, r.sMatrix)
  except CatchableError as e:
    handleGpuFailure(chosen, "attention_fused", e)
    return cpuAttentionFused(q, k, v, mask, B, H, S, D, scale)

# ============================================================
# SỬA: GPU ATTENTION FUSED BACKWARD - CUDA GỌI KERNEL THẬT
# ============================================================
proc gpuAttentionFusedBackward*(backend: GpuBackend, q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda:
      return vulkanAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbMetal:
      return metalAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbVulkan:
      return vulkanAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
    of gbCpu, gbAuto:
      return cpuAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)
  except CatchableError as e:
    handleGpuFailure(chosen, "attention_fused_backward", e)
    return cpuAttentionFusedBackward(q, k, v, s_matrix, dy, B, H, S, D, scale)

proc gpuSoftmax*(backend: GpuBackend, x: seq[float32], rows, cols: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanSoftmax(x, rows, cols)
    of gbMetal: return metalSoftmax(x, rows, cols)
    of gbVulkan: return vulkanSoftmax(x, rows, cols)
    of gbCpu, gbAuto: return cpu.softmax(x, rows, cols)
  except CatchableError as e:
    handleGpuFailure(chosen, "softmax", e)
    return cpuSoftmax(x, rows, cols)

proc gpuLayernorm*(backend: GpuBackend, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanLayernorm(x, gamma, beta, rows, cols, eps)
    of gbMetal: return metalLayernorm(x, gamma, beta, rows, cols, eps)
    of gbVulkan: return vulkanLayernorm(x, gamma, beta, rows, cols, eps)
    of gbCpu, gbAuto: return cpu.layernorm(x, gamma, beta, rows, cols, eps)
  except CatchableError as e:
    handleGpuFailure(chosen, "layernorm", e)
    return cpuLayernorm(x, gamma, beta, rows, cols, eps)

proc gpuEmbeddingLookup*(backend: GpuBackend, table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanEmbeddingLookup(table, indices, vocab, dim)
    of gbMetal: return metalEmbeddingLookup(table, indices, vocab, dim)
    of gbVulkan: return vulkanEmbeddingLookup(table, indices, vocab, dim)
    of gbCpu, gbAuto: return cpu.embeddingLookup(table, indices, vocab, dim)
  except CatchableError as e:
    handleGpuFailure(chosen, "embedding_lookup", e)
    return cpuEmbeddingLookup(table, indices, vocab, dim)

proc cpuMatmul(a, b: seq[float32], m, k, n: int): seq[float32] =
  ## Safe CPU matmul with bounds checking. Raises a clear error if inputs
  ## are invalid instead of returning an empty sequence that would cause
  ## IndexDefect downstream.
  if a.len == 0 or b.len == 0 or m <= 0 or k <= 0 or n <= 0:
    raise newException(ValueError,
      "[cpuMatmul] invalid input: empty seq or zero dims (m=" & $m & ", k=" & $k & ", n=" & $n & ")")
  if a.len < m * k or b.len < k * n:
    raise newException(ValueError,
      "[cpuMatmul] input seq too small for dims: a.len=" & $a.len & " < m*k=" & $(m*k) &
      " or b.len=" & $b.len & " < k*n=" & $(k*n))
  
  const blockSize = 64
  result = newSeq[float32](m * n)
  var ii = 0
  while ii < m:
    let iMax = min(ii + blockSize, m)
    var kk = 0
    while kk < k:
      let kMax = min(kk + blockSize, k)
      var jj = 0
      while jj < n:
        let jMax = min(jj + blockSize, n)
        for i in ii..<iMax:
          let cRow = i * n
          for p in kk..<kMax:
            let aVal = a[i*k + p]
            if aVal == 0'f32: continue
            let bRow = p * n
            for j in jj..<jMax:
              result[cRow + j] += aVal * b[bRow + j]
        jj += blockSize
      kk += blockSize
    ii += blockSize

proc gpuMatmul*(backend: GpuBackend, a, b: seq[float32], m, k, n: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  # Guard empty / zero dims before any GPU alloc (avoids MoltenVK size-0 OOM
  # and IndexDefect on empty seq after a broken fallback).
  if m <= 0 or k <= 0 or n <= 0 or a.len == 0 or b.len == 0:
    stderr.writeLine("[gpuMatmul] empty/zero dims -> CPU zero result dims=(" &
      $m & "," & $k & "," & $n & ") a.len=" & $a.len & " b.len=" & $b.len)
    return newSeq[float32](max(m, 0) * max(n, 0))
  if a.len < m * k or b.len < k * n:
    raise newException(ValueError,
      "gpuMatmul: buffer too small a.len=" & $a.len & " need " & $(m*k) &
      " b.len=" & $b.len & " need " & $(k*n))

  try:
    case chosen
    of gbCuda:
      return vulkanMatmul(a, b, m, k, n)
    of gbMetal:
      return metalMatmul(a, b, m, k, n)
    of gbVulkan:
      return vulkanMatmul(a, b, m, k, n)
    of gbCpu, gbAuto:
      return cpu.matmul(a, b, m, k, n)
  except CatchableError as e:
    handleGpuFailure(chosen, "matmul", e)
    try:
      result = cpuMatmul(a, b, m, k, n)
      if result.len == 0 and m * n > 0:
        stderr.writeLine("[gpuMatmul] WARNING: CPU fallback empty but m*n>0; padding zeros")
        result = newSeq[float32](m * n)
      return result
    except CatchableError as e2:
      stderr.writeLine("[gpuMatmul] CPU fallback also failed: " & e2.msg)
      try: stderr.writeLine(getStackTrace(e2)) except: discard
      return newSeq[float32](m * n)

proc gpuMatmul2*(backend: GpuBackend,
                  a1, b1: seq[float32], m1, k1, n1: int,
                  a2, b2: seq[float32], m2, k2, n2: int):
                  tuple[c1, c2: seq[float32]] =
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbMetal:
      let r = metalMatmul2(a1, b1, m1, k1, n1, a2, b2, m2, k2, n2)
      return (r.c1, r.c2)
    else:
      let c1 = gpuMatmul(chosen, a1, b1, m1, k1, n1)
      let c2 = gpuMatmul(chosen, a2, b2, m2, k2, n2)
      return (c1, c2)
  except CatchableError as e:
    handleGpuFailure(chosen, "matmul2", e)
    let c1 = cpuMatmul(a1, b1, m1, k1, n1)
    let c2 = cpuMatmul(a2, b2, m2, k2, n2)
    if c1.len == 0 or c2.len == 0:
      stderr.writeLine("[gpuMatmul2] WARNING: CPU fallback returned empty result")
    return (c1, c2)

proc cpuDequantMatmulQ4Fallback(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                                 m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
  ## Plain CPU fallback (for CUDA/gbCpu - this backend has NO direct int4
  ## kernel yet, only Metal has matmul_q4_kernel).
  ## Dequantizes with EXACTLY the quant.nim formula (16-value LUT per group)
  ## and calls the normal gpuMatmul - the result matches the
  ## matmul_q4_kernel path on Metal, it just does not save upload
  ## bandwidth (wT still materialised in fp32).
  let bytesPerRow = (k + 1) div 2
  var wT = newSeq[float32](k * n)  # layout [K, N] = transposed, matching gpuMatmul(a[m,k], wT[k,n])
  var lut: array[16, float32]
  for row in 0 ..< n:
    let rowByteOff = row * bytesPerRow
    let rowGroupOff = row * nGroupsPerRow
    var g = -1
    for c in 0 ..< k:
      let gNew = if groupSize > 0: c div groupSize else: 0
      if gNew != g:
        g = gNew
        let sc = scales[rowGroupOff + g]
        let zp = zeros[rowGroupOff + g]
        for kk in 0 ..< 16: lut[kk] = (float32(kk) - zp) * sc
      let byteIdx = rowByteOff + c div 2
      let nibble = if c mod 2 == 0: wq[byteIdx] and 0x0F'u8 else: (wq[byteIdx] shr 4) and 0x0F'u8
      wT[c * n + row] = lut[int(nibble)]
  return cpuMatmul(a, wT, m, k, n)

proc gpuMatmulQ4*(backend: GpuBackend, a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                   m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
  ## CHANGED (item 3 - quantization): avoids dequantizeTensorTransposed() to full fp32
  ## on the CPU + uploading that whole fp32 array on every Linear.forward()
  ## call. All 3 backends (Metal/CUDA/Vulkan) have a real int4 kernel
  ## (metalMatmulQ4 / vulkanMatmulQ4 / vulkanMatmulQ4) - only when
  ## that kernel REALLY fails at run time (module not rebuilt, etc.) does it
  ## fall back to cpuDequantMatmulQ4Fallback through handleGpuFailure (raising directly when
  ## gForbidCpuFallback=true, the file-wide policy).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbMetal:
      return metalMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    of gbCuda:
      try:
        return vulkanMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
      except CatchableError as e:
        handleGpuFailure(chosen, "matmul_q4", e)
        return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    of gbVulkan:
      # ADDED: vulkanMatmulQ4 (kernels/matmul_q4.comp) has a real int4 kernel
      # (mirror of Metal's matmul_q4_kernel_ndst2_paired, see the notes in
      # vulkan_driver.nim) but it was NOT wired in here before - gpuMatmulQ4
      # fell straight into the "else: raise backend not supported" branch although the
      # real kernel existed.
      try:
        return vulkanMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
      except CatchableError as e:
        handleGpuFailure(chosen, "matmul_q4", e)
        return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    of gbCpu, gbAuto:
      return cpu.matmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    else:
      raise newException(ValueError, "gpuMatmulQ4: backend " & $chosen & " does not support matmul_q4 yet")
  except CatchableError as e:
    handleGpuFailure(chosen, "matmul_q4", e)
    return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)

# Global current backend; the generated code sets/reads it (default auto).
var gpuBackendSelected*: GpuBackend = gbAuto

# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# API resident CUDA — REMOVED. All paths redirect to Vulkan.
# Stubs keep binary/source compatibility for callers that still mention cu*.
# ═══════════════════════════════════════════════════════════════════════════

# CudaResidentTensor / cuFree / cuUpload already defined near top of file.

proc cuUploadIndices*(data: seq[int32]): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan session API")
proc cuDownload*(t: CudaResidentTensor): seq[float32] =
  raise newException(CatchableError, "CUDA removed - use Vulkan session API")
proc cuMatmulR*(a, b: CudaResidentTensor, m, k, n: int): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuAddR*(a, b: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuSubR*(a, b: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuMulR*(a, b: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuDivR*(a, b: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuReluR*(x: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuSigmoidR*(x: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuTanhR*(x: CudaResidentTensor): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuSoftmaxR*(x: CudaResidentTensor, rows, cols: int): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuLayernormR*(x, gamma, beta: CudaResidentTensor, rows, cols: int, eps: float32): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuAttentionFusedR*(q, k, v: CudaResidentTensor, B, H, S, D: int, scale: float32): tuple[o, s_matrix: CudaResidentTensor] =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuAttentionDecodeGqaR*(q, k, v: CudaResidentTensor, H, S, D: int, scale: float32): tuple[o, scores: CudaResidentTensor] =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuCopyIntoR*(src, dst: CudaResidentTensor, srcOffsetElems, dstOffsetElems, n: int) =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuRopeR*(x, cosT, sinT: CudaResidentTensor, T, Hn, D: int) =
  raise newException(CatchableError, "CUDA removed - use Vulkan")
proc cuEmbeddingLookupR*(table, indices: CudaResidentTensor, numIdx, vocab, dim: int): CudaResidentTensor =
  raise newException(CatchableError, "CUDA removed - use Vulkan")

proc noop_marker_end_of_cuda_resident_section*() = discard


type
  SessionHandle* = object
    kind*: GpuBackend
    metalHandle*: MetalBufferHandle  # Metal
    cudaHandle*: CudaResidentTensor  # CUDA
    vulkanHandle*: VulkanBufferHandle  # Vulkan
    probeIdentity*: uint64         # Probe-only session identity; survives seq relocation.
    probeBytes*: int               # Known allocation extent for opaque GPU buffers.
    cpuData*: seq[float32]           # CPU host buffer
    cpuIndices*: seq[int32]          # CPU integer buffer (embedding/gather)

proc probeRaw(h: SessionHandle): pointer =
  case h.kind
  of gbMetal: h.metalHandle
  of gbVulkan, gbCuda: cast[pointer](h.vulkanHandle.buf)
  else:
    if h.cpuData.len > 0: cast[pointer](unsafeAddr h.cpuData[0])
    elif h.cpuIndices.len > 0: cast[pointer](unsafeAddr h.cpuIndices[0])
    else: nil

proc probeByteSize*(h: SessionHandle): int =
  case h.kind
  of gbCpu: max(h.cpuData.len, h.cpuIndices.len)*4
  of gbVulkan, gbCuda: int(h.vulkanHandle.sizeBytes)
  else: h.probeBytes

proc probeBufferRef*(h: SessionHandle): string =
  let identity = if h.probeIdentity != 0: h.probeIdentity else: probeIdentity(probeRaw(h))
  probeBuffer(identity, probeByteSize(h))

proc probeValue*(h: SessionHandle): string =
  probeBufferRef(h) & "@0:" & $probeByteSize(h)

proc probeValue*(be: GpuBackend): string =
  ## `$GpuBackend` already yields the canonical lowercase name.
  let name = $normalizeBackend(be)
  probeBackend(name)
  name

proc probeBackendValue*(be: GpuBackend) =
  discard probeValue(be)

proc probeHandleRef*(h: SessionHandle; offsetElems = 0; elems = -1): string =
  ## Buffer reference for a port: identity@byteOffset:byteCount.
  let bytes =
    if elems >= 0: elems*4
    elif offsetElems == 0: probeByteSize(h)
    else: max(0, probeByteSize(h) - offsetElems*4)
  probeBufferRef(h) & "@" & $(offsetElems*4) & ":" & $bytes

proc probeBytesResult*(h: var SessionHandle; bytes: int) =
  h.probeBytes = max(h.probeBytes, bytes)
  discard probeBufferRef(h)

proc probeUploadRecorded*(h: SessionHandle): bool =
  ## True when the backend driver already emitted this upload's transfer.
  let identity = if h.probeIdentity != 0: h.probeIdentity else: probeIdentity(probeRaw(h))
  probeDriverRecorded(identity)

proc probeTensorAttrs*(dtype, tensor: string; shape: openArray[int]): string =
  "dtype=" & dtype & ",tensor=" & probeQuote(tensor) & ",shape=" & probeShape(shape)

proc probePrepare*(h: var SessionHandle; origin = "observed") =
  if h.probeIdentity == 0:
    h.probeIdentity = if origin == "observed": probeIdentity(probeRaw(h)) else: probeNewIdentity()
  probeAlias(probeRaw(h), h.probeIdentity)
  discard probeBuffer(h.probeIdentity, probeByteSize(h), origin)

proc probeRefresh*(h: SessionHandle) =
  if h.probeIdentity != 0: probeAlias(probeRaw(h), h.probeIdentity)
  discard probeBufferRef(h)

proc probeName*(h: SessionHandle; name, dtype: string; shape: openArray[int]; offset = 0; bytes = -1) =
  if unlikely(firProbeOn):
    probeNamed(probeBufferRef(h), name, dtype, shape, offset,
      (if bytes < 0: probeByteSize(h) else: bytes))

proc probeRole*(h: SessionHandle; role: string; offset = 0; bytes = -1) =
  if unlikely(firProbeOn):
    probeRoleRecord(probeBufferRef(h), role, offset,
      (if bytes < 0: probeByteSize(h)-offset else: bytes))

proc probeRoleScalar*(h: SessionHandle; role: string; offset, value: int) =
  if unlikely(firProbeOn): probeScalar(probeBufferRef(h), role, offset, value)

proc probeCode*(h: SessionHandle; words: openArray[uint32]) =
  if unlikely(firProbeOn):
    let identity = if h.probeIdentity != 0: h.probeIdentity else: probeIdentity(probeRaw(h))
    probeCode(identity, words)

proc probeCodeProgram*(h: SessionHandle): string =
  let identity = if h.probeIdentity != 0: h.probeIdentity else: probeIdentity(probeRaw(h))
  if h.kind == gbCpu:
    var words = newSeq[uint32](h.cpuData.len)
    if words.len > 0: copyMem(addr words[0], unsafeAddr h.cpuData[0], words.len*4)
    return probeProgram(words)
  let program = probeCodeProgram(identity)
  if program != "none": return program
  probeEvent("note kind=missing_words api=sessionMicroVmEnc")
  "none"

proc cpuTake(h: SessionHandle; n: int): seq[float32] =
  result = newSeq[float32](max(n, 0))
  let count = min(max(n, 0), h.cpuData.len)
  for i in 0 ..< count: result[i] = h.cpuData[i]

proc cpuTakeIndices(h: SessionHandle; n: int): seq[int32] =
  let count = max(n, 0)
  if h.cpuIndices.len >= count:
    result = h.cpuIndices[0 ..< count]
  else:
    result = newSeq[int32](count)
    let bytes = min(count * sizeof(int32), h.cpuData.len * sizeof(float32))
    if bytes > 0:
      copyMem(addr result[0], unsafeAddr h.cpuData[0], bytes)

# ─────────────────────────────────────────────────────────────
# SHARED SESSION API - one interface for every backend
# ─────────────────────────────────────────────────────────────

proc ensureGpuInit*(backend: GpuBackend): bool =
  ## Initialises the backend's device/queue/library IF NOT YET (Metal: ensureMetalInit, dispatch_once).
  ## REQUIRED before metal_jit_compile: metal_available() (used to PROBE the GPU, prints "found N GPUs")
  ## only picks a local device and does NOT set gDevice -> metal_jit_compile sees gDevice == nil if
  ## no GPU operation ran before (e.g. finetune_lora: warmupMicroVmJit runs before any GPU op).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalInit()
    else:
      return false
  else:
    return true

proc sessionBegin*(backend: GpuBackend): bool {.probed.} =
  ## Starts a session - creates the command buffer for the chosen backend
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionBegin()
    else:
      return false
  of gbCuda:
    # CUDA session = the same stream (already in cuda_driver)
    # Nothing special to do, just mark it
    return true
  of gbVulkan:
    return vulkanSessionBegin()
  of gbCpu:
    return true
  else:
    return false

proc sessionMicroVmEnc*(backend: GpuBackend, code, mem: SessionHandle,
                        nThreads, tgFloats, tgSize: int): bool {.probed.} =
  ## Generic bytecode VM dispatch. Vulkan uses its SPIR-V interpreter; Metal
  ## continues through its encoder/JIT path in mamba2_ssm_gpu.nim.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbVulkan:
    return vulkanSessionMicroVmEnc(code.vulkanHandle, mem.vulkanHandle,
      nThreads, tgFloats, tgSize)
  if chosen == gbCpu:
    var words = newSeq[uint32](code.cpuData.len)
    if words.len > 0:
      copyMem(addr words[0], unsafeAddr code.cpuData[0], words.len * sizeof(uint32))
    var memv = mem.cpuData
    return cpu.vmRun(words, memv, @[], nThreads, tgFloats, tgSize)
  false

proc sessionJitPrewarm*(backend: GpuBackend;
                        jobs: openArray[tuple[name, source: string]]) {.probed.} =
  ## Compile a batch of shaders before the hot path starts. Vulkan shells out
  ## to glslc per kernel, so this runs them concurrently; other backends have
  ## no on-disk shader step and ignore the request.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbVulkan:
    vulkanJitPrewarm(jobs)

proc sessionCooperativeGemmJobs*(backend: GpuBackend):
    seq[tuple[name, source: string]] {.probed.} =
  ## Shaders behind sessionFp4GemmTensorCore / sessionMatmulNTAccelerated, for
  ## sessionJitPrewarm. Empty when no cooperative route is available.
  if normalizeBackend(backend) == gbVulkan: vulkanCoopPrewarmJobs() else: @[]

proc sessionCooperativeGemmEnabled*(backend: GpuBackend): bool {.probed.} =
  ## True when FP4/FP32 prefill GEMMs will try a tensor-core route first.
  normalizeBackend(backend) == gbVulkan and vulkanCooperativeGemmEnabled()

proc sessionTensorCoreRequired*(backend: GpuBackend): bool {.probed.} =
  ## True when every prefill GEMM (M > 1) must run on the tensor cores
  ## (Vulkan dGPU with KHR_cooperative_matrix, see BYBY_VULKAN_TENSOR_CORES).
  normalizeBackend(backend) == gbVulkan and vulkanTensorCoreRequired()

proc sessionJitMicroVmEnc*(backend: GpuBackend, kernelName, source: string,
                           mem: SessionHandle, nThreads, tgSize: int; tgFloats = 1024): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbVulkan:
    return vulkanSessionJitMicroVmEnc(kernelName, source, mem.vulkanHandle,
      nThreads, tgSize, tgFloats)
  false

proc sessionUpload*(backend: GpuBackend, data: seq[float32]): SessionHandle {.probed.} =
  ## Uploads data to the GPU in the current session
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      result.metalHandle = metalSessionUpload(data)
    else:
      result.metalHandle = nil
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    result.vulkanHandle = vulkanSessionUpload(data)
  of gbCpu:
    result.cpuData = data
  else:
    discard

proc sessionUploadBorrowed*(backend: GpuBackend, data: var seq[float32]): SessionHandle {.probed.} =
  ## Session-scoped upload whose host sequence must remain alive through
  ## sessionRead/sessionFree. Metal can bind aligned Shared host storage
  ## without a second full-size copy; other backends retain their normal path.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx): result.metalHandle = metalSessionUploadBorrowed(data)
    else: result.metalHandle = nil
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    result.vulkanHandle = vulkanSessionUpload(data)
  of gbCpu:
    result.cpuData = data
  else:
    discard

proc sessionUploadPrivate*(backend: GpuBackend, data: openArray[float32]): SessionHandle {.probed.} =
  ## Private-storage variant of sessionUpload - ONLY for RESIDENT weights
  ## (uploaded ONCE when the model loads, never read/written by the CPU afterwards) - currently used
  ## by beResidentUploadFp32 (int8/int4-symmetric/tied-embedding weights through
  ## uploadResident in nimformer.nim). Metal: MTLResourceStorageModePrivate.
  ## CUDA: plain alias of cuUpload - CUDA VRAM is already separate from the CPU
  ## (no Shared/Private distinction like Metal UMA), so there is
  ## nothing more to optimise here.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      result.metalHandle = metalUploadPrivate(data)
    else:
      result.metalHandle = nil
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    # One staging upload to device-local VRAM on discrete GPUs. UMA keeps
    # host-visible coherent memory; both paths return a real resident handle.
    result.vulkanHandle = vulkanSessionUploadPrivate(data)
  of gbCpu:
    result.cpuData = @data
  else:
    discard

proc sessionUploadIndices*(backend: GpuBackend, data: seq[int32]): SessionHandle {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbCuda:
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  ## Used to be a STUB (returned an empty handle on EVERY backend -> nil buffer on use). Now: bit-casts int32
  ## -> float32 (keeping all 32 bits) and uploads like a normal buffer; shaders read it back with floatBitsToInt /
  ## as_type<int>. Same 4 bytes/element contract as sessionUpload.
  var bits = newSeq[float32](data.len)
  if data.len > 0:
    copyMem(addr bits[0], unsafeAddr data[0], data.len * sizeof(int32))
  result = sessionUpload(chosen, bits)
  if chosen == gbCpu:
    result.cpuIndices = data

proc cuAllocScratch*(n: int): CudaResidentTensor =
  ## CUDA removed — stub.
  raise newException(CatchableError, "CUDA removed - use Vulkan sessionAllocScratch")

proc sessionAllocHeap*(backend: GpuBackend, totalElems: int): pointer {.probed.} =
  ## NEW: allocates one large region ONCE (device level) so that
  ## sessionAllocFromHeap() can suballocate many CHEAP child buffers from it - for
  ## cases needing many buffers of the same size/lifetime (typically the resident
  ## KV, see ensureKvResident in nim_inference.nim). ONLY Metal has a
  ## real implementation (MTLHeap) - the performance bug was the Intel Iris
  ## driver being slow with MANY separate device allocations; CUDA/Vulkan drivers
  ## do not have that severe problem, so it is not applied there (returns nil,
  ## the caller must fall back to one sessionAllocScratch per buffer as before when nil).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalAllocHeap(totalElems * sizeof(float32))
    else:
      return nil
  of gbCpu:
    return nil
  else:
    return nil

proc sessionAllocFromHeap*(backend: GpuBackend, heap: pointer, n: int): SessionHandle {.probed.} =
  ## Suballocates one child buffer (n float32 elements) from a heap allocated with
  ## sessionAllocHeap(). If heap=nil (backend without MTLHeap), returns an
  ## "empty" handle (metalHandle=nil) - the caller MUST check it and fall back
  ## to a normal sessionAllocScratch(backend, n); it must NOT be treated as a
  ## valid buffer.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      if heap == nil:
        result.metalHandle = nil
      else:
        result.metalHandle = metalAllocFromHeap(heap, n)
    else:
      result.metalHandle = nil
  of gbCpu:
    if heap != nil:
      result.cpuData = newSeq[float32](max(n, 0))
  else:
    discard

proc sessionFreeHeap*(backend: GpuBackend, heap: var pointer) {.probed.} =
  ## Frees the heap - call ONLY when ALL child buffers suballocated from it
  ## are no longer used (like ARC MTLHeap: child buffers hold their own __bridge_retained,
  ## so freeing the heap does not destroy them, but NO more suballocation is possible
  ## after freeing it - use when unloading the model/exiting, NOT
  ## while the resident KV is still in use).
  if heap == nil: return
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      metalFreeHeap(heap)
      heap = nil
  else:
    discard

proc sessionAllocScratch*(backend: GpuBackend, n: int): SessionHandle {.probed.} =
  ## Allocates an output buffer in the session
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  result.kind = chosen
  case chosen
  of gbMetal:
    when defined(macosx):
      result.metalHandle = metalSessionAllocScratch(n)
    else:
      result.metalHandle = nil
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    result.vulkanHandle = vulkanSessionAllocScratch(n)
  of gbCpu:
    result.cpuData = newSeq[float32](max(n, 0))
  else:
    discard

# ─────────────────────────────────────────────────────────────
# ENCODE OPS VÀO SESSION
# ─────────────────────────────────────────────────────────────

proc sessionMatmul*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool {.probed.} =
  ## Encodes a matmul into the session: C(M x N) = A(M x K) @ B(K x N)
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # BUG FIX: c may have no buffer yet (the codegen only declares
      # 'var hc: SessionHandle' and passes it here without calling
      # sessionAllocScratch first) - unlike CUDA (which allocates below), Metal
      # uses the buffer in c directly, so with c.metalHandle nil
      # the kernel wrote to a NULL buffer and the later sessionRead read NULL and raised
      # "metal_buffer_read failed". Allocate when missing, matching the implicit
      # contract CUDA already had.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * N)
      return metalSessionMatmulEnc(a.metalHandle, b.metalHandle, c.metalHandle, M, K, N)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if c.vulkanHandle.buf == nil:
      c.vulkanHandle = vulkanSessionAllocScratch(M * N)
    return vulkanSessionMatmulEnc(a.vulkanHandle, b.vulkanHandle, c.vulkanHandle, M, K, N)
  of gbCpu:
    c.cpuData = cpu.matmul(cpuTake(a, M * K), cpuTake(b, K * N), M, K, N)
    return true
  else:
    return false

proc sessionMatmulMps*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int;
                       transposeB = false; aOffsetElems = 0; cOffsetElems = 0): bool =
  ## Metal only: C(M x N) = A(M x K) @ B with MPSMatrixMultiplication in the open session. A and C
  ## start aOffsetElems / cOffsetElems floats into their buffers (so ports of one arena need no copies).
  ## B is [K,N] (transposeB = false) or [N,K] (transposeB = true). One row (M = 1) goes to MPS only on
  ## machines that are not Apple Silicon (e.g. Intel Iris), where the native GEMV is slower than MPS;
  ## Apple Silicon keeps its GEMV. Returns false when MPS does not apply (other backend, no MPS,
  ## Apple Silicon decode); the caller then falls back to sessionMatmul / sessionMatmulNT.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx):
    if M < 1 or not metalMpsSupported(): return false
    if M == 1 and (metalIsAppleSilicon() or getEnv("BYBY_METAL_MPS_GEMV", "1") == "0"): return false
    if c.metalHandle == nil:
      c.metalHandle = metalSessionAllocScratch(M * N)
    return metalSessionMatmulMpsAnyEnc(a.metalHandle, b.metalHandle, c.metalHandle, M, K, N,
                                       aOffsetElems, cOffsetElems, transposeB)
  else:
    return false

proc sessionMatmulNT*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int;
                      bOffsetElems = 0): bool {.probed.} =
  ## Encodes matmul-NT into the session: C(M x N) = A(M x K) @ B(N x K)^T.
  ## bOffsetElems: B starts that many floats into its buffer (a weight
  ## wrapped in place by sessionWrapHostRange).
  ## B KEEPS its original [N,K] layout (e.g. the natural K-cache/V-cache [S,D]) -
  ## NO CPU transpose before upload/calling this. Used by attention
  ## decode to remove the O(S*D)/token transpose cost entirely.
  ## If the backend does not support it (returns false) - the caller MUST fall back to
  ## sessionMatmul with a pre-transposed B (the old path); false is NOT an error.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # BUG FIX: see the explanation in sessionMatmul - c needs a
      # buffer before it is used as an output on Metal.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * N)
      return metalSessionMatmulNTEnc(a.metalHandle, b.metalHandle, c.metalHandle, M, K, N,
        bOffsetElems * sizeof(float32))
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if c.vulkanHandle.buf == nil:
      c.vulkanHandle = vulkanSessionAllocScratch(M * N)
    return vulkanSessionMatmulNTEnc(a.vulkanHandle, b.vulkanHandle, c.vulkanHandle, M, K, N,
      bOffsetElems)
  of gbCpu:
    let bLen = N * K
    if bOffsetElems < 0 or bLen < 0 or bOffsetElems > b.cpuData.len or
        bLen > b.cpuData.len - bOffsetElems:
      return false
    c.cpuData = cpu.matmulNT(cpuTake(a, M * K),
      b.cpuData[bOffsetElems ..< bOffsetElems + bLen], M, K, N)
    return true
  else:
    return false

proc sessionMatmulNTAccelerated*(backend: GpuBackend, a, b: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool {.probed.} =
  ## Inference opt-in to FP16 tensor inputs with FP32 accumulation. Ordinary
  ## sessionMatmulNT retains its FP32 contract for training and exact callers.
  if normalizeBackend(backend) == gbVulkan:
    if c.vulkanHandle.buf == nil:
      c = sessionAllocScratch(gbVulkan, M * N)
    if vulkanSessionMatmulNTAcceleratedEnc(a.vulkanHandle, b.vulkanHandle,
        c.vulkanHandle, M, K, N): return true
  sessionMatmulNT(backend, a, b, c, M, K, N)

proc sessionMatmulNTFp16Coop*(backend: GpuBackend, a, b: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool {.probed.} =
  ## Try only the Vulkan FP16-input/FP32-accumulate cooperative matrix path.
  ## False lets inference choose its existing MSL/VM fallback without first
  ## silently executing a second FP32 matmul.
  if normalizeBackend(backend) != gbVulkan: return false
  if c.vulkanHandle.buf == nil:
    c = sessionAllocScratch(gbVulkan, M * N)
  vulkanSessionMatmulNTAcceleratedEnc(a.vulkanHandle, b.vulkanHandle,
    c.vulkanHandle, M, K, N)

proc sessionFp4GemmTensorCore*(backend: GpuBackend, a, w, lut: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool {.probed.} =
  ## False means the caller must use its packed FP4 microkernel.
  if normalizeBackend(backend) != gbVulkan: return false
  if c.vulkanHandle.buf == nil: c = sessionAllocScratch(gbVulkan, M * N)
  vulkanSessionFp4GemmEnc(a.vulkanHandle, w.vulkanHandle, lut.vulkanHandle,
    c.vulkanHandle, M, K, N)

proc sessionFp4GemmMetal*(backend: GpuBackend, a, w, lut: SessionHandle,
    c: var SessionHandle; M, K, N: int; simdgroup: bool): bool {.probed.} =
  ## Packed FP4-KM prefill GEMM on Metal: the register-blocked kernel on every GPU, or the
  ## simdgroup_matrix kernel on an Apple GPU. False means unavailable; use the packed micro-VM tiles.
  when defined(macosx):
    if normalizeBackend(backend) != gbMetal: return false
    if a.metalHandle == nil or w.metalHandle == nil or lut.metalHandle == nil or c.metalHandle == nil:
      return false
    metalSessionFp4GemmEnc(a.metalHandle, w.metalHandle, lut.metalHandle,
      c.metalHandle, M, K, N, simdgroup)
  else:
    false

proc sessionFp4GemmTiledEnc*(backend: GpuBackend, a, w, lut: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool {.probed.} =
  ## Only the tiled FP4 cooperative kernel (for parity checks); false if it
  ## cannot run on this device/shape.
  if normalizeBackend(backend) != gbVulkan: return false
  if c.vulkanHandle.buf == nil: c = sessionAllocScratch(gbVulkan, M * N)
  vulkanSessionFp4GemmTiledEnc(a.vulkanHandle, w.vulkanHandle, lut.vulkanHandle,
    c.vulkanHandle, M, K, N)

proc sessionMatmulNTTiledEnc*(backend: GpuBackend, a, w: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool {.probed.} =
  ## Only the tiled FP32-weight cooperative kernel (for parity checks).
  if normalizeBackend(backend) != gbVulkan: return false
  if c.vulkanHandle.buf == nil: c = sessionAllocScratch(gbVulkan, M * N)
  vulkanSessionMatmulNTTiledEnc(a.vulkanHandle, w.vulkanHandle, c.vulkanHandle, M, K, N)

proc sessionSetF32GemmTiled*(backend: GpuBackend; enabled: bool) {.probed.} =
  ## Let sessionMatmulNTAccelerated / sessionMatmulNTFp16Coop prefer the tiled kernel.
  if normalizeBackend(backend) == gbVulkan: vulkanSetF32GemmTiled(enabled)

proc sessionSetFp4GemmTiled*(backend: GpuBackend; enabled: bool) {.probed.} =
  ## Let sessionFp4GemmTensorCore prefer the tiled kernel (after a parity check).
  if normalizeBackend(backend) == gbVulkan: vulkanSetFp4GemmTiled(enabled)

proc sessionSoftmax*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, rows, cols: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # BUG FIX: see the explanation in sessionMatmul.
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(rows * cols)
      return metalSessionSoftmaxEnc(x.metalHandle, y.metalHandle, rows, cols)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if y.vulkanHandle.buf == nil:
      y.vulkanHandle = vulkanSessionAllocScratch(rows * cols)
    return vulkanSessionSoftmaxEnc(x.vulkanHandle, y.vulkanHandle, rows, cols)
  of gbCpu:
    y.cpuData = cpu.softmax(cpuTake(x, rows * cols), rows, cols)
    return true
  else:
    return false

proc sessionLayernormEnc*(backend: GpuBackend, x, gamma, beta: SessionHandle, y: var SessionHandle,
                           rows, cols: int, eps: float32): bool {.probed.} =
  ## Encodes layernorm into the current session.
  ## Metal: metal_layernorm_enc (already in the shim, now bound to Nim).
  ## CUDA: cuLayernormR already existed and ALREADY launched asynchronously on
  ## the same gStream without syncing - usable directly, only y's old buffer
  ## must be freed (as sessionMatmul/sessionVecOp CUDA do) to avoid leaking VRAM.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      # BUG FIX: see the explanation in sessionMatmul.
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(rows * cols)
      return metalSessionLayernormEnc(x.metalHandle, gamma.metalHandle, beta.metalHandle, y.metalHandle, rows, cols, eps)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if y.vulkanHandle.buf == nil:
      y.vulkanHandle = vulkanSessionAllocScratch(rows * cols)
    return vulkanSessionLayernormEnc(x.vulkanHandle, gamma.vulkanHandle, beta.vulkanHandle, y.vulkanHandle, rows, cols, eps)
  of gbCpu:
    y.cpuData = cpu.layernorm(cpuTake(x, rows * cols), cpuTake(gamma, cols),
                              cpuTake(beta, cols), rows, cols, eps)
    return true
  else:
    return false

proc sessionRmsnormEnc*(backend: GpuBackend, x, gamma: SessionHandle, y: var SessionHandle,
                         rows, cols: int, eps: float32): bool {.probed.} =
  ## Encodes RMSNorm (no mean subtraction, no beta - unlike layernorm) into the current
  ## session. Mirror of sessionLayernormEnc above.
  ## Metal: metalSessionRmsnormEnc (metal_backend.nim - returns false when the build
  ## is not macosx).
  ## CUDA: cudaRmsnormR - if the "rmsnorm_kernel" kernel is not in the loaded .ptx,
  ## getFn raises CatchableError; that error is CAUGHT here and treated as "not
  ## supported" (returns false) so nim_inference.nim falls back to the old CPU path, NO
  ## env var needed to use this fast path.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(rows * cols)
      return metalSessionRmsnormEnc(x.metalHandle, gamma.metalHandle, y.metalHandle, rows, cols, eps)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if y.vulkanHandle.buf == nil:
      y.vulkanHandle = vulkanSessionAllocScratch(rows * cols)
    return vulkanSessionRmsnormEnc(x.vulkanHandle, gamma.vulkanHandle, y.vulkanHandle, rows, cols, eps)
  of gbCpu:
    y.cpuData = cpu.rmsnorm(cpuTake(x, rows * cols), cpuTake(gamma, cols), rows, cols, eps)
    return true
  else:
    return false

proc sessionAttentionFusedEnc*(backend: GpuBackend, q, k, v: SessionHandle, o, sMatrix: var SessionHandle,
                                B, H, S, D: int, scale: float32): bool {.probed.} =
  ## Encodes attention_fused into the current session.
  ## Metal: metal_attention_fused_enc (newly written).
  ## CUDA: cuAttentionFusedR (newly written, mirrors cudaLayernormR - launches async
  ## on gStream without syncing).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      # BUG FIX: see the explanation in sessionMatmul - both outputs (o,
      # sMatrix) need buffers first. o has size B*H*S*D (like
      # q/k/v), sMatrix is the B*H*S*S attention score matrix.
      if o.metalHandle == nil:
        o.metalHandle = metalSessionAllocScratch(B * H * S * D)
      if sMatrix.metalHandle == nil:
        sMatrix.metalHandle = metalSessionAllocScratch(B * H * S * S)
      return metalSessionAttentionFusedEnc(q.metalHandle, k.metalHandle, v.metalHandle,
                                            o.metalHandle, sMatrix.metalHandle, B, H, S, D, scale)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if o.vulkanHandle.buf == nil:
      o.vulkanHandle = vulkanSessionAllocScratch(B * H * S * D)
    if sMatrix.vulkanHandle.buf == nil:
      sMatrix.vulkanHandle = vulkanSessionAllocScratch(B * H * S * S)
    return vulkanSessionAttentionFusedEnc(q.vulkanHandle, k.vulkanHandle, v.vulkanHandle,
                                           o.vulkanHandle, sMatrix.vulkanHandle, B, H, S, D, scale)
  of gbCpu:
    let r = cpu.attention(cpuTake(q, B * H * S * D), cpuTake(k, B * H * S * D),
                          cpuTake(v, B * H * S * D), @[], 0, B, H, S, D, scale)
    o.cpuData = r.o
    sMatrix.cpuData = r.sMatrix
    return true
  else:
    return false

proc sessionAttentionDecodeGqaEnc*(backend: GpuBackend, q, k, v: SessionHandle, o, scores: var SessionHandle,
                                    H, S, D: int, scale: float32): bool {.probed.} =
  ## NEW for single-token decode with GQA - UNLIKE sessionAttentionFusedEnc
  ## q has only H*D elements (one row/head, the new token) and k/v have S*D elements
  ## SHARED by all H heads (one GQA KV-head pair), not B*H*S*D as in
  ## the fused version (prefill-style self-attention, Q and K/V with the same S rows).
  ## Called once per GQA group in forwardDecode, replacing the manual
  ## matmul+softmax+matmul loop.
  ##
  ## Returns `false` when the backend does not support it (Metal app not rebuilt with
  ## the new kernel, CUDA vecop.ptx not rebuilt, or CPU/unknown) -
  ## the caller (forwardDecode) MUST fall back to the old safe path on
  ## false; false must NOT be silently ignored.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if o.metalHandle == nil:
        o.metalHandle = metalSessionAllocScratch(H * D)
      if scores.metalHandle == nil:
        scores.metalHandle = metalSessionAllocScratch(H * S)
      return metalSessionAttentionDecodeGqaEnc(q.metalHandle, k.metalHandle, v.metalHandle,
                                                o.metalHandle, scores.metalHandle, H, S, D, scale)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if o.vulkanHandle.buf == nil:
      o.vulkanHandle = vulkanSessionAllocScratch(H * D)
    if scores.vulkanHandle.buf == nil:
      scores.vulkanHandle = vulkanSessionAllocScratch(H * S)
    return vulkanSessionAttentionDecodeGqaEnc(q.vulkanHandle, k.vulkanHandle, v.vulkanHandle,
                                               o.vulkanHandle, scores.vulkanHandle, H, S, D, scale)
  of gbCpu:
    let qv = cpuTake(q, H * D)
    let kv = cpuTake(k, S * D)
    let vv = cpuTake(v, S * D)
    o.cpuData = cpuAttentionDecodeGqa(qv, kv, vv, H, S, D, scale)
    var sc = newSeq[float32](H * S)
    for h in 0 ..< H:
      var mx = -1e30'f32
      for t in 0 ..< S:
        var dot = 0'f32
        for d in 0 ..< D: dot += qv[h * D + d] * kv[t * D + d]
        sc[h * S + t] = dot * scale
        if sc[h * S + t] > mx: mx = sc[h * S + t]
      var total = 0'f32
      for t in 0 ..< S:
        let e = exp(sc[h * S + t] - mx)
        sc[h * S + t] = e
        total += e
      for t in 0 ..< S: sc[h * S + t] /= total
    scores.cpuData = sc
    return true
  else:
    return false

proc sessionAttentionDecodeAllHeadsEnc*(backend: GpuBackend, q, kAll, vAll: SessionHandle, o: var SessionHandle,
                                         H, HKV, D, S, maxSeqLen: int): bool {.probed.} =
  ## NEW: FUSED attention decode for all H heads in EXACTLY one dispatch,
  ## replacing the CPU loop "for hkv in 0..<HKV: copyInto+matmulNT+softmax+
  ## matmul+copyInto" (H*5 separate dispatches). kAll/vAll MUST be buffers
  ## CONTIGUOUS per head (kResidentCombined/vResidentCombined in
  ## nim_inference.nim, offset of head hkv = hkv*maxSeqLen*D).
  ##
  ## Supports Metal + Vulkan (CUDA merged into Vulkan) - the caller
  ## (encodeDecodeLayerIntoOpenSession) MUST fall back to the old loop on
  ## false, like every other _enc function in this file - false must
  ## NOT be silently ignored.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if o.metalHandle == nil:
        o.metalHandle = metalSessionAllocScratch(H * D)
      return metalSessionAttentionDecodeAllHeadsEnc(q.metalHandle, kAll.metalHandle, vAll.metalHandle,
                                                      o.metalHandle, H, HKV, D, S, maxSeqLen)
    else:
      return false
  of gbVulkan:
    ## Uses the GLSL shader attention_decode_all_heads.comp (1 dispatch for all heads).
    ## Returns false when it cannot run (D>256, no glslc...) -> the caller falls back to the old path.
    if o.vulkanHandle.buf == nil:
      o.vulkanHandle = vulkanSessionAllocScratch(H * D)
    return vulkanSessionAttentionDecodeAllHeadsEnc(q.vulkanHandle, kAll.vulkanHandle, vAll.vulkanHandle,
                                                    o.vulkanHandle, H, HKV, D, S, maxSeqLen)
  of gbCpu:
    let qv = cpuTake(q, H * D)
    let rep = max(1, H div HKV)
    var outv = newSeq[float32](H * D)
    let scale = 1.0'f32 / sqrt(float32(D))
    for h in 0 ..< H:
      let hkv = min(HKV - 1, h div rep)
      let kOff = hkv * maxSeqLen * D
      let kv = cpuTake(kAll, kOff + S * D)[kOff ..< kOff + S * D]
      let vv = cpuTake(vAll, kOff + S * D)[kOff ..< kOff + S * D]
      let oh = cpuAttentionDecodeGqa(qv[h * D ..< (h + 1) * D], kv, vv, 1, S, D, scale)
      for d in 0 ..< D: outv[h * D + d] = oh[d]
    o.cpuData = outv
    return true
  else:
    return false

proc sessionPacesCommandBuffers*(backend: GpuBackend): bool {.probed.} =
  ## True when long GPU work should be split into time-bounded command buffers
  ## (the macOS watchdog kills one that runs for seconds on an Intel/AMD Mac,
  ## and Vulkan drivers have timeouts too). False on Apple Silicon, whose
  ## command buffers stay whole, and on the CPU.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return not metalIsAppleGpu()
    else:
      return false
  of gbVulkan:
    return true
  else:
    return false

proc sessionFlashAttentionPrepare*(backend: GpuBackend, D: int): string {.probed.} =
  ## Picks the prefill flash-attention kernel for head dim D; call it OUTSIDE a
  ## session (Vulkan checks its kernels against a reference first). Returns
  ## "coop" (Vulkan tensor cores), "scalar" (Vulkan/Metal FP32), "cpu", or ""
  ## when there is none - the caller then keeps its old attention path.
  ## BYBY_FLASH_ATTENTION=0 turns it off; =fp16 prefers scalar FP16;
  ## =scalar forces scalar FP32.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  let mode = getEnv("BYBY_FLASH_ATTENTION", "auto")
  if mode == "0": return ""
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionFlashAttentionPrepare(D)
    else:
      return ""
  of gbVulkan:
    return vulkanFlashAttentionPrepare(D)
  of gbCpu:
    return "cpu"
  else:
    return ""

proc sessionFlashAttentionPrefillEnc*(backend: GpuBackend, q, kAll, vAll: SessionHandle,
                                      o: var SessionHandle, T, H, HKV, D, pos, maxSeqLen: int,
                                      scale: float32): bool {.probed.} =
  ## Causal prefill flash attention in one dispatch: o[T, H, D] =
  ## softmax(q K^T * scale) V with row t attending cache positions 0 .. pos + t.
  ## q/o position-major [T, H, D]; kAll/vAll [HKV, maxSeqLen, D] already hold
  ## positions 0 .. pos + T - 1 (sessionKvScatterEnc). False = not available,
  ## the caller falls back (sessionFlashAttentionPrepare must have run).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if o.metalHandle == nil:
        o.metalHandle = metalSessionAllocScratch(T * H * D)
      return metalSessionFlashAttentionPrefillEnc(q.metalHandle, kAll.metalHandle, vAll.metalHandle,
                                                  o.metalHandle, T, H, HKV, D, pos, maxSeqLen, scale)
    else:
      return false
  of gbVulkan:
    return vulkanSessionFlashAttentionPrefillEnc(q.vulkanHandle, kAll.vulkanHandle, vAll.vulkanHandle,
                                                 o.vulkanHandle, T, H, HKV, D, pos, maxSeqLen, scale)
  of gbCpu:
    let rep = max(1, H div max(1, HKV))
    let qv = cpuTake(q, T * H * D)
    let kv = cpuTake(kAll, HKV * maxSeqLen * D)
    let vv = cpuTake(vAll, HKV * maxSeqLen * D)
    var outv = newSeq[float32](T * H * D)
    var sc = newSeq[float32](pos + T)
    for t in 0 ..< T:
      for h in 0 ..< H:
        let base = min(h div rep, HKV - 1) * maxSeqLen * D
        let n = pos + t + 1
        var mx = -Inf.float32
        for key in 0 ..< n:
          var s = 0.0'f32
          for d in 0 ..< D: s += qv[(t * H + h) * D + d] * kv[base + key * D + d]
          sc[key] = s * scale
          mx = max(mx, sc[key])
        var total = 0.0'f32
        for key in 0 ..< n:
          sc[key] = exp(sc[key] - mx)
          total += sc[key]
        for d in 0 ..< D:
          var acc = 0.0'f32
          for key in 0 ..< n: acc += sc[key] * vv[base + key * D + d]
          outv[(t * H + h) * D + d] = acc / total
    o.cpuData = outv
    return true
  else:
    return false

proc sessionKvScatterEnc*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                          T, HKV, D, pos, maxSeqLen: int): bool {.probed.} =
  ## KV cache append for T new positions: dst[h, pos + t, :] = src[t, h, :]
  ## (src = the K or V projection [T, HKV, D], dst = [HKV, maxSeqLen, D]).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionKvScatterEnc(src.metalHandle, dst.metalHandle, T, HKV, D, pos, maxSeqLen)
    else:
      return false
  of gbVulkan:
    return vulkanSessionKvScatterEnc(src.vulkanHandle, dst.vulkanHandle, T, HKV, D, pos, maxSeqLen)
  of gbCpu:
    if pos + T > maxSeqLen: return false
    if dst.cpuData.len < HKV * maxSeqLen * D: dst.cpuData.setLen(HKV * maxSeqLen * D)
    for t in 0 ..< T:
      for h in 0 ..< HKV:
        for d in 0 ..< D:
          dst.cpuData[(h * maxSeqLen + pos + t) * D + d] = src.cpuData[(t * HKV + h) * D + d]
    return true
  else:
    return false

proc sessionMamba2SsdPrepare*(backend: GpuBackend, N, D, Q: int): string {.probed.} =
  ## Picks and parity-checks the chunked Mamba2 SSD/FLA prefill pipelines for
  ## (d_state N, head dim D, chunk length Q), outside any session. Returns
  ## "coop" (tensor cores), "scalar" (FP32 tiles) or "" (unavailable; keep the
  ## scan path). Vulkan only.
  if normalizeBackend(backend) != gbVulkan: return ""
  vulkanMamba2SsdPrepare(N, D, Q)

proc sessionMamba2SsdPrefillEnc*(backend: GpuBackend, arena: SessionHandle,
    T, H, G, N, D, Q, C, cDim: int,
    offConv, offDt, offCum, offALog, offDSkip,
    offYPre, offStates, offSloc: int, cont: bool): bool {.probed.} =
  ## Encodes the whole chunked SSD pass (cumsum, FLA intra-chunk, chunk states,
  ## combine, inter-chunk: five dispatches) over one float32 arena, replacing
  ## the Mamba2 sequence scan for this layer. Offsets are float32 indices into
  ## the arena; conv rows [t, cDim] = [x(C) | B(G*N) | C(G*N)], dt [t, H],
  ## yPre [t, C], states [2, H*D, N] (slot 0 = carry in when cont, slot 1 =
  ## final state out), cum [T, H] and Sloc [nC, H, N, D] are scratch. False =
  ## not available (sessionMamba2SsdPrepare must have picked a kernel); the
  ## caller falls back to its scan.
  if normalizeBackend(backend) != gbVulkan: return false
  vulkanSessionMamba2SsdPrefillEnc(arena.vulkanHandle, T, H, G, N, D, Q, C, cDim,
    offConv, offDt, offCum, offALog, offDSkip, offYPre, offStates, offSloc, cont)

proc sessionVecOp*(backend: GpuBackend, op: string, a, b: SessionHandle, c: var SessionHandle, n: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      let opCode =
        case op
        of "add": 0
        of "sub": 1
        of "mul": 2
        of "div": 3
        else: 0
      # BUG FIX: see the explanation in sessionMatmul.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(n)
      return metalSessionVecOpEnc(opCode.cint, a.metalHandle, b.metalHandle, c.metalHandle, n.cint) != 0
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    let opCode =
      case op
      of "add": 0
      of "sub": 1
      of "mul": 2
      of "div": 3
      else: 0
    if c.vulkanHandle.buf == nil:  # no scratch buffer yet -> allocate once
      c.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionVecOpEnc(a.vulkanHandle, b.vulkanHandle, c.vulkanHandle, opCode, n)
  of gbCpu:
    let av = cpuTake(a, n)
    let bv = cpuTake(b, n)
    c.cpuData = case op
      of "add": cpu.add(av, bv)
      of "sub": cpu.sub(av, bv)
      of "mul": cpu.mul(av, bv)
      of "div": cpu.divide(av, bv)
      else: cpu.add(av, bv)
    return true
  else:
    return false

proc sessionRope*(backend: GpuBackend, x: SessionHandle, cosT, sinT: SessionHandle,
                   T, Hn, D: int): bool {.probed.} =
  ## In-place rotate_half RoPE, encoded into the open session. Currently ONLY
  ## implemented on Metal (see metal_rope_enc in metal_shim.m). On
  ## CUDA it returns false on purpose - the caller (fusedNormQKVRoPE) MUST
  ## treat false as "not supported, fall back to the old CPU path", NOT as
  ## done, to avoid garbage tokens from a skipped RoPE on
  ## backends without this kernel.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionRopeEnc(x.metalHandle, cosT.metalHandle, sinT.metalHandle,
                                  T.cint, Hn.cint, D.cint) != 0
    else:
      return false
  of gbVulkan:
    var xv = x
    return vulkanSessionRopeEnc(xv.vulkanHandle, cosT.vulkanHandle, sinT.vulkanHandle, Hn, D, T)
  of gbCpu:
    var xv = x.cpuData
    cpu.rope(xv, cosT.cpuData, sinT.cpuData, T, Hn, D)
    return true
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  else:
    return false

proc sessionScale*(backend: GpuBackend, x: SessionHandle, scalar: float32, n: int): bool {.probed.} =
  ## NEW (BYBY_MERGE_QKV_ATTN_SESSION): in-place scalar multiply (x *= s)
  ## INSIDE the open session. Used to apply attn.scale to Q on the GPU
  ## (instead of reading Q back, scaling on the CPU and uploading again) - required to merge
  ## Session #1 (QKV+RoPE) and Session #2 (attention+oProj+FFN) into one.
  ## ONLY implemented on Metal (metal_scale_enc). Other backends return
  ## false on purpose - the caller MUST treat false as "not supported, fall back to the
  ## old CPU path", the same contract as sessionRope above.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionScaleEnc(x.metalHandle, scalar, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    # NEW: reuses the activation.comp shader (opCode 8 = "scale",
    # y = v*p, next to opCode 7 "pow") through vulkanSessionActivationEnc,
    # passing the SAME buffer handle as input and output -> in-place overwrite
    # (safe because activation.comp is elementwise, each invocation
    # reads/writes exactly its own index, no race between threads).
    # NEEDS: activation.spv rebuilt from the updated activation.comp
    # (glslc kernels/activation.comp -o kernels/activation.spv) before
    # running, otherwise opCode 8 hits the else branch (relu) and gives wrong
    # results instead of a clear error - VERIFY THIS ON REAL HARDWARE.
    var xv = x
    return vulkanSessionActivationEnc(8, xv.vulkanHandle, xv.vulkanHandle, n, scalar)
  of gbCpu:
    var xv = x.cpuData
    cpu.scale(xv, scalar)
    return true
  else:
    return false

proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                       srcOffsetElems, dstOffsetElems, n: int): bool {.probed.} =
  ## dst[dstOffsetElems ..< +n] = src[srcOffsetElems ..< +n], device-to-device,
  ## encoded into the open session.
  ## NEW srcOffsetElems: extracts a slice from a large SOURCE buffer
  ## (e.g. upload H*D elements for all KV heads once, then copy-append each
  ## head's D-element slice into its place in its own resident buffer -
  ## instead of SMALL uploads once per head as before, fewer GPU commands/token).
  ## Real implementations on both Metal and CUDA (Metal: copy_into_kernel + setBuffer
  ## offset; CUDA: cuMemcpyDtoDAsync with both pointers offset) - both
  ## use the NATIVE offset mechanism of the driver/runtime, not a
  ## new math kernel.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionCopyIntoEnc(src.metalHandle, dst.metalHandle, srcOffsetElems, dstOffsetElems, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    return vulkanSessionCopyInto(src.vulkanHandle, dst.vulkanHandle, srcOffsetElems, dstOffsetElems, n)
  of gbCpu:
    let need = dstOffsetElems + n
    if dst.cpuData.len < need: dst.cpuData.setLen(need)
    for i in 0 ..< n:
      dst.cpuData[dstOffsetElems + i] = src.cpuData[srcOffsetElems + i]
    return true
  else:
    return false

proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                       dstOffsetElems, n: int): bool {.probed.} =
  ## Keeps the old overload (srcOffsetElems=0) - callers that already worked
  ## (e.g. merge oProj) need no change, 100% backward compatible.
  sessionCopyInto(backend, src, dst, 0, dstOffsetElems, n)

proc sessionActivation*(backend: GpuBackend, op: string, x: SessionHandle, y: var SessionHandle, n: int, p: float32 = 0.0'f32): bool {.probed.} =
  ## CHANGED: added parameter p (default 0.0, old call sites unchanged) + more
  ## opCodes for sqrt/exp/log/pow (4/5/6/7) - before, only
  ## relu/sigmoid/tanh/apflu (0-3) were mapped, so "gpu sqrt/exp/log/pow" in a
  ## session always fell back to immediate (see the traceWarn in bybylang.nim).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  let opCode =
    case op
    of "relu": 0
    of "sigmoid": 1
    of "tanh": 2
    of "apflu": 3
    of "exp": 4
    of "log": 5
    of "sqrt": 6
    of "pow": 7
    else: 0

  case chosen
  of gbMetal:
    when defined(macosx):
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(n)
      return metalSessionActivationPEnc(opCode, x.metalHandle, y.metalHandle, n, p)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if y.vulkanHandle.buf == nil:
      y.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionActivationEnc(opCode, x.vulkanHandle, y.vulkanHandle, n, p)
  of gbCpu:
    let xv = cpuTake(x, n)
    y.cpuData = case opCode
      of 0: cpu.relu(xv)
      of 1: cpu.sigmoid(xv)
      of 2: cpu.tanh(xv)
      of 3: cpu.apflu(xv, 0.1'f32, 1.0'f32)
      of 4: cpu.exp(xv)
      of 5: cpu.log(xv)
      of 6: cpu.sqrt(xv)
      of 7: cpu.pow(xv, p)
      else: cpu.relu(xv)
    return true
  else:
    return false

proc sessionApfluEnc*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, n: int,
                       alpha, beta: float32): bool {.probed.} =
  ## APFLU forward through a session. Metal: the kernel currently supports ONLY
  ## alpha=0.1/beta=1.0 (see the immediate metalApflu) - returns false otherwise
  ## so the caller falls back to immediate (correct, no silent error).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if alpha != 0.1'f32 or beta != 1.0'f32: return false
      if y.metalHandle == nil:
        y.metalHandle = metalSessionAllocScratch(n)
      return metalSessionActivationPEnc(3, x.metalHandle, y.metalHandle, n, 0.0'f32)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if y.vulkanHandle.buf == nil: y.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionApfluEnc(x.vulkanHandle, y.vulkanHandle, n, alpha, beta)
  of gbCpu:
    y.cpuData = cpu.apflu(cpuTake(x, n), alpha, beta)
    return true
  else:
    return false

proc sessionApfluBackwardEnc*(backend: GpuBackend, x, dy: SessionHandle, dx: var SessionHandle, n: int,
                               alpha, beta: float32): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if alpha != 0.1'f32 or beta != 1.0'f32: return false
      if dx.metalHandle == nil:
        dx.metalHandle = metalSessionAllocScratch(n)
      return metalSessionActivationBackwardEnc(3, x.metalHandle, dy.metalHandle, dx.metalHandle, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if dx.vulkanHandle.buf == nil: dx.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionApfluBackwardEnc(x.vulkanHandle, dy.vulkanHandle, dx.vulkanHandle, n, alpha, beta)
  of gbCpu:
    dx.cpuData = cpu.apfluBackward(cpuTake(x, n), cpuTake(dy, n), alpha, beta)
    return true
  else:
    return false

proc sessionGatherEnc*(backend: GpuBackend, src, idx: SessionHandle, dst: var SessionHandle, n: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if dst.metalHandle == nil:
        dst.metalHandle = metalSessionAllocScratch(n)
      return metalSessionGatherEnc(src.metalHandle, idx.metalHandle, dst.metalHandle, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if dst.vulkanHandle.buf == nil:
      dst.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionGatherEnc(src.vulkanHandle, idx.vulkanHandle, dst.vulkanHandle, n)
  of gbCpu:
    dst.cpuData = cpu.gather(src.cpuData, cpuTakeIndices(idx, n))
    return true
  else:
    return false

proc sessionCompareEnc*(backend: GpuBackend, a, b: SessionHandle, mask: var SessionHandle, op, n: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if mask.metalHandle == nil:
        mask.metalHandle = metalSessionAllocScratch(n)
      return metalSessionCompareEnc(a.metalHandle, b.metalHandle, mask.metalHandle, op, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if mask.vulkanHandle.buf == nil:
      mask.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionCompareEnc(a.vulkanHandle, b.vulkanHandle, mask.vulkanHandle, op, n)
  of gbCpu:
    let r = cpu.compare(cpuTake(a, n), cpuTake(b, n), op)
    mask.cpuData = newSeq[float32](n)
    for i in 0 ..< n: mask.cpuData[i] = (if r[i]: 1.0'f32 else: 0.0'f32)
    return true
  else:
    return false

proc sessionDropoutMaskEnc*(backend: GpuBackend, mask: var SessionHandle, n: int, p: float32, seed: uint32): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  let scale = if p < 1'f32: 1'f32 / (1'f32 - p) else: 0'f32
  case chosen
  of gbMetal:
    when defined(macosx):
      if mask.metalHandle == nil:
        mask.metalHandle = metalSessionAllocScratch(n)
      return metalSessionDropoutMaskEnc(mask.metalHandle, seed, p, scale, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if mask.vulkanHandle.buf == nil:
      mask.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionDropoutMaskEnc(mask.vulkanHandle, seed, p, n)
  of gbCpu:
    mask.cpuData = cpu.dropoutMask(n, p, seed)
    return true
  else:
    return false

proc sessionArgmaxIndexEnc*(backend: GpuBackend, x: SessionHandle, val, idx: var SessionHandle,
                             outer, dimSize, inner: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  let pairs = outer * inner
  case chosen
  of gbMetal:
    when defined(macosx):
      if val.metalHandle == nil: val.metalHandle = metalSessionAllocScratch(pairs)
      if idx.metalHandle == nil: idx.metalHandle = metalSessionAllocScratch(pairs)
      return metalSessionArgmaxEnc(x.metalHandle, val.metalHandle, idx.metalHandle, outer, dimSize, inner)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if val.vulkanHandle.buf == nil: val.vulkanHandle = vulkanSessionAllocScratch(pairs)
    if idx.vulkanHandle.buf == nil: idx.vulkanHandle = vulkanSessionAllocScratch(pairs)
    return vulkanSessionArgmaxIndexEnc(x.vulkanHandle, val.vulkanHandle, idx.vulkanHandle, outer, dimSize, inner)
  of gbCpu:
    let r = cpu.argmaxIndex(cpuTake(x, outer * dimSize * inner), outer, dimSize, inner)
    val.cpuData = r.val
    idx.cpuData = newSeq[float32](r.idx.len)
    for i in 0 ..< r.idx.len: idx.cpuData[i] = float32(r.idx[i])
    return true
  else:
    return false

proc sessionPermuteEnc*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                         newStrides, oldStrides, dimsMap: SessionHandle, ndim, n: int): bool {.probed.} =
  ## newStrides/oldStrides/dimsMap: ALREADY uploaded (sessionUploadIndices) by the
  ## caller - small O(rank) tables, not re-uploaded on each call here.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if dst.metalHandle == nil: dst.metalHandle = metalSessionAllocScratch(n)
      return metalSessionPermuteEnc(src.metalHandle, dst.metalHandle,
        newStrides.metalHandle, oldStrides.metalHandle, dimsMap.metalHandle, ndim, n)
    else:
      return false
  of gbVulkan:
    if dst.vulkanHandle.buf == nil: dst.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionPermuteBufEnc(src.vulkanHandle, newStrides.vulkanHandle,
      oldStrides.vulkanHandle, dimsMap.vulkanHandle, dst.vulkanHandle, ndim, n)
  of gbCpu:
    dst.cpuData = cpu.permute(cpuTake(src, n), cpuTakeIndices(newStrides, ndim),
      cpuTakeIndices(oldStrides, ndim), cpuTakeIndices(dimsMap, ndim), n)
    return true
  else:
    return false

proc sessionSplitSliceEnc*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                            dimTotal, thisSize, offset, inner, n: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if dst.metalHandle == nil: dst.metalHandle = metalSessionAllocScratch(n)
      return metalSessionSplitSliceEnc(src.metalHandle, dst.metalHandle, dimTotal, thisSize, offset, inner, n)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if dst.vulkanHandle.buf == nil: dst.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionSplitSliceEnc(src.vulkanHandle, dst.vulkanHandle, dimTotal, thisSize, offset, inner, n)
  of gbCpu:
    let unit = thisSize * inner
    let outer = if unit > 0: n div unit else: 0
    dst.cpuData = cpu.splitSlice(cpuTake(src, outer * dimTotal * inner),
      dimTotal, thisSize, offset, inner, n)
    return true
  else:
    return false

proc sessionLayernormBackwardEnc*(backend: GpuBackend, dy, x, gamma, beta: SessionHandle,
                                   dx, dgamma, dbeta: var SessionHandle, rows, cols: int, eps: float32): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if dx.metalHandle == nil: dx.metalHandle = metalSessionAllocScratch(rows * cols)
      if dgamma.metalHandle == nil: dgamma.metalHandle = metalSessionAllocScratch(cols)
      if dbeta.metalHandle == nil: dbeta.metalHandle = metalSessionAllocScratch(cols)
      return metalSessionLayernormBackwardEnc(dy.metalHandle, x.metalHandle, gamma.metalHandle, beta.metalHandle,
        dx.metalHandle, dgamma.metalHandle, dbeta.metalHandle, rows, cols, eps)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if dx.vulkanHandle.buf == nil: dx.vulkanHandle = vulkanSessionAllocScratch(rows * cols)
    if dgamma.vulkanHandle.buf == nil: dgamma.vulkanHandle = vulkanSessionAllocScratch(cols)
    if dbeta.vulkanHandle.buf == nil: dbeta.vulkanHandle = vulkanSessionAllocScratch(cols)
    return vulkanSessionLayernormBackwardEnc(dy.vulkanHandle, x.vulkanHandle, gamma.vulkanHandle,
      dx.vulkanHandle, dgamma.vulkanHandle, dbeta.vulkanHandle, rows, cols, eps)
  of gbCpu:
    let r = cpu.layernormBackward(cpuTake(dy, rows * cols), cpuTake(x, rows * cols),
      cpuTake(gamma, cols), cpuTake(beta, cols), rows, cols, eps)
    dx.cpuData = r.dx
    dgamma.cpuData = r.dgamma
    dbeta.cpuData = r.dbeta
    return true
  else:
    return false

proc sessionAttentionFusedBackwardEnc*(backend: GpuBackend, q, k, v, sMatrix, dy: SessionHandle,
                                        dq, dk, dv: var SessionHandle, B, H, S, D: int, scale: float32): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      let qkvLen = B * H * S * D
      if dq.metalHandle == nil: dq.metalHandle = metalSessionAllocScratch(qkvLen)
      if dk.metalHandle == nil: dk.metalHandle = metalSessionAllocScratch(qkvLen)
      if dv.metalHandle == nil: dv.metalHandle = metalSessionAllocScratch(qkvLen)
      return metalSessionAttentionFusedBackwardEnc(q.metalHandle, k.metalHandle, v.metalHandle,
        sMatrix.metalHandle, dy.metalHandle, dq.metalHandle, dk.metalHandle, dv.metalHandle, B, H, S, D, scale)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    let qkvLen = B * H * S * D
    if dq.vulkanHandle.buf == nil: dq.vulkanHandle = vulkanSessionAllocScratch(qkvLen)
    if dk.vulkanHandle.buf == nil: dk.vulkanHandle = vulkanSessionAllocScratch(qkvLen)
    if dv.vulkanHandle.buf == nil: dv.vulkanHandle = vulkanSessionAllocScratch(qkvLen)
    return vulkanSessionAttentionFusedBackwardEnc(q.vulkanHandle, k.vulkanHandle, v.vulkanHandle,
      sMatrix.vulkanHandle, dy.vulkanHandle, dq.vulkanHandle, dk.vulkanHandle, dv.vulkanHandle, B, H, S, D, scale)
  of gbCpu:
    let r = cpuAttentionFusedBackward(cpuTake(q, B * H * S * D), cpuTake(k, B * H * S * D),
      cpuTake(v, B * H * S * D), cpuTake(sMatrix, B * H * S * S),
      cpuTake(dy, B * H * S * D), B, H, S, D, scale)
    dq.cpuData = r.dq
    dk.cpuData = r.dk
    dv.cpuData = r.dv
    return true
  else:
    return false

proc sessionEmbeddingLookupEnc*(backend: GpuBackend, table, indices: SessionHandle, y: var SessionHandle,
                                 vocab, dim, numIndices: int): bool {.probed.} =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      if y.metalHandle == nil: y.metalHandle = metalSessionAllocScratch(numIndices * dim)
      return metalSessionEmbeddingLookupEnc(table.metalHandle, indices.metalHandle, y.metalHandle, vocab, dim, numIndices)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if y.vulkanHandle.buf == nil: y.vulkanHandle = vulkanSessionAllocScratch(numIndices * dim)
    return vulkanSessionEmbeddingLookupEnc(table.vulkanHandle, indices.vulkanHandle, y.vulkanHandle, dim, numIndices)
  of gbCpu:
    y.cpuData = cpu.embeddingLookup(table.cpuData, cpuTakeIndices(indices, numIndices), vocab, dim)
    return true
  else:
    return false

# ─────────────────────────────────────────────────────────────
# KẾT THÚC SESSION - COMMIT + WAIT
# ─────────────────────────────────────────────────────────────

proc sessionEnd*(backend: GpuBackend): bool {.probed.} =
  ## Ends the session - commits all ops and waits
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionEnd()
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    return vulkanSessionEnd()
  of gbCpu:
    return true
  else:
    return false

proc sessionCommitContinue*(backend: GpuBackend): bool {.probed.} =
  ## NEW: "commitAndContinue" (following Apple's MPSCommandBuffer / PyTorch MPS
  ## SyncType::COMMIT_ADAPTIVE) - commits the current buffer WITHOUT WAITING and opens
  ## a new buffer right away to keep encoding in the SAME logical "session". Call periodically
  ## (e.g. every N layers) to bound the size of EACH command buffer,
  ## avoiding a GPU watchdog timeout WITHOUT paying a CPU<->GPU sync each time
  ## (unlike sessionEnd()+sessionBegin() in the middle - slow because that is
  ## a REAL waitUntilCompleted). See the full explanation in
  ## metal_shim.m::metal_session_commit_continue.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionCommitContinue()
    else:
      return false
  of gbVulkan:
    return vulkanSessionCommitContinue()
  of gbCpu:
    return true
  else:
    return false

proc sessionRecoverQueue*(backend: GpuBackend): bool {.probed.} =
  ## NEW: call right after sessionEnd() returns false - recreates the
  ## "poisoned" MTLCommandQueue (see the details in
  ## metalRecoverQueue/metal_recover_queue). WITHOUT this call after a failed
  ## sessionEnd, EVERY later session keeps being rejected
  ## ("Ignored") and sessionRead returns OLD/GARBAGE GPU data (not an error
  ## - the buffer simply is not written by the GPU) - this was the cause
  ## of the "garbage" tokens observed earlier.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalRecoverQueue()
    else:
      return false
  of gbVulkan:
    return vulkanRecoverQueue()
  of gbCpu:
    return true
  else:
    return false

proc sessionGpuSeconds*(backend: GpuBackend): float =
  ## Summed GPU execution time of the session command buffers so far (Metal only,
  ## 0 elsewhere).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbMetal: return metalSessionGpuSeconds()
  0.0

proc sessionGetLastError*(backend: GpuBackend): string {.probed.} =
  ## NEW: the REAL NSError/error description of the last failed sessionEnd()
  ## (see metal_shim.h::metal_get_last_session_error) - call ONLY
  ## right after sessionEnd() returns false, before opening another
  ## session (overwritten by the next metal_session_end). Returns ""
  ## if the backend does not support it (CUDA/Vulkan not wired) or there is no error.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalGetLastSessionError()
    else:
      return ""
  of gbVulkan:
    return vulkanGetLastError()
  of gbCpu:
    return ""
  else:
    return ""

proc sessionRead*(backend: GpuBackend, h: SessionHandle, n: int): seq[float32] {.probed.} =
  ## Reads data from the GPU back to the CPU after sessionEnd
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      return metalSessionRead(h.metalHandle, n)
    else:
      return @[]
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    return vulkanSessionRead(h.vulkanHandle, n)
  of gbCpu:
    result = newSeq[float32](max(n, 0))
    let count = min(n, h.cpuData.len)
    for i in 0 ..< count: result[i] = h.cpuData[i]
    return result
  else:
    return @[]

proc sessionFree*(backend: GpuBackend, h: var SessionHandle) {.probed.} =
  ## Frees a buffer in the session
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      metalSessionFree(h.metalHandle)
    else:
      discard
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    vulkanSessionFree(h.vulkanHandle)
  of gbCpu:
    h.cpuData = @[]
    h.cpuIndices = @[]
  else:
    discard
  h.kind = gbCpu
  h.metalHandle = nil
  h.cudaHandle = CudaResidentTensor(dptr: 0, n: 0)

proc gpuFusedAddAct*(backend: GpuBackend, a, b: seq[float32], act: string): seq[float32] =
  ## BUILT FROM THE SESSION API: fusedAddAct = sessionVecOp("add") + sessionActivation(act).
  ## NO separate new .cu/.metal/.comp kernel for fusedAddAct (none tested
  ## on real hardware) - instead it combines 2 EXISTING kernels that RUN
  ## on all 3 backends (Metal/CUDA/Vulkan) through session encode-in-place:
  ##   1) sessionVecOp(chosen, "add", a, b, c, n)      -- c = a + b (real GPU)
  ##   2) sessionActivation(chosen, act, c, y, n)       -- y = act(c) (real GPU)
  ## in the SAME session (one sessionBegin/sessionEnd), so it is still fused
  ## on the GPU in the sense of "no CPU round trip between the 2 steps", it only differs from
  ## a single kernel by an intermediate buffer (c) in VRAM.
  ##
  ## NO hidden CPU fallback: if the backend is not Metal/CUDA/
  ## Vulkan, or sessionVecOp/sessionActivation/sessionBegin/sessionEnd return
  ## false (unsupported backend / init error), it RAISES - the
  ## gForbidCpuFallback policy of this whole file. To run on the CPU call
  ## cpuFusedAddAct() directly and DELIBERATELY.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  if chosen == gbCpu:
    let actCode = case act
      of "relu": 1
      of "sigmoid": 2
      of "tanh": 3
      else: 0
    return cpu.fusedAddAct(a, b, actCode)
  if chosen notin {gbMetal, gbCuda, gbVulkan}:
    raise newException(ValueError,
      "gpuFusedAddAct: CPU fallback forbidden - no GPU backend available")

  if act notin ["relu", "sigmoid", "tanh", "none"]:
    raise newException(ValueError, "gpuFusedAddAct: unknown activation: " & act)

  let n = min(a.len, b.len)

  if not sessionBegin(chosen):
    raise newException(CatchableError,
      "gpuFusedAddAct: sessionBegin failed on backend " & $chosen)

  var ha = sessionUpload(chosen, a)
  var hb = sessionUpload(chosen, b)
  var hc: SessionHandle
  var hy: SessionHandle
  let needsActivation = act != "none"
  try:
    if not sessionVecOp(chosen, "add", ha, hb, hc, n):
      raise newException(CatchableError,
        "gpuFusedAddAct: sessionVecOp('add') failed / no session support on backend " & $chosen)

    if needsActivation:
      if not sessionActivation(chosen, act, hc, hy, n):
        raise newException(CatchableError,
          "gpuFusedAddAct: sessionActivation('" & act & "') failed / no session support on backend " & $chosen)
    else:
      hy = hc

    if not sessionEnd(chosen):
      raise newException(CatchableError,
        "gpuFusedAddAct: sessionEnd failed on backend " & $chosen)

    result = sessionRead(chosen, hy, n)
  finally:
    sessionFree(chosen, ha)
    sessionFree(chosen, hb)
    if needsActivation:
      sessionFree(chosen, hc)
    sessionFree(chosen, hy)


# ─────────────────────────────────────────────────────────────
# CUDA HELPER - scratch buffer allocation
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# USAGE: Linear++ attention with the SHARED session API
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# RESIDENT WEIGHTS (Q4) - weights uploaded ONCE when the model loads and kept
# on the GPU for the process lifetime. Each forward step uploads only the activation
# (small, changes every token) through a normal SessionHandle and calls sessionMatmulQ4
# with the already-resident weight handle - the weight bytes are NOT touched again.
# SHARED API for Metal (native int4 kernel running directly on the packed bytes) and
# CUDA (NO .cu source for an int4 kernel (only a prebuilt vecop.ptx binary,
# no .cu to edit/rebuild - see cpuDequantMatmulQ4Fallback
# above), so it dequantizes ONCE at upload and keeps an fp32 resident copy
# through a normal sessionUpload - still avoids dequant+upload every
# token, it just does not save 4-bit bandwidth like Metal).
# ─────────────────────────────────────────────────────────────
type
  ResidentWeightHandle* = object
    kind*: GpuBackend
    k*, n*, groupSize*, nGroupsPerRow*: int
    wqOffset*: int                  ## byte position of the packed weights in metalWq/vulkanWq
    metalWq*, metalScales*, metalZeros*: MetalBufferHandle
    vulkanWq*, vulkanScales*, vulkanZeros*: VulkanBufferHandle
    ## CUDA fields kept as stubs for binary/layout compatibility only.
    cudaWq*, cudaScales*, cudaZeros*: CudaResidentTensor
    fallbackFp32*: SessionHandle    ## CPU fallback: weights dequantized once
    cpuWq*: seq[uint8]
    cpuScales*, cpuZeros*: seq[float32]
    cudaQ4Ready*: bool              ## legacy flag; always false (CUDA removed)

proc probeQ4Buffers(w: ResidentWeightHandle): array[3, string] =
  var handles: array[3, SessionHandle]
  for h in handles.mitems: h.kind = w.kind
  handles[0].metalHandle = w.metalWq; handles[1].metalHandle = w.metalScales; handles[2].metalHandle = w.metalZeros
  handles[0].vulkanHandle = w.vulkanWq; handles[1].vulkanHandle = w.vulkanScales; handles[2].vulkanHandle = w.vulkanZeros
  let sizes = [w.n*((w.k+1) div 2), w.n*w.nGroupsPerRow*4, w.n*w.nGroupsPerRow*4]
  for i in 0 .. 2:
    handles[i].probeBytes = sizes[i]
    let raw = if w.kind == gbCpu:
      (case i
       of 0: (if w.cpuWq.len > 0: cast[pointer](unsafeAddr w.cpuWq[0]) else: nil)
       of 1: (if w.cpuScales.len > 0: cast[pointer](unsafeAddr w.cpuScales[0]) else: nil)
       else: (if w.cpuZeros.len > 0: cast[pointer](unsafeAddr w.cpuZeros[0]) else: nil))
      else: probeRaw(handles[i])
    result[i] = probeBuffer(probeIdentity(raw), sizes[i], "upload")

proc probeValue*(w: ResidentWeightHandle): string =
  let b = probeQ4Buffers(w)
  b[0] & "@" & $w.wqOffset & ":" & $(w.n*((w.k+1) div 2))

proc probeWeightRef*(w: ResidentWeightHandle): string = probeValue(w)

proc probeWeightAttrs*(w: ResidentWeightHandle): string =
  "k=" & $w.k & ",n=" & $w.n & ",group=" & $w.groupSize &
    ",groupsPerRow=" & $w.nGroupsPerRow

proc probeName*(w: ResidentWeightHandle; name, dtype: string; shape: openArray[int]) =
  if unlikely(firProbeOn):
    let b = probeQ4Buffers(w)
    probeNamed(b[0], name, dtype, shape, w.wqOffset, w.n*((w.k+1) div 2))
    probeNamed(b[1], name & ".scales", "f32", [w.n, w.nGroupsPerRow], 0, w.n*w.nGroupsPerRow*4)
    probeNamed(b[2], name & ".zeros", "f32", [w.n, w.nGroupsPerRow], 0, w.n*w.nGroupsPerRow*4)

proc probeBytesResult*(w: var ResidentWeightHandle; bytes: int) =
  if unlikely(firProbeOn):
    discard probeQ4Buffers(w)

proc probePrepare*(w: var ResidentWeightHandle; origin = "upload") =
  if unlikely(firProbeOn):
    discard probeQ4Buffers(w)

proc residentPrepareQ4*(w: ResidentWeightHandle; M = 1): bool {.probed.} =
  ## Opt-in preparation from inference. The ordinary Q4 session API reuses
  ## the selected plan; training and unsupported devices retain their kernel.
  if w.kind == gbMetal:
    when defined(macosx):
      return metalPrepareQ4Tuned(w.metalWq, w.metalScales, w.metalZeros,
        w.k, w.n, w.groupSize, w.nGroupsPerRow, M, w.wqOffset)
  if w.kind == gbVulkan:
    return vulkanPrepareQ4Tuned(w.vulkanWq, w.vulkanScales, w.vulkanZeros,
      w.k, w.n, w.groupSize, w.nGroupsPerRow, M, w.wqOffset)
  false

proc residentUploadQ4*(backend: GpuBackend, wq: seq[uint8], scales, zeros: seq[float32],
                        k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle {.probed.} =
  ## Uploads packed int4 weights to the GPU ONCE. Call when the model loads
  ## (loadLlamaWeightsL), keep the result, do NOT call it again per token.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  result.kind = chosen
  result.k = k; result.n = n
  result.groupSize = groupSize; result.nGroupsPerRow = nGroupsPerRow

  case chosen
  of gbMetal:
    when defined(macosx):
      # CHANGED: Private storage (not Shared) - wq/scales/zeros are ONLY
      # read by the GPU every token from now on, the CPU never touches them again. Costs
      # one extra blit through a staging buffer ONLY here (at model load), not
      # every token.
      result.metalWq = metalUploadBytesPrivate(wq)
      result.metalScales = metalUploadPrivate(scales)
      result.metalZeros = metalUploadPrivate(zeros)
    else:
      discard
  of gbVulkan:
    result.vulkanWq = vulkanSessionUploadBytesPrivate(wq)
    result.vulkanScales = vulkanSessionUploadPrivate(scales)
    result.vulkanZeros = vulkanSessionUploadPrivate(zeros)
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbCpu:
    result.cpuWq = wq
    result.cpuScales = scales
    result.cpuZeros = zeros
  else:
    discard

proc sessionWrapHostRange*(backend: GpuBackend; data: pointer; bytes: int): tuple[h: SessionHandle, offset: int] {.probed.} =
  ## Read-only GPU buffer over long-lived host memory (a model mmap) without
  ## copying, on GPUs that share system memory. `data` may sit anywhere in
  ## the mapping: the buffer starts at the enclosing page and `offset` is the
  ## byte position of `data` inside it, for kernels that take an offset. The
  ## handle is empty (nil metal/vulkan buffer) when wrapping is not possible;
  ## see sessionUploadPrivatePtr.
  result.h.kind = normalizeBackend(backend)
  case result.h.kind
  of gbMetal:
    when defined(macosx):
      let r = metalWrapHostRange(data, bytes)
      result.h.metalHandle = r.buf
      result.offset = r.offset
  of gbVulkan:
    let r = vulkanImportHostRange(data, bytes)
    result.h.vulkanHandle = r.h
    result.offset = r.offset
  else: discard

proc sessionUploadPrivatePtr*(backend: GpuBackend; data: pointer; bytes: int): SessionHandle {.probed.} =
  ## GPU-only copy straight from host memory (VRAM on a dGPU), without an
  ## intermediate Nim seq.
  result.kind = normalizeBackend(backend)
  case result.kind
  of gbMetal:
    when defined(macosx): result.metalHandle = metalUploadBytesPrivatePtr(data, bytes)
  of gbVulkan:
    result.vulkanHandle = vulkanSessionUploadBytesPrivate(
      toOpenArray(cast[ptr UncheckedArray[uint8]](data), 0, bytes - 1))
  of gbCpu:
    if bytes < 0 or bytes mod sizeof(float32) != 0:
      raise newException(ValueError, "CPU session upload size must be a non-negative multiple of float32")
    if bytes > 0 and data == nil:
      raise newException(ValueError, "CPU session upload pointer cannot be nil")
    result.cpuData = newSeq[float32](bytes div sizeof(float32))
    if bytes > 0:
      copyMem(addr result.cpuData[0], data, bytes)
  else: discard

proc residentUploadQ4Mapped*(backend: GpuBackend; wq: pointer; wqBytes: int;
                             scales, zeros: openArray[float32];
                             k, n, groupSize, nGroupsPerRow: int): tuple[w: ResidentWeightHandle, zeroCopy: bool] {.probed.} =
  ## residentUploadQ4 for packed weights that live in a model mmap. On GPUs
  ## that share system memory (Apple Silicon, Intel iGPU) the GPU buffer wraps
  ## the pages holding wq in place, at any offset (w.wqOffset); otherwise wq
  ## is copied straight from the mapping (into VRAM on a dGPU), never through
  ## an intermediate Nim seq. The caller keeps the mapping alive until
  ## residentFreeQ4. Scales/zeros are small and always copied.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  var w: ResidentWeightHandle
  w.kind = chosen
  w.k = k; w.n = n
  w.groupSize = groupSize; w.nGroupsPerRow = nGroupsPerRow
  case chosen
  of gbMetal:
    when defined(macosx):
      (w.metalWq, w.wqOffset) = metalWrapHostRange(wq, wqBytes)
      result.zeroCopy = w.metalWq != nil
      if w.metalWq == nil: w.metalWq = metalUploadBytesPrivatePtr(wq, wqBytes)
      w.metalScales = metalUploadPrivate(scales)
      w.metalZeros = metalUploadPrivate(zeros)
  of gbVulkan:
    (w.vulkanWq, w.wqOffset) = vulkanImportHostRange(wq, wqBytes)
    result.zeroCopy = w.vulkanWq.buf != nil
    if w.vulkanWq.buf == nil:
      w.vulkanWq = vulkanSessionUploadBytesPrivate(
        toOpenArray(cast[ptr UncheckedArray[uint8]](wq), 0, wqBytes - 1))
    w.vulkanScales = vulkanSessionUploadPrivate(scales)
    w.vulkanZeros = vulkanSessionUploadPrivate(zeros)
  else:
    var bytes = newSeq[uint8](wqBytes)
    if wqBytes > 0: copyMem(addr bytes[0], wq, wqBytes)
    result.w = residentUploadQ4(chosen, bytes, @scales, @zeros, k, n, groupSize, nGroupsPerRow)
    return
  result.w = w

proc sessionMatmulQ4*(backend: GpuBackend, a: SessionHandle, w: ResidentWeightHandle,
                       c: var SessionHandle, M: int): bool {.probed.} =
  ## Encodes the matmul C(M x N) = A(M x K) @ Wq(K x N) into the open session, using
  ## the resident weights uploaded with residentUploadQ4. K/N/groupSize come from w.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # SEVERE BUG FIX (the source of garbage tokens): c had NO buffer
      # before being used as the output - the same bug fixed in sessionMatmul (see
      # the comment there), but THIS function (used by EVERY Q/K/V/O/gate/up/down
      # projection of a Q4-per-group model) was missing that step. Callers do
      # sessionMatmulQ4(be, x, w, hOut, M) with a default SessionHandle hOut
      # (metalHandle=nil) - the kernel wrote to an invalid buffer -> completely
      # wrong logits WITHOUT an obvious crash -> exactly the garbage-token
      # symptom. int8/fp32 models do NOT use this function (they use the sessionMatmul fixed
      # earlier), so they were not affected.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * w.n)
      # NEW (a REAL BUG found by reading the .metal kernel, see matmul_q4_batched.metal):
      # the original kernel (matmul_q4_kernel_ndst2_paired) dispatches grid Y=M - EACH
      # row of M re-reads ALL of its weights from scratch, independently of the
      # other rows. For M=1 (decode) this is HARMLESS (one row, nothing
      # to reuse). For M>1 (prefill, T tokens at once) it is a SEVERE
      # waste: all the weights are read M TIMES instead of once -
      # which is why prefill (92 tokens) used to be almost as slow as 92 decode
      # steps instead of tens of times faster. The new
      # batch kernel reads the weights EXACTLY once and reuses them for MBLOCK=4 consecutive
      # rows - ONLY when M>1 (the stable M=1 decode path is untouched).
      # If the batch kernel fails (not compiled in an old build, etc.) ->
      # it falls back to the old kernel, no path is left broken.
      if M > 1:
        if metalSessionMatmulQ4BatchedEnc(a.metalHandle, w.metalWq, w.metalScales, w.metalZeros,
                                           c.metalHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow,
                                           w.wqOffset):
          return true
        # fallback: drop down to the old kernel below (no return here)
      return metalSessionMatmulQ4Enc(a.metalHandle, w.metalWq, w.metalScales, w.metalZeros,
                                      c.metalHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow,
                                      w.wqOffset)
    else:
      return false
  of gbVulkan:
    if c.vulkanHandle.buf == nil:
      c.vulkanHandle = vulkanSessionAllocScratch(M * w.n)
    return vulkanSessionMatmulQ4Enc(a.vulkanHandle, w.vulkanWq, w.vulkanScales, w.vulkanZeros,
                                     c.vulkanHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow,
                                     w.wqOffset)
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbCpu:
    if w.cpuWq.len == 0 or w.cpuScales.len == 0 or w.cpuZeros.len == 0:
      return false
    c.cpuData = cpu.matmulQ4(cpuTake(a, M * w.k), w.cpuWq, w.cpuScales, w.cpuZeros,
      M, w.k, w.n, w.groupSize, w.nGroupsPerRow)
    return true
  else:
    return false

proc sessionKvAppendEnc*(backend: GpuBackend, k, v, kAll, vAll: SessionHandle,
                          hkv, d, maxSeq, pos: int): bool =
  ## One dispatch appending the token's K/V for all KV heads into the combined caches
  ## ([head][maxSeq][D]). Metal only; false means the caller uses per-head copies.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx):
    return metalSessionKvAppendEnc(k.metalHandle, v.metalHandle, kAll.metalHandle,
                                   vAll.metalHandle, hkv, d, maxSeq, pos)
  else:
    return false

proc sessionEndAsync*(backend: GpuBackend): bool =
  ## Commits the open session without waiting (Metal only). Pair with sessionWaitOldest.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx): return metalSessionEndAsync()
  else: return false

proc sessionWaitOldest*(backend: GpuBackend): bool =
  ## Waits for the oldest session committed by sessionEndAsync; false if it failed.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx): return metalSessionWaitOldest()
  else: return false

proc sessionTokenRingGet*(backend: GpuBackend, ring: SessionHandle, slot: int): int =
  when defined(macosx): metalTokenRingGet(ring.metalHandle, slot)
  else: 0

proc sessionTokenRingSet*(backend: GpuBackend, ring: SessionHandle, slot, value: int) =
  when defined(macosx): metalTokenRingSet(ring.metalHandle, slot, value)

proc sessionEmbedTokenEnc*(backend: GpuBackend, table, ring: SessionHandle,
                            o: SessionHandle, dim, slot: int): bool =
  ## Encodes o[0..<dim] = table row of the token id in ring[slot] (Metal only).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx): return metalSessionEmbedTokenEnc(table.metalHandle, ring.metalHandle, o.metalHandle, dim, slot)
  else: return false

proc sessionArgmaxTokenEnc*(backend: GpuBackend, logits, ring: SessionHandle,
                             n, slot: int): bool =
  ## Encodes ring[slot] = greedy token of logits[0..<n] (Metal only).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx): return metalSessionArgmaxTokenEnc(logits.metalHandle, ring.metalHandle, n, slot)
  else: return false

proc sessionMatmulQ4Fused*(backend: GpuBackend, mode: int, a: SessionHandle,
                            ws: openArray[ResidentWeightHandle],
                            outs: openArray[SessionHandle],
                            res: SessionHandle, hasRes: bool): bool =
  ## Fused decode GEMV (M=1) on Metal. mode 0: ws.len (1..3) matrices share `a` and each
  ## writes its own outs[i]; mode 1: ws = [gate, up], outs[0] = silu(gate)*up.
  ## `res` is added to the result when hasRes. Returns false when unsupported, so
  ## callers fall back to the unfused sequence. Output handles must already be allocated.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen != gbMetal: return false
  when defined(macosx):
    if ws.len < 1 or ws.len > 3 or outs.len < ws.len or (mode == 1 and ws.len != 2): return false
    if hasRes and res.metalHandle == nil: return false
    var wq, sc, zr, o: array[3, MetalBufferHandle]
    var n, off: array[3, int]
    for i in 0 ..< ws.len:
      if outs[i].metalHandle == nil: return false
      if ws[i].k != ws[0].k or ws[i].groupSize != ws[0].groupSize or
         ws[i].nGroupsPerRow != ws[0].nGroupsPerRow: return false
      wq[i] = ws[i].metalWq; sc[i] = ws[i].metalScales; zr[i] = ws[i].metalZeros
      o[i] = outs[i].metalHandle; n[i] = ws[i].n; off[i] = ws[i].wqOffset
    return metalSessionMatmulQ4FusedEnc(mode, a.metalHandle, wq, sc, zr, o,
      (if hasRes: res.metalHandle else: nil), ws[0].k, ws[0].groupSize, ws[0].nGroupsPerRow, n, off)
  else:
    return false

proc residentFreeQ4*(w: var ResidentWeightHandle) {.probed.} =
  ## Frees resident weights - call only when unloading the model/exiting,
  ## NOT between tokens.
  case w.kind
  of gbMetal:
    when defined(macosx):
      metalSessionFree(w.metalWq)
      metalSessionFree(w.metalScales)
      metalSessionFree(w.metalZeros)
  of gbVulkan:
    vulkanSessionFree(w.vulkanWq); vulkanSessionFree(w.vulkanScales); vulkanSessionFree(w.vulkanZeros)
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  else:
    var f = w.fallbackFp32
    sessionFree(w.kind, f)
    w.fallbackFp32 = f
    w.cpuWq = @[]
    w.cpuScales = @[]
    w.cpuZeros = @[]

proc linearPlusAttentionSession*(backend: GpuBackend, q, k, v: seq[float32],
                                  B, H, T, D: int, scale: float32): seq[float32] =
  ## Linear++ attention through the shared session API - runs on Metal/CUDA
  let qkvLen = B * H * T * D
  result = newSeq[float32](qkvLen)

  if not sessionBegin(backend):
    echo "Session begin failed,stop"
    return

  # Upload the data
  var hQ = sessionUpload(backend, q)
  var hK = sessionUpload(backend, k)
  var hV = sessionUpload(backend, v)
  var hKV = sessionAllocScratch(backend, qkvLen)
  var hO = sessionAllocScratch(backend, qkvLen)

  # Encode the ops into the session
  # Step 1: K^T @ V (matmul for now)
  # TODO: K and V need a transpose first
  # Step 2: Q @ KV
  # Step 3: scale + norm

  # End the session
  if not sessionEnd(backend):
    echo "Session failed,stop"
    return

  # Read the result
  result = sessionRead(backend, hO, qkvLen)

  # Free the buffers
  sessionFree(backend, hQ); sessionFree(backend, hK)
  sessionFree(backend, hV); sessionFree(backend, hKV)
  sessionFree(backend, hO)# gpubackend_additions.nim
# Wired into gpubackend.nim with the same try/case/handleGpuFailure pattern as
# the gpuSqrt/gpuExp above in that file.

proc gpuGather*(backend: GpuBackend, src: seq[float32], idx: seq[int32]): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanGather(src, idx)
    of gbMetal: return metalGather(src, idx)
    of gbVulkan: return vulkanGather(src, idx)
    of gbCpu:
      return cpu.gather(src, idx)
    of gbAuto:
      result = newSeq[float32](idx.len)
      for i in 0 ..< idx.len: result[i] = src[idx[i]]
  except CatchableError as e:
    handleGpuFailure(chosen, "gather", e)
    result = newSeq[float32](idx.len)
    for i in 0 ..< idx.len: result[i] = src[idx[i]]

proc gpuArgmaxIndex*(backend: GpuBackend, x: seq[float32], outer, dimSize, inner: int): tuple[val: seq[float32], idx: seq[int32]] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanArgmaxIndex(x, outer, dimSize, inner)
    of gbMetal: return metalArgmaxIndex(x, outer, dimSize, inner)
    of gbVulkan: return vulkanArgmaxIndex(x, outer, dimSize, inner)
    of gbCpu:
      return cpu.argmaxIndex(x, outer, dimSize, inner)
    of gbAuto: discard # falls through to the CPU version below
  except CatchableError as e:
    handleGpuFailure(chosen, "argmax", e)
  var outVal = newSeq[float32](outer * inner)
  var outIdx = newSeq[int32](outer * inner)
  for o in 0 ..< outer:
    for i in 0 ..< inner:
      var best = x[(o * dimSize) * inner + i]
      var bestIdx = 0
      for d in 1 ..< dimSize:
        let v = x[(o * dimSize + d) * inner + i]
        if v > best: (best = v; bestIdx = d)
      outVal[o * inner + i] = best
      outIdx[o * inner + i] = int32(bestIdx)
  result = (val: outVal, idx: outIdx)

proc gpuCompare*(backend: GpuBackend, a, b: seq[float32], op: int): seq[bool] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  var mask: seq[uint8]
  try:
    case chosen
    of gbCuda, gbVulkan:
      let m32 = vulkanCompare(a, b, op)
      mask = newSeq[uint8](m32.len)
      for i in 0 ..< m32.len: mask[i] = uint8(m32[i])
    of gbMetal: mask = metalCompare(a, b, op)
    of gbCpu:
      return cpu.compare(a, b, op)
    of gbAuto: mask = @[]
  except CatchableError as e:
    handleGpuFailure(chosen, "compare", e)
    mask = @[]
  if mask.len == 0:
    result = newSeq[bool](a.len)
    for i in 0 ..< a.len:
      result[i] = case op
        of 0: a[i] > b[i]
        of 1: a[i] < b[i]
        of 2: a[i] == b[i]
        of 3: a[i] >= b[i]
        else: a[i] <= b[i]
  else:
    result = newSeq[bool](mask.len)
    for i in 0 ..< mask.len: result[i] = mask[i] != 0

proc gpuDropoutMask*(backend: GpuBackend, n: int, p: float32, seed: uint32): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanDropoutMask(n, p, seed)
    of gbMetal: return metalDropoutMask(n, p, seed)
    of gbVulkan: return vulkanDropoutMask(n, p, seed)
    of gbCpu:
      return cpu.dropoutMask(n, p, seed)
    of gbAuto: discard
  except CatchableError as e:
    handleGpuFailure(chosen, "dropout_mask", e)
  # CPU xorshift32 fallback - the SAME formula as the 3 GPU kernels (runs only when
  # backend=cpu explicitly or handleGpuFailure allows falling back).
  result = newSeq[float32](n)
  let scale = if p < 1'f32: 1'f32 / (1'f32 - p) else: 0'f32
  for i in 0 ..< n:
    var state = seed xor (uint32(i) * 2654435761'u32 + 1'u32)
    state = state xor (state shl 13); state = state xor (state shr 17); state = state xor (state shl 5)
    state = state xor (state shl 13); state = state xor (state shr 17); state = state xor (state shl 5)
    let u = float32(state shr 8) * (1'f32 / 16777216'f32)
    result[i] = if u < p: 0'f32 else: scale

# ─────────────────────────────────────────────────────────────
# REAL GPU PORT (round 3) - LIKE TORCH: permute/split/concat dispatch. The CPU
# only prepares a small O(rank)/O(numTensors) table (not O(n)) and calls
# the matching kernel on one of the 3 backends.
# ─────────────────────────────────────────────────────────────

proc gpuPermute*(backend: GpuBackend, src: seq[float32], newStrides, oldStrides, dimsMap: seq[int32], n: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanPermute(src, newStrides, oldStrides, dimsMap, n)
    of gbMetal: return metalPermute(src, newStrides, oldStrides, dimsMap, n)
    of gbVulkan: return vulkanPermute(src, newStrides, oldStrides, dimsMap, n)
    of gbCpu:
      return cpu.permute(src, newStrides, oldStrides, dimsMap, n)
    of gbAuto: discard
  except CatchableError as e:
    handleGpuFailure(chosen, "permute", e)
  # CPU fallback (runs only when backend=cpu explicitly or the fallback is allowed)
  let ndim = newStrides.len
  result = newSeq[float32](n)
  for linear in 0 ..< n:
    var rem = linear
    var oldLinear = 0
    for i in 0 ..< ndim:
      let idxI = rem div int(newStrides[i])
      rem = rem mod int(newStrides[i])
      oldLinear += idxI * int(oldStrides[dimsMap[i]])
    result[linear] = src[oldLinear]

proc gpuSplitSlice*(backend: GpuBackend, src: seq[float32], dimTotal, thisSize, offset, inner, n: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda: return vulkanSplitSlice(src, dimTotal, thisSize, offset, inner, n)
    of gbMetal: return metalSplitSlice(src, dimTotal, thisSize, offset, inner, n)
    of gbVulkan: return vulkanSplitSlice(src, dimTotal, thisSize, offset, inner, n)
    of gbCpu:
      return cpu.splitSlice(src, dimTotal, thisSize, offset, inner, n)
    of gbAuto: discard
  except CatchableError as e:
    handleGpuFailure(chosen, "split_slice", e)
  result = newSeq[float32](n)
  for linear in 0 ..< n:
    let i = linear mod inner
    let rem = linear div inner
    let d = rem mod thisSize
    let o = rem div thisSize
    result[linear] = src[(o * dimTotal + offset + d) * inner + i]

proc gpuConcat*(backend: GpuBackend, flatSrc: seq[float32], srcBase, srcDimSize, srcDimOffset: seq[int32],
                totalDim, inner, n: int): seq[float32] =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  try:
    case chosen
    of gbCuda:
      # CUDA removed — chosen is remapped to gbVulkan before case
      raise newException(CatchableError, "CUDA backend removed; use Vulkan")
    of gbMetal: return metalConcat(flatSrc, srcBase, srcDimSize, srcDimOffset, totalDim, inner, n)
    of gbVulkan: return vulkanConcat(flatSrc, srcBase, srcDimSize, srcDimOffset, totalDim, inner, n)
    of gbCpu:
      return cpu.concat(flatSrc, srcBase, srcDimSize, srcDimOffset, totalDim, inner, n)
    of gbAuto: discard
  except CatchableError as e:
    handleGpuFailure(chosen, "concat", e)
  result = newSeq[float32](n)
  let numTensors = srcBase.len
  for linear in 0 ..< n:
    let i = linear mod inner
    let rem = linear div inner
    let dGlobal = rem mod totalDim
    let o = rem div totalDim
    var ti = 0
    for k in 0 ..< numTensors:
      if dGlobal < int(srcDimOffset[k]) + int(srcDimSize[k]): (ti = k; break)
    let dLocal = dGlobal - int(srcDimOffset[ti])
    result[linear] = flatSrc[int(srcBase[ti]) + (o * int(srcDimSize[ti]) + dLocal) * inner + i]
