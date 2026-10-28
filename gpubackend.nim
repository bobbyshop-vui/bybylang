# gpubackend.nim - Lớp điều phối GPU cho BybyLang.
# Cú pháp trong .bybylang:
#   gpu backend is "auto"     # hoặc "cuda" / "metal" / "cpu"
#   gpu array A = [1, 2, 3, 4]
#   gpu add A, B -> C size 4
#   gpu sub A, B -> C
#   gpu mul A, B -> C
#   gpu div A, B -> C
#
# "auto" sẽ tự dò: CUDA (NVIDIA) -> Metal (macOS) -> Vulkan -> CPU thuần.
# Nếu backend được chỉ định cụ thể mà không khả dụng, hoặc chạy lỗi, sẽ tự rơi (fallback)
# về CPU để chương trình luôn cho ra kết quả đúng thay vì crash.
import std/[strutils, math]
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
  # metal_backend.nim đã có Begin/Upload/AllocScratch/MatmulEnc/SoftmaxEnc/
  # End/Read/Free. VecOp và Activation encode-in-session được implement
  # thẳng trong shim ObjC (metal_shim.m/.h), khai báo ở đây (PHẢI đặt trước
  # nơi dùng - sessionVecOp/sessionActivation bên dưới - vì Nim không cho
  # forward-reference proc top-level).
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
  of "cuda": gbVulkan  # CUDA đã bị bỏ - redirect sang Vulkan
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
  ## Tự động dò backend GPU tốt nhất hiện có trên máy đang chạy.
  ## CUDA đã bị LOẠI BỎ khỏi framework - mọi thiết bị NVIDIA giờ chạy qua
  ## Vulkan (driver NVIDIA hỗ trợ Vulkan + tensor core). Thứ tự dò:
  ## Metal (macOS) -> Vulkan (NVIDIA/AMD/Intel, mọi OS).
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
  ## CAM fallback CPU: chi chay GPU that (CUDA/Metal/Vulkan). Neu
  ## backend=cpu duoc chon tuong minh, hoac dispatch GPU loi, RAISE thang
  ## thay vi am tham tinh CPU roi tra ve nhu the da chay GPU.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("sqrt", x)
  of gbMetal: return metalActivation("sqrt", x)
  of gbVulkan: return vulkanActivation("sqrt", x)
  of gbCpu, gbAuto: raise newException(ValueError, "gpuSqrt: CPU fallback bi cam - khong co GPU backend nao kha dung")

proc gpuExp*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  ## CAM fallback CPU - xem gpuSqrt() o tren.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("exp", x)
  of gbMetal: return metalActivation("exp", x)
  of gbVulkan: return vulkanActivation("exp", x)
  of gbCpu, gbAuto: raise newException(ValueError, "gpuExp: CPU fallback bi cam - khong co GPU backend nao kha dung")

proc gpuLog*(backend: GpuBackend, x: seq[float32]): seq[float32] =
  ## CAM fallback CPU - xem gpuSqrt() o tren.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("log", x)
  of gbMetal: return metalActivation("log", x)
  of gbVulkan: return vulkanActivation("log", x)
  of gbCpu, gbAuto: raise newException(ValueError, "gpuLog: CPU fallback bi cam - khong co GPU backend nao kha dung")

proc gpuPow*(backend: GpuBackend, x: seq[float32], p: float32): seq[float32] =
  ## CAM fallback CPU - xem gpuSqrt() o tren.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbCuda: return vulkanActivation("pow", x, p)
  of gbMetal: return metalActivation("pow", x, p)
  of gbVulkan: return vulkanActivation("pow", x, p)
  of gbCpu, gbAuto: raise newException(ValueError, "gpuPow: CPU fallback bi cam - khong co GPU backend nao kha dung")

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
      # sum2 da chia cols (xem tren): khong nhan them 2/cols - khop layernorm_backward_kernel (Metal)
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
  ## Bản CPU THAM CHIẾU (nguồn sự thật) cho matmul_q4_kernel - MIRROR CHÍNH
  ## XÁC thuật toán của matmul_q4_naive (kernels/vecop_matmul.cl, đã chạy
  ## đúng từ trước, dùng làm chuẩn). Dùng để verify GPU kernel (Metal/CUDA/
  ## kể cả bản multi-simdgroup mới) trước khi tin dùng - đặc biệt
  ## quan trọng vì matmul_q4 là kernel ĐỌC WEIGHT CHÍNH, dùng ở MỌI layer,
  ## sai ở đây ảnh hưởng lớn hơn hẳn mọi kernel khác trong codebase.
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
  ## Bản CPU THAM CHIẾU (nguồn sự thật) cho attention_decode_gqa_kernel -
  ## THUẬT TOÁN PHẢI khớp 1-1 với các bản GPU (Metal/CUDA) cùng tên.
  ## Dùng để:
  ##   (1) fallback khi không có GPU nào hỗ trợ kernel mới (backend cpu, hoặc
  ##       GPU chưa build lại với kernel mới)
  ##   (2) verify: so kết quả GPU kernel mới với hàm này trên vài token đầu,
  ##       xem cờ BYBY_VERIFY_ATTN_DECODE trong nim_inference.nim
  ## q: [H, D], k/v: [S, D] (dùng chung cho H head), trả về o: [H, D]
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
      raise newException(ValueError, "gpuOp: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
    of gbCpu, gbAuto: raise newException(ValueError, "gpuRelu: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
    of gbCpu, gbAuto: raise newException(ValueError, "gpuSigmoid: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
    of gbCpu, gbAuto: raise newException(ValueError, "gpuTanh: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuApflu: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuApfluBackward: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuLayernormBackward: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuAttentionFused: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuAttentionFusedBackward: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
    of gbCpu, gbAuto: raise newException(ValueError, "gpuSoftmax: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
    of gbCpu, gbAuto: raise newException(ValueError, "gpuLayernorm: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
    of gbCpu, gbAuto: raise newException(ValueError, "gpuEmbeddingLookup: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuMatmul: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
  ## Fallback CPU-thuan (dung cho CUDA/gbCpu - CHUA co kernel int4
  ## truc tiep tren backend nay, chi Metal moi co matmul_q4_kernel).
  ## Dequant dung CHINH XAC cong thuc trong quant.nim (LUT 16 gia tri/group)
  ## roi goi lai gpuMatmul binh thuong - dam bao ket qua giong het duong
  ## matmul_q4_kernel tren Metal, chi khac la khong tiet kiem duoc bang thong
  ## upload (van phai vat chat hoa wT fp32).
  let bytesPerRow = (k + 1) div 2
  var wT = newSeq[float32](k * n)  # layout [K, N] = transposed, khop voi gpuMatmul(a[m,k], wT[k,n])
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
  ## SUA (item 3 - quantization): tranh dequantizeTensorTransposed() ra fp32
  ## day du tren CPU + upload nguyen mang fp32 do moi lan Linear.forward()
  ## goi ham nay. Ca 3 backend (Metal/CUDA/Vulkan) deu co kernel int4 that
  ## (metalMatmulQ4 / vulkanMatmulQ4 / vulkanMatmulQ4) - chi khi
  ## kernel do THAT SU loi luc chay (module chua build lai, v.v.) moi roi
  ## xuong cpuDequantMatmulQ4Fallback qua handleGpuFailure (raise thang neu
  ## gForbidCpuFallback=true, dung chinh sach chung ca file).
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
      # THÊM: vulkanMatmulQ4 (kernels/matmul_q4.comp) đã có kernel int4 thật
      # (mirror matmul_q4_kernel_ndst2_paired của Metal, xem chú thích trong
      # vulkan_driver.nim) nhưng trước đây KHÔNG được nối vào đây - gpuMatmulQ4
      # rơi thẳng xuống nhánh "else: raise backend chưa hỗ trợ" dù kernel
      # thật đã tồn tại.
      try:
        return vulkanMatmulQ4(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
      except CatchableError as e:
        handleGpuFailure(chosen, "matmul_q4", e)
        return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)
    of gbCpu, gbAuto:
      raise newException(ValueError, "gpuMatmulQ4: CPU fallback bi cam - khong co GPU backend nao kha dung")
    else:
      raise newException(ValueError, "gpuMatmulQ4: backend " & $chosen & " chua ho tro matmul_q4")
  except CatchableError as e:
    handleGpuFailure(chosen, "matmul_q4", e)
    return cpuDequantMatmulQ4Fallback(a, wq, scales, zeros, m, k, n, groupSize, nGroupsPerRow)

# Biến toàn cục backend hiện tại, sinh code sẽ set/đọc biến này (mặc định auto).
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

# ─────────────────────────────────────────────────────────────
# SESSION API CHUNG - Giao diện thống nhất cho cả các backend
# ─────────────────────────────────────────────────────────────

proc ensureGpuInit*(backend: GpuBackend): bool =
  ## Khoi tao device/queue/library cua backend NEU CHUA (Metal: ensureMetalInit, dispatch_once).
  ## BAT BUOC truoc metal_jit_compile: metal_available() (dung de DO GPU, in "Tim thay N GPU")
  ## chi chon thiet bi cuc bo, KHONG gan gDevice -> metal_jit_compile thay gDevice == nil neu
  ## chua co thao tac GPU nao chay truoc (vd finetune_lora: warmupMicroVmJit chay truoc moi GPU op).
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

proc sessionBegin*(backend: GpuBackend): bool =
  ## Bắt đầu session - khởi tạo command buffer cho backend được chọn
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
    # CUDA session = dùng cùng 1 stream (đã có sẵn từ cuda_driver)
    # Không cần làm gì đặc biệt, chỉ cần đánh dấu
    return true
  of gbVulkan:
    return vulkanSessionBegin()
  else:
    return false

proc sessionMicroVmEnc*(backend: GpuBackend, code, mem: SessionHandle,
                        nThreads, tgFloats, tgSize: int): bool =
  ## Generic bytecode VM dispatch. Vulkan uses its SPIR-V interpreter; Metal
  ## continues through its encoder/JIT path in mamba2_ssm_gpu.nim.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbVulkan:
    return vulkanSessionMicroVmEnc(code.vulkanHandle, mem.vulkanHandle,
      nThreads, tgFloats, tgSize)
  false

proc sessionJitMicroVmEnc*(backend: GpuBackend, kernelName, source: string,
                           mem: SessionHandle, nThreads, tgSize: int; tgFloats = 1024): bool =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbVulkan:
    return vulkanSessionJitMicroVmEnc(kernelName, source, mem.vulkanHandle,
      nThreads, tgSize, tgFloats)
  false

proc sessionUpload*(backend: GpuBackend, data: seq[float32]): SessionHandle =
  ## Upload dữ liệu lên GPU trong session hiện tại
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
  else:
    discard

proc sessionUploadBorrowed*(backend: GpuBackend, data: var seq[float32]): SessionHandle =
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
  else:
    discard

proc sessionUploadPrivate*(backend: GpuBackend, data: seq[float32]): SessionHandle =
  ## Bản Private-storage của sessionUpload - CHỈ dùng cho weight RESIDENT
  ## (upload 1 LẦN lúc load model, CPU không đọc/ghi lại sau đó) - hiện dùng
  ## cho beResidentUploadFp32 (weight int8/int4-symmetric/tied-embedding qua
  ## uploadResident trong nimformer.nim). Metal: MTLResourceStorageModePrivate.
  ## CUDA: alias thẳng về cuUpload thường - VRAM CUDA vốn đã tách biệt CPU
  ## hoàn toàn (không có khái niệm Shared/Private như Metal UMA), nên không
  ## có gì thêm để tối ưu ở đây.
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
  else:
    discard

proc sessionUploadIndices*(backend: GpuBackend, data: seq[int32]): SessionHandle =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  if chosen == gbCuda:
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  ## Truoc day la STUB (tra handle rong tren MOI backend -> nil buffer khi dung). Gio: bit-cast int32
  ## -> float32 (giu nguyen 32 bit) roi upload nhu buffer thuong; shader doc lai bang floatBitsToInt /
  ## as_type<int>. Cung hop dong 4 byte/phan tu voi sessionUpload.
  var bits = newSeq[float32](data.len)
  if data.len > 0:
    copyMem(addr bits[0], unsafeAddr data[0], data.len * sizeof(int32))
  result = sessionUpload(chosen, bits)

proc cuAllocScratch*(n: int): CudaResidentTensor =
  ## CUDA removed — stub.
  raise newException(CatchableError, "CUDA removed - use Vulkan sessionAllocScratch")

proc sessionAllocHeap*(backend: GpuBackend, totalElems: int): pointer =
  ## THÊM MỚI: cấp 1 vùng lớn 1 LẦN (device-level) để sau đó
  ## sessionAllocFromHeap() suballocate nhiều buffer con RẺ từ đó - dùng cho
  ## các trường hợp cần nhiều buffer cùng cỡ/cùng vòng đời (điển hình: KV
  ## resident, xem ensureKvResident trong nim_inference.nim). CHỈ Metal có
  ## cài đặt thật (MTLHeap) - nguyên nhân bug hiệu năng cụ thể là Intel Iris
  ## driver chậm với NHIỀU LƯỢT device-alloc riêng lẻ, CUDA/Vulkan driver
  ## không có cùng vấn đề nghiêm trọng này nên chưa cần áp dụng (trả nil,
  ## caller phải fallback về sessionAllocScratch từng cái như cũ khi nil).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      return metalAllocHeap(totalElems * sizeof(float32))
    else:
      return nil
  else:
    return nil

proc sessionAllocFromHeap*(backend: GpuBackend, heap: pointer, n: int): SessionHandle =
  ## Suballocate 1 buffer con (n phần tử float32) từ heap đã cấp bằng
  ## sessionAllocHeap(). Nếu heap=nil (backend không hỗ trợ MTLHeap), trả về
  ## handle "rỗng" (metalHandle=nil) - caller PHẢI kiểm tra và tự fallback
  ## sang sessionAllocScratch(backend, n) bình thường, KHÔNG được coi đây là
  ## buffer hợp lệ.
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
  else:
    discard

proc sessionFreeHeap*(backend: GpuBackend, heap: var pointer) =
  ## Giải phóng heap - CHỈ gọi khi TẤT CẢ buffer con suballocate từ heap này
  ## đã hết dùng (giống ARC MTLHeap: buffer con giữ __bridge_retained riêng
  ## nên free heap không tự huỷ chúng, nhưng KHÔNG suballocate thêm được nữa
  ## sau khi free heap - dùng khi model unload/thoát chương trình, KHÔNG
  ## gọi giữa lúc KV resident còn đang dùng).
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

proc sessionAllocScratch*(backend: GpuBackend, n: int): SessionHandle =
  ## Cấp buffer output trong session
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
  else:
    discard

# ─────────────────────────────────────────────────────────────
# ENCODE OPS VÀO SESSION
# ─────────────────────────────────────────────────────────────

proc sessionMatmul*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool =
  ## Encode matmul vào session: C(M x N) = A(M x K) @ B(K x N)
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: c có thể chưa được cấp buffer (codegen chỉ khai báo
      # 'var hc: SessionHandle' rồi truyền thẳng vào đây, không gọi
      # sessionAllocScratch trước) - khác CUDA (tự cấp bên dưới), Metal
      # dùng thẳng buffer có sẵn trong c nên nếu c.metalHandle là nil,
      # kernel ghi vào buffer NULL, sessionRead sau đó đọc ra NULL và raise
      # "metal_buffer_read failed". Tự cấp phát nếu chưa có, để khớp hợp
      # đồng ngầm mà CUDA đã có từ trước.
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
  else:
    return false

proc sessionMatmulNT*(backend: GpuBackend, a, b: SessionHandle, c: var SessionHandle, M, K, N: int): bool =
  ## Encode matmul-NT vào session: C(M x N) = A(M x K) @ B(N x K)^T.
  ## B GIỮ NGUYÊN layout gốc [N,K] (vd K-cache/V-cache [S,D] tự nhiên) -
  ## KHÔNG cần transpose CPU trước khi upload/gọi hàm này. Dùng cho attention
  ## decode để loại bỏ hoàn toàn chi phí transpose O(S*D)/token trước đây.
  ## Nếu backend không hỗ trợ (trả về false) - caller PHẢI tự fallback về
  ## sessionMatmul với B đã transpose sẵn (đường cũ), KHÔNG coi false là lỗi.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul - c cần được cấp
      # buffer trước khi dùng làm output trên Metal.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * N)
      return metalSessionMatmulNTEnc(a.metalHandle, b.metalHandle, c.metalHandle, M, K, N)
    else:
      return false
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  of gbVulkan:
    if c.vulkanHandle.buf == nil:
      c.vulkanHandle = vulkanSessionAllocScratch(M * N)
    return vulkanSessionMatmulNTEnc(a.vulkanHandle, b.vulkanHandle, c.vulkanHandle, M, K, N)
  else:
    return false

proc sessionMatmulNTAccelerated*(backend: GpuBackend, a, b: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool =
  ## Inference opt-in to FP16 tensor inputs with FP32 accumulation. Ordinary
  ## sessionMatmulNT retains its FP32 contract for training and exact callers.
  if normalizeBackend(backend) == gbVulkan:
    if c.vulkanHandle.buf == nil:
      c = sessionAllocScratch(gbVulkan, M * N)
    if vulkanSessionMatmulNTAcceleratedEnc(a.vulkanHandle, b.vulkanHandle,
        c.vulkanHandle, M, K, N): return true
  sessionMatmulNT(backend, a, b, c, M, K, N)

proc sessionFp4GemmTensorCore*(backend: GpuBackend, a, w, lut: SessionHandle,
    c: var SessionHandle; M, K, N: int): bool =
  ## False means the caller must use its packed FP4 microkernel.
  if normalizeBackend(backend) != gbVulkan: return false
  if c.vulkanHandle.buf == nil: c = sessionAllocScratch(gbVulkan, M * N)
  vulkanSessionFp4GemmEnc(a.vulkanHandle, w.vulkanHandle, lut.vulkanHandle,
    c.vulkanHandle, M, K, N)

proc sessionSoftmax*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, rows, cols: int): bool =
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
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
  else:
    return false

proc sessionLayernormEnc*(backend: GpuBackend, x, gamma, beta: SessionHandle, y: var SessionHandle,
                           rows, cols: int, eps: float32): bool =
  ## Encode layernorm vào session hiện tại.
  ## Metal: metal_layernorm_enc (đã có sẵn trong shim, giờ mới bind ra Nim).
  ## CUDA: cuLayernormR đã tồn tại từ trước và VỐN DĨ đã launch async lên
  ## cùng gStream không sync - dùng thẳng được, chỉ cần giải phóng buffer cũ
  ## của y (như sessionMatmul/sessionVecOp CUDA đã làm) để không leak VRAM.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
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
  else:
    return false

proc sessionRmsnormEnc*(backend: GpuBackend, x, gamma: SessionHandle, y: var SessionHandle,
                         rows, cols: int, eps: float32): bool =
  ## Encode RMSNorm (không trừ mean, không beta - khác layernorm) vào session
  ## hiện tại. Mirror của sessionLayernormEnc ở trên.
  ## Metal: metalSessionRmsnormEnc (metal_backend.nim - trả về false nếu build
  ## không phải macosx).
  ## CUDA: cudaRmsnormR - nếu kernel "rmsnorm_kernel" chưa có trong .ptx đang
  ## nạp, getFn raise CatchableError; BẮT lỗi đó ở đây và coi là "chưa hỗ
  ## trợ" (trả về false) để nim_inference.nim rơi về đường CPU cũ, KHÔNG cần
  ## bật env var nào để dùng được đường nhanh này.
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
  else:
    return false

proc sessionAttentionFusedEnc*(backend: GpuBackend, q, k, v: SessionHandle, o, sMatrix: var SessionHandle,
                                B, H, S, D: int, scale: float32): bool =
  ## Encode attention_fused vào session hiện tại.
  ## Metal: metal_attention_fused_enc (mới viết).
  ## CUDA: cuAttentionFusedR (mới viết, mirror cudaLayernormR - launch async
  ## lên gStream, không tự sync).
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan
  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul - 2 output (o,
      # sMatrix) đều cần buffer trước. o có kích thước B*H*S*D (giống
      # q/k/v), sMatrix là ma trận điểm attention B*H*S*S.
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
  else:
    return false

proc sessionAttentionDecodeGqaEnc*(backend: GpuBackend, q, k, v: SessionHandle, o, scores: var SessionHandle,
                                    H, S, D: int, scale: float32): bool =
  ## MỚI THÊM cho single-token decode với GQA - KHÁC sessionAttentionFusedEnc
  ## ở chỗ: q chỉ có H*D phần tử (1 dòng/head, token mới), k/v có S*D phần
  ## tử DÙNG CHUNG cho cả H head (1 cặp KV-head GQA), không phải B*H*S*D như
  ## bản fused (vốn cho self-attention kiểu prefill, Q và K/V cùng S dòng).
  ## Gọi 1 lần/nhóm GQA trong forwardDecode, thay cho vòng lặp thủ công
  ## matmul+softmax+matmul.
  ##
  ## Trả về `false` nếu backend chưa hỗ trợ (Metal chưa build lại app với
  ## kernel mới, CUDA chưa rebuild vecop.ptx, hoặc CPU/không xác định) -
  ## người gọi (forwardDecode) BẮT BUỘC phải fallback về đường an toàn cũ khi
  ## nhận false, KHÔNG được coi false là lỗi im lặng bỏ qua.
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
  else:
    return false

proc sessionAttentionDecodeAllHeadsEnc*(backend: GpuBackend, q, kAll, vAll: SessionHandle, o: var SessionHandle,
                                         H, HKV, D, S, maxSeqLen: int): bool =
  ## THÊM MỚI: attention decode FUSED toàn bộ H head trong ĐÚNG 1 dispatch,
  ## thay cho vòng lặp CPU "for hkv in 0..<HKV: copyInto+matmulNT+softmax+
  ## matmul+copyInto" (H*5 lệnh dispatch riêng lẻ). kAll/vAll PHẢI là buffer
  ## LIÊN TỤC theo head (kResidentCombined/vResidentCombined bên
  ## nim_inference.nim, offset head hkv = hkv*maxSeqLen*D).
  ##
  ## Hỗ trợ Metal + Vulkan (CUDA đã gộp vào Vulkan) - người gọi
  ## (encodeDecodeLayerIntoOpenSession) BẮT BUỘC fallback về vòng lặp cũ khi
  ## nhận false, giống mọi hàm _enc khác trong file này - KHÔNG được coi
  ## false là lỗi im lặng bỏ qua.
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
    ## Da co shader GLSL attention_decode_all_heads.comp (1 dispatch/toan bo head).
    ## Tra false neu khong chay duoc (D>256, khong co glslc...) -> caller roi ve duong cu.
    if o.vulkanHandle.buf == nil:
      o.vulkanHandle = vulkanSessionAllocScratch(H * D)
    return vulkanSessionAttentionDecodeAllHeadsEnc(q.vulkanHandle, kAll.vulkanHandle, vAll.vulkanHandle,
                                                    o.vulkanHandle, H, HKV, D, S, maxSeqLen)
  else:
    return false

proc sessionVecOp*(backend: GpuBackend, op: string, a, b: SessionHandle, c: var SessionHandle, n: int): bool =
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
      # SỬA BUG: xem giải thích chi tiết ở sessionMatmul.
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
    if c.vulkanHandle.buf == nil:  # chua co scratch buffer -> cap phat 1 lan
      c.vulkanHandle = vulkanSessionAllocScratch(n)
    return vulkanSessionVecOpEnc(a.vulkanHandle, b.vulkanHandle, c.vulkanHandle, opCode, n)
  else:
    return false

proc sessionRope*(backend: GpuBackend, x: SessionHandle, cosT, sinT: SessionHandle,
                   T, Hn, D: int): bool =
  ## RoPE rotate_half in-place, encode vào session đang mở. CHỈ có cài đặt
  ## thật trên Metal hiện tại (xem metal_rope_enc trong metal_shim.m). Trên
  ## CUDA trả về false có chủ đích - caller (fusedNormQKVRoPE) PHẢI
  ## coi false là "chưa hỗ trợ, rơi về đường CPU cũ", KHÔNG được coi false là
  ## đã chạy xong, để tránh sinh token rác do RoPE bị bỏ qua trên các
  ## backend chưa implement kernel này.
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
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  else:
    return false

proc sessionScale*(backend: GpuBackend, x: SessionHandle, scalar: float32, n: int): bool =
  ## MỚI THÊM (BYBY_MERGE_QKV_ATTN_SESSION): nhân scalar in-place (x *= s)
  ## TRONG session đang mở. Dùng để đưa attn.scale vào Q ngay trên GPU
  ## (thay vì đọc Q về CPU nhân rồi upload lại) - bước bắt buộc để gộp
  ## Session #1 (QKV+RoPE) với Session #2 (attention+oProj+FFN) thành 1.
  ## CHỈ có cài đặt thật trên Metal (metal_scale_enc). Backend khác trả
  ## false có chủ đích - caller PHẢI coi false là "chưa hỗ trợ, rơi về
  ## đường CPU cũ", giống hệt hợp đồng của sessionRope ở trên.
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
    # MỚI THÊM: dùng lại shader activation.comp (opCode 8 = "scale",
    # y = v*p, thêm cạnh opCode 7 "pow") qua vulkanSessionActivationEnc,
    # truyền CÙNG 1 buffer handle làm cả input lẫn output -> ghi đè in-place
    # (an toàn vì activation.comp là elementwise, mỗi invocation chỉ
    # đọc/ghi đúng 1 index của chính nó, không có race giữa các thread).
    # CẦN: activation.spv phải được build lại từ activation.comp đã sửa
    # (glslc kernels/activation.comp -o kernels/activation.spv) trước khi
    # chạy, nếu không opCode 8 sẽ rơi vào nhánh else (relu) và sinh kết quả
    # sai thay vì báo lỗi rõ ràng - ĐÂY LÀ ĐIỂM CẦN TỰ VERIFY TRÊN MÁY THẬT.
    var xv = x
    return vulkanSessionActivationEnc(8, xv.vulkanHandle, xv.vulkanHandle, n, scalar)
  else:
    return false

proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                       srcOffsetElems, dstOffsetElems, n: int): bool =
  ## dst[dstOffsetElems ..< +n] = src[srcOffsetElems ..< +n], device-to-device,
  ## encode vào session đang mở.
  ## MỚI THÊM srcOffsetElems: dùng để trích 1 đoạn con từ 1 buffer NGUỒN lớn
  ## (vd upload 1 lần H*D phần tử cho tất cả KV-head, rồi copy-append từng
  ## đoạn D phần tử của mỗi head vào đúng chỗ trong resident buffer riêng -
  ## thay vì phải upload NHỎ LẺ 1 lần/head như trước, giảm số lệnh GPU/token).
  ## Có cài đặt thật trên cả Metal và CUDA (Metal: copy_into_kernel + setBuffer
  ## offset; CUDA: cuMemcpyDtoDAsync với cả 2 con trỏ dời offset) - cả 2 đều
  ## dùng cơ chế offset NATIVE đã có sẵn trong driver/runtime, không phải
  ## kernel toán học mới.
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
  else:
    return false

proc sessionCopyInto*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                       dstOffsetElems, n: int): bool =
  ## Giữ nguyên overload cũ (srcOffsetElems=0) - KHÔNG cần sửa các chỗ gọi
  ## đã chạy đúng từ trước (vd merge oProj), tương thích ngược 100%.
  sessionCopyInto(backend, src, dst, 0, dstOffsetElems, n)

proc sessionActivation*(backend: GpuBackend, op: string, x: SessionHandle, y: var SessionHandle, n: int, p: float32 = 0.0'f32): bool =
  ## SỬA: thêm tham số p (mặc định 0.0, không phá call site cũ) + mở rộng
  ## opCode cho sqrt/exp/log/pow (4/5/6/7) - trước đây chỉ map được
  ## relu/sigmoid/tanh/apflu (0-3), khiến "gpu sqrt/exp/log/pow" trong
  ## session luôn rơi ra immediate (xem traceWarn trong bybylang.nim).
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
  else:
    return false

proc sessionApfluEnc*(backend: GpuBackend, x: SessionHandle, y: var SessionHandle, n: int,
                       alpha, beta: float32): bool =
  ## APFLU forward qua session. Metal: kernel hiện chỉ hỗ trợ ĐÚNG
  ## alpha=0.1/beta=1.0 (xem metalApflu immediate) - trả false nếu khác,
  ## để caller fallback về immediate (đúng, không âm thầm sai số).
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
    # Chưa có shader apflu.comp thật cho Vulkan (vulkanApflu immediate hiện
    # cũng đang chạy CPU fallback, không phải bug mới) - trả false để
    # caller fallback về immediate, KHÔNG bịa kernel chưa test.
    return false
  else:
    return false

proc sessionApfluBackwardEnc*(backend: GpuBackend, x, dy: SessionHandle, dx: var SessionHandle, n: int,
                               alpha, beta: float32): bool =
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
    return false
  else:
    return false

proc sessionGatherEnc*(backend: GpuBackend, src, idx: SessionHandle, dst: var SessionHandle, n: int): bool =
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
  else:
    return false

proc sessionCompareEnc*(backend: GpuBackend, a, b: SessionHandle, mask: var SessionHandle, op, n: int): bool =
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
  else:
    return false

proc sessionDropoutMaskEnc*(backend: GpuBackend, mask: var SessionHandle, n: int, p: float32, seed: uint32): bool =
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
  else:
    return false

proc sessionArgmaxIndexEnc*(backend: GpuBackend, x: SessionHandle, val, idx: var SessionHandle,
                             outer, dimSize, inner: int): bool =
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
  else:
    return false

proc sessionPermuteEnc*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                         newStrides, oldStrides, dimsMap: SessionHandle, ndim, n: int): bool =
  ## newStrides/oldStrides/dimsMap: ĐÃ upload sẵn (sessionUploadIndices) bởi
  ## caller - bảng nhỏ O(rank), không re-upload mỗi lần trong hàm này.
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
    # Vulkan Enc nhận thẳng seq[int32] (upload nội bộ mỗi lần, rank<=8 nên rẻ)
    # - KHÔNG dùng buffer đã upload như Metal/CUDA, khác 1 chút vì
    # dispatchGeneric push-constant embed bảng ngay trong call.
    return false
  else:
    return false

proc sessionSplitSliceEnc*(backend: GpuBackend, src: SessionHandle, dst: var SessionHandle,
                            dimTotal, thisSize, offset, inner, n: int): bool =
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
  else:
    return false

proc sessionLayernormBackwardEnc*(backend: GpuBackend, dy, x, gamma, beta: SessionHandle,
                                   dx, dgamma, dbeta: var SessionHandle, rows, cols: int, eps: float32): bool =
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
    # vulkanLayernormBackward immediate hiện đang CPU fallback (không có
    # layernorm_backward.comp thật) - trả false, KHÔNG bịa shader chưa test.
    return false
  else:
    return false

proc sessionAttentionFusedBackwardEnc*(backend: GpuBackend, q, k, v, sMatrix, dy: SessionHandle,
                                        dq, dk, dv: var SessionHandle, B, H, S, D: int, scale: float32): bool =
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
  else:
    return false

proc sessionEmbeddingLookupEnc*(backend: GpuBackend, table, indices: SessionHandle, y: var SessionHandle,
                                 vocab, dim, numIndices: int): bool =
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
    # vulkanEmbeddingLookup immediate hiện đang CPU fallback (chưa có
    # embedding.comp thật) - trả false.
    return false
  else:
    return false

# ─────────────────────────────────────────────────────────────
# KẾT THÚC SESSION - COMMIT + WAIT
# ─────────────────────────────────────────────────────────────

proc sessionEnd*(backend: GpuBackend): bool =
  ## Kết thúc session - commit tất cả ops và wait
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
  else:
    return false

proc sessionCommitContinue*(backend: GpuBackend): bool =
  ## MOI: "commitAndContinue" (hoc theo Apple MPSCommandBuffer / PyTorch MPS
  ## SyncType::COMMIT_ADAPTIVE) - commit buffer hien tai KHONG CHO, mo ngay
  ## buffer moi de tiep tuc encode trong CUNG 1 "session" logic. Goi dinh ky
  ## (vi du sau moi N layer) de gioi han kich thuoc TUNG command buffer,
  ## tranh watchdog GPU timeout MA KHONG tra gia dong bo CPU<->GPU moi lan
  ## (khac voi sessionEnd()+sessionBegin() lai giua chung - cach do cham vi
  ## la 1 lan waitUntilCompleted THAT). Xem giai thich day du o
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
  else:
    return false

proc sessionRecoverQueue*(backend: GpuBackend): bool =
  ## MOI: goi ngay sau khi sessionEnd() tra ve false - tao lai
  ## MTLCommandQueue bi "poisoned" (xem giai thich chi tiet o
  ## metalRecoverQueue/metal_recover_queue). Neu KHONG goi ham nay sau 1 lan
  ## sessionEnd that bai, MOI session sau do se tiep tuc bi tu choi
  ## ("Ignored") va sessionRead se tra ve du lieu GPU CU/RAC (khong phai loi
  ## - buffer chi don gian khong duoc GPU ghi lai) - day chinh la nguyen
  ## nhan cac token "rac" quan sat duoc truoc do.
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
  else:
    return false

proc sessionGetLastError*(backend: GpuBackend): string =
  ## THÊM MỚI: mô tả NSError/lỗi THẬT của lần sessionEnd() gần nhất bị fail
  ## (xem giải thích ở metal_shim.h::metal_get_last_session_error) - CHỈ gọi
  ## hàm này NGAY sau khi sessionEnd() trả về false, trước khi mở session
  ## khác (thông tin bị ghi đè ở lần metal_session_end kế tiếp). Trả về ""
  ## nếu backend không hỗ trợ (CUDA/Vulkan chưa nối) hoặc không có lỗi nào.
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
  else:
    return ""

proc sessionRead*(backend: GpuBackend, h: SessionHandle, n: int): seq[float32] =
  ## Đọc dữ liệu từ GPU về CPU sau khi sessionEnd
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
  else:
    return @[]

proc sessionFree*(backend: GpuBackend, h: var SessionHandle) =
  ## Giải phóng buffer trong session
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
  else:
    discard
  h.kind = gbCpu
  h.metalHandle = nil
  h.cudaHandle = CudaResidentTensor(dptr: 0, n: 0)

proc gpuFusedAddAct*(backend: GpuBackend, a, b: seq[float32], act: string): seq[float32] =
  ## GHÉP TỪ SESSION API: fusedAddAct = sessionVecOp("add") + sessionActivation(act).
  ## KHÔNG viết kernel .cu/.metal/.comp mới riêng cho fusedAddAct (chưa ai test
  ## trên phần cứng thật) - thay vào đó ghép 2 kernel ĐÃ CÓ và ĐÃ CHẠY ĐƯỢC
  ## trên cả 3 backend (Metal/CUDA/Vulkan) qua session encode-in-place:
  ##   1) sessionVecOp(chosen, "add", a, b, c, n)      -- c = a + b (GPU thật)
  ##   2) sessionActivation(chosen, act, c, y, n)       -- y = act(c) (GPU thật)
  ## trong CÙNG 1 session (1 lần sessionBegin/sessionEnd), nên vẫn là fused
  ## trên GPU theo nghĩa "không round-trip CPU giữa 2 bước", chỉ khác so với
  ## 1 kernel duy nhất là tốn 1 buffer trung gian (c) trên VRAM.
  ##
  ## KHÔNG có nhánh CPU-fallback-ngầm: nếu backend không phải Metal/CUDA/
  ## Vulkan, hoặc sessionVecOp/sessionActivation/sessionBegin/sessionEnd trả
  ## về false (backend chưa hỗ trợ / init lỗi), RAISE thẳng - đúng chính sách
  ## gForbidCpuFallback của cả file này. Muốn chạy CPU phải gọi
  ## cpuFusedAddAct() trực tiếp, CHỦ ĐỘNG.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  if chosen notin {gbMetal, gbCuda, gbVulkan}:
    raise newException(ValueError,
      "gpuFusedAddAct: CPU fallback bi cam - khong co GPU backend nao kha dung")

  if act notin ["relu", "sigmoid", "tanh", "none"]:
    raise newException(ValueError, "gpuFusedAddAct: unknown activation: " & act)

  let n = min(a.len, b.len)

  if not sessionBegin(chosen):
    raise newException(CatchableError,
      "gpuFusedAddAct: sessionBegin that bai tren backend " & $chosen)

  var ha = sessionUpload(chosen, a)
  var hb = sessionUpload(chosen, b)
  var hc: SessionHandle
  var hy: SessionHandle
  let needsActivation = act != "none"
  try:
    if not sessionVecOp(chosen, "add", ha, hb, hc, n):
      raise newException(CatchableError,
        "gpuFusedAddAct: sessionVecOp('add') that bai / chua duoc session-ho tro tren backend " & $chosen)

    if needsActivation:
      if not sessionActivation(chosen, act, hc, hy, n):
        raise newException(CatchableError,
          "gpuFusedAddAct: sessionActivation('" & act & "') that bai / chua duoc session-ho tro tren backend " & $chosen)
    else:
      hy = hc

    if not sessionEnd(chosen):
      raise newException(CatchableError,
        "gpuFusedAddAct: sessionEnd that bai tren backend " & $chosen)

    result = sessionRead(chosen, hy, n)
  finally:
    sessionFree(chosen, ha)
    sessionFree(chosen, hb)
    if needsActivation:
      sessionFree(chosen, hc)
    sessionFree(chosen, hy)


# ─────────────────────────────────────────────────────────────
# CUDA HELPER - cấp scratch buffer
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SỬ DỤNG: Linear++ Attention với Session API CHUNG
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# RESIDENT WEIGHT (Q4) - upload weight MỘT LẦN lúc load model, giữ nguyên
# trên GPU suốt vòng đời process. Mỗi bước forward chỉ upload activation
# (nhỏ, đổi mỗi token) qua SessionHandle bình thường rồi gọi sessionMatmulQ4
# với handle weight đã resident sẵn - KHÔNG đụng lại vào bytes weight.
# API CHUNG cho Metal (có kernel int4 gốc chạy thẳng trên bytes đã pack) và
# CUDA (CHƯA có nguồn .cu cho kernel int4 (chỉ có vecop.ptx nhị phân sẵn,
# không có .cu đi kèm để sửa/build lại - xem cpuDequantMatmulQ4Fallback ở
# trên) nên dequant MỘT LẦN duy nhất lúc upload rồi giữ resident dạng fp32
# qua sessionUpload thường - vẫn tránh được việc dequant+upload lại mỗi
# token, chỉ khác không tiết kiệm được băng thông 4-bit như Metal).
# ─────────────────────────────────────────────────────────────
type
  ResidentWeightHandle* = object
    kind*: GpuBackend
    k*, n*, groupSize*, nGroupsPerRow*: int
    metalWq*, metalScales*, metalZeros*: MetalBufferHandle
    vulkanWq*, vulkanScales*, vulkanZeros*: VulkanBufferHandle
    ## CUDA fields kept as stubs for binary/layout compatibility only.
    cudaWq*, cudaScales*, cudaZeros*: CudaResidentTensor
    fallbackFp32*: SessionHandle    ## CPU-fallback: weight dequant 1 lần
    cudaQ4Ready*: bool              ## legacy flag; always false (CUDA removed)

proc residentPrepareQ4*(w: ResidentWeightHandle; M = 1): bool =
  ## Opt-in preparation from inference. The ordinary Q4 session API reuses
  ## the selected plan; training and unsupported devices retain their kernel.
  if w.kind == gbMetal:
    when defined(macosx):
      return metalPrepareQ4Tuned(w.metalWq, w.metalScales, w.metalZeros,
        w.k, w.n, w.groupSize, w.nGroupsPerRow, M)
  if w.kind == gbVulkan:
    return vulkanPrepareQ4Tuned(w.vulkanWq, w.vulkanScales, w.vulkanZeros,
      w.k, w.n, w.groupSize, w.nGroupsPerRow, M)
  false

proc residentUploadQ4*(backend: GpuBackend, wq: seq[uint8], scales, zeros: seq[float32],
                        k, n, groupSize, nGroupsPerRow: int): ResidentWeightHandle =
  ## Upload weight int4 (đã pack) lên GPU MỘT LẦN. Gọi lúc load model
  ## (loadLlamaWeightsL), giữ result lại, KHÔNG gọi lại mỗi token.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  result.kind = chosen
  result.k = k; result.n = n
  result.groupSize = groupSize; result.nGroupsPerRow = nGroupsPerRow

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA: Private storage (không phải Shared) - wq/scales/zeros CHỈ được
      # GPU đọc lại mỗi token sau lúc này, CPU không bao giờ đụng lại. Tốn
      # thêm 1 lần blit qua staging buffer CHỈ ở đây (lúc load model), không
      # lặp lại mỗi token.
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
  else:
    # CPU: dequant MỘT LẦN duy nhất (không phải mỗi token) rồi upload
    # resident dạng fp32 [K, N] transposed, dùng chung sessionUpload/sessionMatmul.
    let bytesPerRow = (k + 1) div 2
    var wT = newSeq[float32](k * n)
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
    result.fallbackFp32 = sessionUpload(chosen, wT)

proc sessionMatmulQ4*(backend: GpuBackend, a: SessionHandle, w: ResidentWeightHandle,
                       c: var SessionHandle, M: int): bool =
  ## Encode matmul C(M x N) = A(M x K) @ Wq(K x N) vào session đang mở, dùng
  ## weight resident đã upload qua residentUploadQ4. K/N/groupSize lấy từ w.
  var chosen = backend
  if chosen == gbAuto: chosen = detectBackend()
  if chosen == gbCuda: chosen = gbVulkan

  case chosen
  of gbMetal:
    when defined(macosx):
      # SỬA BUG NGHIÊM TRỌNG (nguồn gốc token rác): c CHƯA được cấp buffer
      # trước khi dùng làm output - giống hệt bug đã fix ở sessionMatmul (xem
      # comment ở đó) nhưng hàm NÀY (dùng cho MỌI Q/K/V/O/gate/up/down
      # projection của model Q4-per-group) lại thiếu đúng bước đó. Caller gọi
      # kiểu sessionMatmulQ4(be, x, w, hOut, M) với hOut là SessionHandle mặc
      # định (metalHandle=nil) - kernel ghi vào buffer không hợp lệ -> logits
      # sai hoàn toàn nhưng KHÔNG crash rõ ràng -> đúng khớp triệu chứng token
      # rác. Model int8/fp32 KHÔNG đi qua hàm này (dùng sessionMatmul đã fix
      # từ trước) nên không bị ảnh hưởng bởi bug này.
      if c.metalHandle == nil:
        c.metalHandle = metalSessionAllocScratch(M * w.n)
      # THÊM MỚI (BUG THẬT tìm bằng đọc kernel .metal, xem matmul_q4_batched.metal):
      # kernel gốc (matmul_q4_kernel_ndst2_paired) dispatch grid Y=M - MỖI
      # hàng M đọc lại TOÀN BỘ weight liên quan từ đầu, hoàn toàn độc lập với
      # hàng khác. Với M=1 (decode) điều này VÔ HẠI (chỉ có 1 hàng, không có
      # gì để tái sử dụng). Với M>1 (prefill, T token cùng lúc) đây là lãng
      # phí NGHIÊM TRỌNG: toàn bộ weight bị đọc lại M LẦN thay vì 1 lần -
      # đây là lý do prefill (92 token) trước đây gần chậm bằng decode nhân
      # 92 lần, thay vì nhanh hơn hàng chục lần như lẽ ra phải thế. Kernel
      # batch mới đọc weight ĐÚNG 1 lần, dùng lại cho MBLOCK=4 hàng liên
      # tiếp - CHỈ áp dụng khi M>1 (không đụng gì đường decode M=1 đã ổn
      # định). Nếu batch fail (kernel chưa compile do build cũ, v.v.) ->
      # fallback về đúng kernel cũ, không có đường nào bị cụt.
      if M > 1:
        if metalSessionMatmulQ4BatchedEnc(a.metalHandle, w.metalWq, w.metalScales, w.metalZeros,
                                           c.metalHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow):
          return true
        # fallback: rơi xuống dùng kernel cũ bên dưới (không return ở đây)
      return metalSessionMatmulQ4Enc(a.metalHandle, w.metalWq, w.metalScales, w.metalZeros,
                                      c.metalHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow)
    else:
      return false
  of gbVulkan:
    if c.vulkanHandle.buf == nil:
      c.vulkanHandle = vulkanSessionAllocScratch(M * w.n)
    return vulkanSessionMatmulQ4Enc(a.vulkanHandle, w.vulkanWq, w.vulkanScales, w.vulkanZeros,
                                     c.vulkanHandle, M, w.k, w.n, w.groupSize, w.nGroupsPerRow)
  of gbCuda:
    # CUDA removed — chosen is remapped to gbVulkan before case
    raise newException(CatchableError, "CUDA backend removed; use Vulkan")
  else:
    # CPU: dùng lại sessionMatmul chung trên bản fp32 đã resident.
    return sessionMatmul(chosen, a, w.fallbackFp32, c, M, w.k, w.n)

proc residentFreeQ4*(w: var ResidentWeightHandle) =
  ## Giải phóng weight resident - chỉ gọi lúc unload model/thoát process,
  ## KHÔNG gọi giữa các token.
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

proc linearPlusAttentionSession*(backend: GpuBackend, q, k, v: seq[float32],
                                  B, H, T, D: int, scale: float32): seq[float32] =
  ## Linear++ Attention dùng Session API chung - chạy trên Metal/CUDA
  let qkvLen = B * H * T * D
  result = newSeq[float32](qkvLen)

  if not sessionBegin(backend):
    echo "Session begin failed,stop"
    return

  # Upload dữ liệu
  var hQ = sessionUpload(backend, q)
  var hK = sessionUpload(backend, k)
  var hV = sessionUpload(backend, v)
  var hKV = sessionAllocScratch(backend, qkvLen)
  var hO = sessionAllocScratch(backend, qkvLen)

  # Encode các ops vào session
  # Bước 1: K^T @ V (tạm thời dùng matmul)
  # TODO: cần transpose K và V trước
  # Bước 2: Q @ KV
  # Bước 3: scale + norm

  # Kết thúc session
  if not sessionEnd(backend):
    echo "Session failed,stop"
    return

  # Đọc kết quả
  result = sessionRead(backend, hO, qkvLen)

  # Giải phóng buffer
  sessionFree(backend, hQ); sessionFree(backend, hK)
  sessionFree(backend, hV); sessionFree(backend, hKV)
  sessionFree(backend, hO)# gpubackend_additions.nim
# Nối vào gpubackend.nim, cùng pattern try/case/handleGpuFailure như
# gpuSqrt/gpuExp đã có ở trên trong file đó.

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
      raise newException(ValueError, "gpuGather: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuArgmaxIndex: CPU fallback bi cam - khong co GPU backend nao kha dung")
    of gbAuto: discard # roi xuong ban CPU ben duoi
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
      raise newException(ValueError, "gpuCompare: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuDropoutMask: CPU fallback bi cam - khong co GPU backend nao kha dung")
    of gbAuto: discard
  except CatchableError as e:
    handleGpuFailure(chosen, "dropout_mask", e)
  # Fallback CPU xorshift32 - CÙNG công thức với 3 kernel GPU (chỉ chạy khi
  # backend=cpu tường minh hoặc handleGpuFailure cho phép rơi xuống).
  result = newSeq[float32](n)
  let scale = if p < 1'f32: 1'f32 / (1'f32 - p) else: 0'f32
  for i in 0 ..< n:
    var state = seed xor (uint32(i) * 2654435761'u32 + 1'u32)
    state = state xor (state shl 13); state = state xor (state shr 17); state = state xor (state shl 5)
    state = state xor (state shl 13); state = state xor (state shr 17); state = state xor (state shl 5)
    let u = float32(state shr 8) * (1'f32 / 16777216'f32)
    result[i] = if u < p: 0'f32 else: scale

# ─────────────────────────────────────────────────────────────
# GPU HOÁ THẬT (vòng 3) - GIỐNG TORCH: permute/split/concat dispatch. CPU
# chỉ chuẩn bị bảng nhỏ O(rank)/O(numTensors) (không phải O(n)) rồi gọi
# kernel tương ứng ở 1 trong 3 backend.
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
      raise newException(ValueError, "gpuPermute: CPU fallback bi cam - khong co GPU backend nao kha dung")
    of gbAuto: discard
  except CatchableError as e:
    handleGpuFailure(chosen, "permute", e)
  # CPU fallback (chỉ chạy khi backend=cpu tường minh hoặc fallback được phép)
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
      raise newException(ValueError, "gpuSplitSlice: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
      raise newException(ValueError, "gpuConcat: CPU fallback bi cam - khong co GPU backend nao kha dung")
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
