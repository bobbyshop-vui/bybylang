# gpubackend.nim - Lớp điều phối GPU cho BybyLang.
# Cú pháp trong .bybylang:
#   gpu backend is "auto"     # hoặc "cuda" / "metal" / "opencl" / "cpu"
#   gpu array A = [1, 2, 3, 4]
#   gpu add A, B -> C size 4
#   gpu sub A, B -> C
#   gpu mul A, B -> C
#   gpu div A, B -> C
#
# "auto" sẽ tự dò: CUDA (NVIDIA) -> Metal (macOS) -> OpenCL (mọi GPU/CPU khác) -> CPU thuần.
# Nếu backend được chỉ định cụ thể mà không khả dụng, hoặc chạy lỗi, sẽ tự rơi (fallback)
# về CPU để chương trình luôn cho ra kết quả đúng thay vì crash.
import std/strutils
import backends/cuda/cuda_driver
import backends/cuda/cuda_runtime
import backends/opencl/opencl_api
import backends/metal/metal_backend
import tsic_ir

type
  GpuBackend* = enum
    gbAuto, gbCpu, gbCuda, gbMetal, gbOpenCL, gbTsic

proc parseBackend*(s: string): GpuBackend =
  case s.strip().toLowerAscii()
  of "cuda": gbCuda
  of "metal": gbMetal
  of "opencl": gbOpenCL
  of "cpu": gbCpu
  of "tsic": gbTsic
  else: gbAuto

proc `$`*(b: GpuBackend): string =
  case b
  of gbAuto: "auto"
  of gbCpu: "cpu"
  of gbCuda: "cuda"
  of gbMetal: "metal"
  of gbOpenCL: "opencl"
  of gbTsic: "tsic"

proc detectBackend*(): GpuBackend =
  ## Tự động dò backend GPU tốt nhất hiện có trên máy đang chạy.
  ## "tsic" là backend trung gian cho GPU tự chế -> không nằm trong auto-detect,
  ## phải chọn tường minh bằng `gpu backend is "tsic"`.
  if cudaAvailable():
    return gbCuda
  when defined(macosx):
    if metalAvailable():
      return gbMetal
  if openclAvailable():
    return gbOpenCL
  return gbCpu

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

proc gpuOp*(op: string, backend: GpuBackend, a, b: seq[float32]): seq[float32] =
  ## Điểm vào chính: chạy phép toán elementwise `op` (add/sub/mul/div) trên `backend`.
  ## Nếu backend là auto thì tự detect. Nếu backend chọn thất bại lúc chạy, fallback CPU.
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbCuda:
      return cudaVecOp(op, a, b)
    of gbMetal:
      return metalVecOp(op, a, b)
    of gbOpenCL:
      return openclVecOp(op, a, b)
    of gbTsic:
      return tsicVecOp(op, a, b)
    of gbCpu, gbAuto:
      return cpuVecOp(op, a, b)
  except CatchableError as e:
    stderr.writeLine("[GPU] backend '" & $chosen & "' failed (" & e.msg & "), fallback CPU")
    return cpuVecOp(op, a, b)

proc cpuMatmul(a, b: seq[float32], m, k, n: int): seq[float32] =
  result = newSeq[float32](m * n)
  for i in 0..<m:
    for j in 0..<n:
      var s: float32 = 0
      for p in 0..<k:
        s += a[i*k + p] * b[p*n + j]
      result[i*n + j] = s

proc gpuMatmul*(backend: GpuBackend, a, b: seq[float32], m, k, n: int): seq[float32] =
  ## C(m x n) = A(m x k) * B(k x n), row-major. Trên "cuda" dùng cudart + cuBLAS
  ## (cuda_runtime.nim) với Tensor Core math mode bật sẵn -> tận dụng Tensor Core
  ## khi GPU hỗ trợ, tự fallback FP32 CUDA core bình thường nếu không.
  var chosen = backend
  if chosen == gbAuto:
    chosen = detectBackend()

  try:
    case chosen
    of gbCuda:
      return cudaMatmulF32(a, b, m, k, n)
    of gbMetal:
      return metalMatmul(a, b, m, k, n)
    of gbOpenCL:
      return openclMatmul(a, b, m, k, n)
    of gbTsic:
      return tsicMatmulOp(a, b, m, k, n)
    of gbCpu, gbAuto:
      return cpuMatmul(a, b, m, k, n)
  except CatchableError as e:
    stderr.writeLine("[GPU] matmul backend '" & $chosen & "' failed (" & e.msg & "), fallback CPU")
    return cpuMatmul(a, b, m, k, n)

# Biến toàn cục backend hiện tại, sinh code sẽ set/đọc biến này (mặc định auto).
var gpuBackendSelected*: GpuBackend = gbAuto
