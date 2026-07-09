# metal_backend.nim - Metal GPU backend wrapper (chỉ biên dịch trên macOS).
# Trên Linux/Windows module này export các proc "not available" để gpubackend.nim
# vẫn import được bình thường trên mọi OS mà không cần #if trong code gọi.

import std/os

when defined(macosx):
  # Header/.m nằm CÙNG thư mục với chính file này (backends/metal/), nhưng
  # clang chỉ được Nim truyền -I cho thư mục chứa main module (vd. demo/),
  # nên phải tự thêm -I trỏ đúng thư mục này thì mới #include "metal_shim.h"
  # được, bất kể main module nằm ở đâu.
  const metalDir = currentSourcePath().parentDir()
  {.passC: "-I" & metalDir.}
  {.passC: "-fobjc-arc".}
  {.passL: "-framework Metal -framework Foundation".}
  {.compile: "metal_shim.m".}

  # Kernel source là file .metal THẬT (backends/metal/kernels/vecop_matmul.metal),
  # không còn escape thành C-string trong metal_shim.m nữa. staticRead đọc file
  # lúc Nim compile-time (không cần công cụ build ngoài, không cần metallib build-time).
  const kMetalKernelSrc = staticRead(metalDir / "kernels" / "vecop_matmul.metal")

  proc metal_available_c(): cint {.importc: "metal_available", header: "metal_shim.h".}
  proc metal_vecop_c(kernelSrc: cstring, op: cint, a, b: ptr float32, c: ptr float32, n: cint): cint
    {.importc: "metal_vecop", header: "metal_shim.h".}
  proc metal_matmul_c(kernelSrc: cstring, a, b: ptr float32, c: ptr float32, m, k, n: cint): cint
    {.importc: "metal_matmul", header: "metal_shim.h".}

  proc metalAvailable*(): bool =
    metal_available_c() != 0

  proc metalVecOp*(op: string, a, b: seq[float32]): seq[float32] =
    let opCode: cint =
      case op
      of "add": 0
      of "sub": 1
      of "mul": 2
      of "div": 3
      else: 0
    let n = a.len
    result = newSeq[float32](n)
    var aVar = a
    var bVar = b
    let ok = metal_vecop_c(kMetalKernelSrc.cstring, opCode, addr aVar[0], addr bVar[0], addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_vecop failed")

  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    result = newSeq[float32](m * n)
    var aVar = a
    var bVar = b
    let ok = metal_matmul_c(kMetalKernelSrc.cstring, addr aVar[0], addr bVar[0], addr result[0], cint(m), cint(k), cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_matmul failed")

else:
  proc metalAvailable*(): bool = false
  proc metalVecOp*(op: string, a, b: seq[float32]): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    raise newException(CatchableError, "Metal backend chỉ khả dụng trên macOS")
