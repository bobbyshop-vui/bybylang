# metal_backend.nim - Metal GPU backend wrapper (compiled on macOS only).
import std/os
proc c_free(p: pointer) {.importc: "free", header: "<stdlib.h>".}

when defined(macosx):
  const metalDir = currentSourcePath().parentDir()
  {.passC: "-I" & metalDir.}
  {.passC: "-fobjc-arc".}
  {.passL: "-framework Metal -framework Foundation -framework MetalPerformanceShaders".}
  {.compile: "metal_shim.m".}

  const kMetalKernelSrc = staticRead(metalDir / "kernels" / "vecop_matmul.metal") &
    "\n" & staticRead(metalDir / "kernels" / "attention_decode_all_heads.metal") &
    "\n" & staticRead(metalDir / "kernels" / "attention_prefill_flash.metal") &
    "\n" & staticRead(metalDir / "kernels" / "attention_prefill_flash_fp16.metal") &
    "\n" & staticRead(metalDir / "kernels" / "matmul_q4_batched.metal") &
    "\n" & staticRead(metalDir / "kernels" / "matmul_q4_gemm.metal") &
    "\n" & staticRead(metalDir / "kernels" / "micro_vm.metal") &
    "\n" & staticRead(metalDir / "kernels" / "q4_dynamic.metal")
    ## real batch-M kernel for prefill (matmul_q4_batched_m4_kernel) - see
    ## the full explanation in that file (original bug: the M=1 decode kernel re-read
    ## all weights for EVERY row when called with M>1).

  proc metal_available_c(): cint {.importc: "metal_available", header: "metal_shim.h".}
  proc metal_is_apple_gpu_c(): cint {.importc: "metal_is_apple_gpu", header: "metal_shim.h".}
  proc metalIsAppleGpu*(): bool = metal_is_apple_gpu_c() != 0
  proc metal_vecop_c(kernelSrc: cstring, op: cint, a, b: ptr float32, c: ptr float32, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_vecop", header: "metal_shim.h".}
  proc metal_matmul_c(kernelSrc: cstring, a, b: ptr float32, c: ptr float32, m, k, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_matmul", header: "metal_shim.h".}
  proc metal_matmul_mps_c(a, b: ptr float32, c: ptr float32, m, k, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_matmul_mps", header: "metal_shim.h".}
  proc metal_matmul2_c(kernelSrc: cstring,
                          a1, b1: ptr float32, c1: ptr float32, m1, k1, n1: cint,
                          a2, b2: ptr float32, c2: ptr float32, m2, k2, n2: cint, error_out: ptr cstring): cint
    {.importc: "metal_matmul2", header: "metal_shim.h".}
  proc metal_matmul_q4_c(kernelSrc: cstring, a: ptr float32, wq: ptr uint8,
                            scales, zeros: ptr float32, c: ptr float32,
                            m, k, n, groupSize, nGroupsPerRow: cint, error_out: ptr cstring): cint
    {.importc: "metal_matmul_q4", header: "metal_shim.h".}
  proc metal_activation_c(kernelSrc: cstring, op: cint, x: ptr float32, y: ptr float32, n: cint, p: cfloat, error_out: ptr cstring): cint
    {.importc: "metal_activation", header: "metal_shim.h".}
  proc metal_recover_queue_c(): cint {.importc: "metal_recover_queue", header: "metal_shim.h".}

  proc metal_gather_c(kernelSrc: cstring, src: ptr float32, srcLen: cint,
                         idx: ptr cint, dst: ptr float32, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_gather", header: "metal_shim.h".}
  proc metal_argmax_c(kernelSrc: cstring, x: ptr float32, outVal: ptr float32,
                         outIdx: ptr cint, outer, dimSize, inner: cint, error_out: ptr cstring): cint
    {.importc: "metal_argmax", header: "metal_shim.h".}
  proc metal_compare_c(kernelSrc: cstring, a, b: ptr float32, mask: ptr uint8,
                          op: cint, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_compare", header: "metal_shim.h".}
  proc metal_dropout_mask_c(kernelSrc: cstring, mask: ptr float32,
                               seed: cuint, p, scale: cfloat, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_dropout_mask", header: "metal_shim.h".}

  proc metal_permute_c(kernelSrc: cstring, src: ptr float32, dst: ptr float32,
                          newStrides, oldStrides, dimsMap: ptr cint, ndim, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_permute", header: "metal_shim.h".}
  proc metal_split_slice_c(kernelSrc: cstring, src: ptr float32, srcLen: cint, dst: ptr float32,
                              dimTotal, thisSize, offset, inner, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_split_slice", header: "metal_shim.h".}
  proc metal_concat_gather_c(kernelSrc: cstring, flatSrc: ptr float32, flatLen: cint,
                                srcBase, srcDimSize, srcDimOffset: ptr cint, numTensors: cint,
                                dst: ptr float32, totalDim, inner, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_concat_gather", header: "metal_shim.h".}
  proc metal_softmax_c(kernelSrc: cstring, x: ptr float32, y: ptr float32, rows, cols: cint, error_out: ptr cstring): cint
    {.importc: "metal_softmax", header: "metal_shim.h".}
  proc metal_layernorm_c(kernelSrc: cstring, x, gamma, beta: ptr float32, y: ptr float32, rows, cols: cint, eps: float32, error_out: ptr cstring): cint
    {.importc: "metal_layernorm", header: "metal_shim.h".}
  proc metal_embedding_lookup_c(kernelSrc: cstring, table: ptr float32, indices: ptr int32, y: ptr float32, vocab, dim, num_indices: cint, error_out: ptr cstring): cint
    {.importc: "metal_embedding_lookup", header: "metal_shim.h".}
  proc metal_activation_backward_c(kernelSrc: cstring, op: cint, x, dy: ptr float32, dx: ptr float32, n: cint, error_out: ptr cstring): cint
    {.importc: "metal_activation_backward", header: "metal_shim.h".}
  proc metal_layernorm_backward_c(kernelSrc: cstring, dy, x, gamma, beta: ptr float32,
                                     dx, dgamma, dbeta: ptr float32, rows, cols: cint, eps: float32, error_out: ptr cstring): cint
    {.importc: "metal_layernorm_backward", header: "metal_shim.h".}
  
    # === THÊM MỚI: Attention forward + backward ===
  proc metal_attention_fused_c(kernelSrc: cstring, q, k, v: ptr float32, o, s_matrix: ptr float32,
                                  B, H, S, D: cint, scale: float32, error_out: ptr cstring): cint
    {.importc: "metal_attention_fused", header: "metal_shim.h".}
  proc metal_attention_fused_backward_c(kernelSrc: cstring, q, k, v, s_matrix, dy: ptr float32,
                                           dq, dk, dv: ptr float32, B, H, S, D: cint, scale: float32, error_out: ptr cstring): cint
    {.importc: "metal_attention_fused_backward", header: "metal_shim.h".}

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
    var err: cstring = nil
    let ok = metal_vecop_c(kMetalKernelSrc.cstring, opCode, addr aVar[0], addr bVar[0], addr result[0], cint(n), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_vecop failed (no error detail)"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)

  # ─────────────────────────────────────────────────────────────────────
  # MPS ON/OFF SWITCH - edit this value to choose the matmul path.
    #   bmUseMps    -> always MPSMatrixMultiplication (falls back automatically
    #                  to the hand-written kernel if unsupported / on error).
    #   bmUseKernel -> always the hand-written matmul_kernel_t{8,16,32}
    #                  (skips MPS entirely - for benchmark comparisons
    #                  or when MPS is suspected of wrong results on a given GPU).
    # ─────────────────────────────────────────────────────────────────────
  type BbMatmulBackend = enum
    bmUseMps, bmUseKernel

  const bbMatmulBackend: BbMatmulBackend = bmUseKernel   # <<< EDIT HERE
  # CHANGED: default bmUseMps -> bmUseKernel. Measured on Iris Plus
  # 640: MPS is SLOWER than the hand-written kernel for LLM decode (16942ms/token
  # vs ~7000-7800ms/token before). Reason: each decode step multiplies only
  # M=1 (one token) - really a matrix-VECTOR product, not a large GEMM, and
  # MPS is optimised for large matrices/batches. On a weak GPU MPS's internal
  # setup (algorithm choice, descriptor validation...) for M=1 costs more
  # than the actual math -> counterproductive. The MPS code stays (set
  # bmUseMps to compare, or for large-M prefill later).

  var gMpsFailedOnce = false  # CHANGED: if MPS fails once (e.g. unsupported GPU),
                              # do not retry every token - each costs a failed
                              # native call. Fall back to the hand-written
                              # kernel for the rest of the process.

  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    # FIX: Check for empty inputs before GPU allocation
    if a.len == 0 or b.len == 0 or m == 0 or k == 0 or n == 0:
      stderr.writeLine("[metalMatmul] WARNING: empty input detected, fallback to CPU, dims=(" & $m & "," & $k & "," & $n & "), a.len=" & $a.len & ", b.len=" & $b.len)
      result = newSeq[float32](m * n)
      return
    result = newSeq[float32](m * n)
    var aVar = a
    var bVar = b
    # CHANGED (#3 - use MPS): prefer MPSMatrixMultiplication - Apple picks
    # the best algorithm for the GPU in use, usually noticeably faster
    # than the hand-written matmul_kernel_t{8,16,32}, with no threadgroup
    # limit / tile size concerns. Falls back to the hand-written kernel (with dynamic tiles
    # from #A) if MPS is unavailable or fails. Can be disabled entirely
    # by setting bbMatmulBackend above to bmUseKernel.
    if m > 1 and bbMatmulBackend == bmUseMps and not gMpsFailedOnce:
      var err: cstring = nil
      let okMps = metal_matmul_mps_c(addr aVar[0], addr bVar[0], addr result[0], cint(m), cint(k), cint(n), addr err)
      if okMps != 0:
        if err != nil: c_free(err)
        return
      if err != nil: c_free(err)
      gMpsFailedOnce = true
      stderr.write("[metal] MPS matmul unavailable/failed - using the hand-written kernel for the rest of the session\n")
    for attempt in 0 ..< 2:
      var err: cstring = nil
      let ok = metal_matmul_c(kMetalKernelSrc.cstring, addr aVar[0], addr bVar[0], addr result[0], cint(m), cint(k), cint(n), addr err)
      if ok != 0:
        if err != nil:
          let msg = $err
          c_free(err)
          raise newException(CatchableError, msg)
        return
      if err != nil: c_free(err)
      stderr.writeLine("[metal] matmul that bai (co the timeout), retry " & $(attempt + 1) & "/2")
      discard metal_recover_queue_c()
    raise newException(CatchableError, "metal_matmul failed after automatic retries (metal_matmul_c returned 0)")

  proc metalMatmul2*(a1, b1: seq[float32], m1, k1, n1: int,
                      a2, b2: seq[float32], m2, k2, n2: int):
                      tuple[c1, c2: seq[float32]] =
    ## Intel iGPU watchdog: a backward Linear merging dW+dX can make one
    ## command buffer run too long.  A timed-out command buffer also
    ## poisons the old queue, so a retry must create a new queue first.
    ## After two identical retries, split by rows to shorten each
    ## dispatch.  This is still the GPU path, training never falls back to the CPU.
    proc tiled(a, b: seq[float32], m, k, n, rowTile: int): seq[float32] =
      result = newSeq[float32](m * n)
      var row = 0
      while row < m:
        let rows = min(rowTile, m - row)
        var aPart = newSeq[float32](rows * k)
        for i in 0 ..< rows:
          for j in 0 ..< k:
            aPart[i * k + j] = a[(row + i) * k + j]
        let part = metalMatmul(aPart, b, rows, k, n)
        for i in 0 ..< rows:
          for j in 0 ..< n:
            result[(row + i) * n + j] = part[i * n + j]
        row += rows

    for attempt in 0 ..< 2:
      result.c1 = newSeq[float32](m1 * n1)
      result.c2 = newSeq[float32](m2 * n2)
      var a1v = a1; var b1v = b1
      var a2v = a2; var b2v = b2
      var err: cstring = nil
      let ok = metal_matmul2_c(kMetalKernelSrc.cstring,
        addr a1v[0], addr b1v[0], addr result.c1[0], cint(m1), cint(k1), cint(n1),
        addr a2v[0], addr b2v[0], addr result.c2[0], cint(m2), cint(k2), cint(n2), addr err)
      if ok != 0:
        if err != nil: c_free(err)
        return
      if err != nil: c_free(err)
      stderr.writeLine("[metal] matmul2 that bai (co the timeout), retry " & $(attempt + 1) & "/2")
      discard metal_recover_queue_c()

    stderr.writeLine("[metal] matmul2 retry failed -> splitting into smaller GPU tiles to avoid the watchdog")
    var lastError = "unknown"
    for rowTile in [16, 8, 4, 2, 1]:
      discard metal_recover_queue_c()
      try:
        result.c1 = tiled(a1, b1, m1, k1, n1, rowTile)
        result.c2 = tiled(a2, b2, m2, k2, n2, rowTile)
        return
      except CatchableError as e:
        lastError = e.msg
        stderr.writeLine("[metal] matmul2 tile=" & $rowTile & " failed; recover queue and retry smaller tile: " & e.msg)
    raise newException(CatchableError,
      "metal_matmul2 failed after retry and adaptive GPU tiling down to one row: " & lastError)

  proc metalMatmulQ4*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                      m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
    ## Matmul directly on packed int4-asymmetric weights, no fp32 dequant
    ## on the CPU first (see matmul_q4_kernel in vecop_matmul.metal and
    ## quant.nim/dequantizeTensorTransposed for the exact layout).
    # FIX: Check for empty inputs before GPU allocation
    if a.len == 0 or wq.len == 0 or scales.len == 0 or zeros.len == 0 or m == 0 or k == 0 or n == 0:
      stderr.writeLine("[metalMatmulQ4] WARNING: empty input detected, fallback to CPU, dims=(" & $m & "," & $k & "," & $n & ")")
      result = newSeq[float32](m * n)
      return
    result = newSeq[float32](m * n)
    var aVar = a
    var wqVar = wq
    var scalesVar = scales
    var zerosVar = zeros
    var err: cstring = nil
    let ok = metal_matmul_q4_c(kMetalKernelSrc.cstring,
      addr aVar[0], addr wqVar[0], addr scalesVar[0], addr zerosVar[0], addr result[0],
      cint(m), cint(k), cint(n), cint(groupSize), cint(nGroupsPerRow), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_matmul_q4 failed (metal_matmul_q4_c returned 0)"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)

  proc metalGather*(src: seq[float32], idx: seq[int32]): seq[float32] =
    let n = idx.len
    result = newSeq[float32](n)
    var srcVar = src
    var idxC = newSeq[cint](n)
    for i in 0 ..< n: idxC[i] = cint(idx[i])
    var err: cstring = nil
    let ok = metal_gather_c(kMetalKernelSrc.cstring, addr srcVar[0], cint(src.len),
      addr idxC[0], addr result[0], cint(n), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_gather failed"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)

  proc metalArgmaxIndex*(x: seq[float32], outer, dimSize, inner: int): tuple[val: seq[float32], idx: seq[int32]] =
    let pairs = outer * inner
    var outVal = newSeq[float32](pairs)
    var outIdxC = newSeq[cint](pairs)
    var xVar = x
    var err: cstring = nil
    let ok = metal_argmax_c(kMetalKernelSrc.cstring, addr xVar[0], addr outVal[0],
      addr outIdxC[0], cint(outer), cint(dimSize), cint(inner), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_argmax failed"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)
    var outIdx = newSeq[int32](pairs)
    for i in 0 ..< pairs: outIdx[i] = int32(outIdxC[i])
    result = (val: outVal, idx: outIdx)

  proc metalCompare*(a, b: seq[float32], op: int): seq[uint8] =
    let n = a.len
    result = newSeq[uint8](n)
    var aVar = a
    var bVar = b
    var err: cstring = nil
    let ok = metal_compare_c(kMetalKernelSrc.cstring, addr aVar[0], addr bVar[0],
      addr result[0], cint(op), cint(n), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_compare failed"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)

  proc metalDropoutMask*(n: int, p: float32, seed: uint32): seq[float32] =
    result = newSeq[float32](n)
    let scale = if p < 1'f32: 1'f32 / (1'f32 - p) else: 0'f32
    var err: cstring = nil
    let ok = metal_dropout_mask_c(kMetalKernelSrc.cstring, addr result[0],
      cuint(seed), cfloat(p), cfloat(scale), cint(n), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_dropout_mask failed"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)

  proc metalPermute*(src: seq[float32], newStrides, oldStrides, dimsMap: seq[int32], n: int): seq[float32] =
    ## LIKE TORCH: only a small O(rank) table (strides+dimsMap) is passed - the device
    ## decodes each element's source index, no O(n) index array.
    let ndim = newStrides.len
    result = newSeq[float32](n)
    var srcVar = src
    var newStrC = newSeq[cint](ndim); var oldStrC = newSeq[cint](ndim); var dimsMapC = newSeq[cint](ndim)
    for i in 0 ..< ndim:
      newStrC[i] = cint(newStrides[i]); oldStrC[i] = cint(oldStrides[i]); dimsMapC[i] = cint(dimsMap[i])
    var err: cstring = nil
    let ok = metal_permute_c(kMetalKernelSrc.cstring, addr srcVar[0], addr result[0],
      addr newStrC[0], addr oldStrC[0], addr dimsMapC[0], cint(ndim), cint(n), addr err)
    if ok == 0:
      let msg = if err != nil: $err else: "metal_permute failed"
      if err != nil: c_free(err)
      raise newException(CatchableError, msg)

  proc metalSplitSlice*(src: seq[float32], dimTotal, thisSize, offset, inner, n: int): seq[float32] =
    result = newSeq[float32](n)
    var srcVar = src
    var err: cstring = nil
    let ok = metal_split_slice_c(kMetalKernelSrc.cstring, addr srcVar[0], cint(src.len), addr result[0],
      cint(dimTotal), cint(thisSize), cint(offset), cint(inner), cint(n), addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_split_slice failed")

  proc metalConcat*(flatSrc: seq[float32], srcBase, srcDimSize, srcDimOffset: seq[int32],
                     totalDim, inner, n: int): seq[float32] =
    let numTensors = srcBase.len
    result = newSeq[float32](n)
    var flatVar = flatSrc
    var baseC = newSeq[cint](numTensors); var sizeC = newSeq[cint](numTensors); var offC = newSeq[cint](numTensors)
    for i in 0 ..< numTensors:
      baseC[i] = cint(srcBase[i]); sizeC[i] = cint(srcDimSize[i]); offC[i] = cint(srcDimOffset[i])
    var err: cstring = nil
    let ok = metal_concat_gather_c(kMetalKernelSrc.cstring, addr flatVar[0], cint(flatSrc.len),
      addr baseC[0], addr sizeC[0], addr offC[0], cint(numTensors),
      addr result[0], cint(totalDim), cint(inner), cint(n), addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_concat_gather failed")

  proc metalActivation*(op: string, x: seq[float32], p: float32 = 0.0'f32): seq[float32] =
    let opCode: cint =
      case op
      of "relu": 0
      of "sigmoid": 1
      of "tanh": 2
      of "exp": 4
      of "log": 5
      of "sqrt": 6
      of "pow": 7
      else: 0
    let n = x.len
    result = newSeq[float32](n)
    var xVar = x
    var err: cstring = nil
    let ok = metal_activation_c(kMetalKernelSrc.cstring, opCode, addr xVar[0], addr result[0], cint(n), cfloat(p), addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_activation failed")

  proc metalSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
    let n = rows * cols
    result = newSeq[float32](n)
    var xVar = x
    var err: cstring = nil
    let ok = metal_softmax_c(kMetalKernelSrc.cstring, addr xVar[0], addr result[0], cint(rows), cint(cols), addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_softmax failed")

  proc metalLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
    let n = rows * cols
    result = newSeq[float32](n)
    var xVar = x
    var gammaVar = gamma
    var betaVar = beta
    var err: cstring = nil
    let ok = metal_layernorm_c(kMetalKernelSrc.cstring, addr xVar[0], addr gammaVar[0], addr betaVar[0], addr result[0], cint(rows), cint(cols), eps, addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_layernorm failed")

  proc metalEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
    let num = indices.len
    result = newSeq[float32](num * dim)
    var tableVar = table
    var indicesVar = indices
    var err: cstring = nil
    let ok = metal_embedding_lookup_c(kMetalKernelSrc.cstring, addr tableVar[0], addr indicesVar[0], addr result[0], cint(vocab), cint(dim), cint(num), addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_embedding_lookup failed")

  proc metalApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
    if alpha != 0.1'f32 or beta != 1.0'f32:
      raise newException(CatchableError,
        "metalApflu: the Metal kernel currently only supports alpha=0.1, beta=1.0")
    let n = x.len
    result = newSeq[float32](n)
    var xVar = x
    var err: cstring = nil
    let ok = metal_activation_c(kMetalKernelSrc.cstring, 3.cint, addr xVar[0], addr result[0], cint(n), 0.0'f32, addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_activation(apflu) failed")

  proc metalApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
    if alpha != 0.1'f32 or beta != 1.0'f32:
      raise newException(CatchableError,
        "metalApfluBackward: the Metal kernel currently only supports alpha=0.1, beta=1.0")
    let n = x.len
    result = newSeq[float32](n)
    var xVar = x; var dyVar = dy
    var err: cstring = nil
    let ok = metal_activation_backward_c(kMetalKernelSrc.cstring, 3.cint, addr xVar[0], addr dyVar[0], addr result[0], cint(n), addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_activation_backward(apflu) failed")

  proc metalLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
    var dx = newSeq[float32](rows * cols)
    var dgamma = newSeq[float32](cols)
    var dbeta = newSeq[float32](cols)
    var dyVar = dy; var xVar = x; var gammaVar = gamma; var betaVar = beta
    var err: cstring = nil
    let ok = metal_layernorm_backward_c(kMetalKernelSrc.cstring,
      addr dyVar[0], addr xVar[0], addr gammaVar[0], addr betaVar[0],
      addr dx[0], addr dgamma[0], addr dbeta[0], cint(rows), cint(cols), eps, addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_layernorm_backward failed")
    return (dx, dgamma, dbeta)

  # === THÊM MỚI: Attention forward ===
  proc metalAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
    var o = newSeq[float32](B * H * S * D)
    var sMatrix = newSeq[float32](B * H * S * S)
    var qVar = q; var kVar = k; var vVar = v
    var err: cstring = nil
    let ok = metal_attention_fused_c(kMetalKernelSrc.cstring,
      addr qVar[0], addr kVar[0], addr vVar[0], addr o[0], addr sMatrix[0],
      cint(B), cint(H), cint(S), cint(D), scale, addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_attention_fused failed")
    return (o, sMatrix)

  # === THÊM MỚI: Attention backward ===
  proc metalAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
    var dq = newSeq[float32](B * H * S * D)
    var dk = newSeq[float32](B * H * S * D)
    var dv = newSeq[float32](B * H * S * D)
    var qVar = q; var kVar = k; var vVar = v; var sVar = s_matrix; var dyVar = dy
    var err: cstring = nil
    let ok = metal_attention_fused_backward_c(kMetalKernelSrc.cstring,
      addr qVar[0], addr kVar[0], addr vVar[0], addr sVar[0], addr dyVar[0],
      addr dq[0], addr dk[0], addr dv[0], cint(B), cint(H), cint(S), cint(D), scale, addr err)
    if ok == 0:
      raise newException(CatchableError, "metal_attention_fused_backward failed")
    return (dq, dk, dv)

  # === Resident API: batches many ops into one MTLCommandBuffer, commit+wait
  # only once in sessionEnd() instead of once per matmul/softmax. Used by the
  # decode loop (per token) to cut dispatch overhead on weak iGPUs.
  type MetalBufferHandle* = pointer

  proc metal_session_begin_c(kernelSrc: cstring, error_out: ptr cstring): cint {.importc: "metal_session_begin", header: "metal_shim.h".}
  proc metal_upload_c(data: ptr float32, n: cint): MetalBufferHandle {.importc: "metal_upload", header: "metal_shim.h".}
  proc metal_upload_borrowed_c(data: ptr float32, n: cint): MetalBufferHandle {.importc: "metal_upload_borrowed", header: "metal_shim.h".}
  proc metal_wrap_host_nocopy_c(data: pointer, bytes: csize_t): MetalBufferHandle {.importc: "metal_wrap_host_nocopy", header: "metal_shim.h".}
  proc metal_alloc_scratch_c(n: cint): MetalBufferHandle {.importc: "metal_alloc_scratch", header: "metal_shim.h".}
  # NEW: see the full explanation in metal_shim.h/metal_shim.m -
  # an MTLHeap to suballocate many buffers with the same lifetime (resident KV) is MUCH
  # cheaper than thousands of separate metal_alloc_scratch calls.
  proc metal_alloc_heap_c(bytes: clong): pointer {.importc: "metal_alloc_heap", header: "metal_shim.h".}
  proc metal_alloc_from_heap_c(heap: pointer, n: cint): MetalBufferHandle {.importc: "metal_alloc_from_heap", header: "metal_shim.h".}
  proc metal_free_heap_c(heap: pointer) {.importc: "metal_free_heap", header: "metal_shim.h".}
  proc metal_matmul_enc_c(a, b, c: MetalBufferHandle, m, k, n: cint): cint {.importc: "metal_matmul_enc", header: "metal_shim.h".}
  proc metal_matmul_nt_enc_c(a, b, c: MetalBufferHandle, m, k, n: cint): cint {.importc: "metal_matmul_nt_enc", header: "metal_shim.h".}
  proc metal_matmul_nt_off_enc_c(a, b: MetalBufferHandle, bOffsetBytes: cint, c: MetalBufferHandle, m, k, n: cint): cint {.importc: "metal_matmul_nt_off_enc", header: "metal_shim.h".}
  proc metal_init_c(kernelSrc: cstring): cint {.importc: "metal_init", header: "metal_shim.h".}
  proc metal_upload_bytes_c(data: ptr uint8, n: cint): MetalBufferHandle {.importc: "metal_upload_bytes", header: "metal_shim.h".}
  proc metal_upload_private_c(data: ptr float32, n: cint): MetalBufferHandle {.importc: "metal_upload_private", header: "metal_shim.h".}
  proc metal_upload_bytes_private_c(data: ptr uint8, n: cint): MetalBufferHandle {.importc: "metal_upload_bytes_private", header: "metal_shim.h".}
  proc metal_matmul_q4_enc_c(a, wq, scales, zeros, c: MetalBufferHandle, m, k, n, groupSize, nGroupsPerRow, wqOffset: cint): cint
    {.importc: "metal_matmul_q4_enc", header: "metal_shim.h".}
  proc metal_matmul_q4_batched_enc_c(a, wq, scales, zeros, c: MetalBufferHandle, m, k, n, groupSize, nGroupsPerRow, wqOffset: cint): cint
    {.importc: "metal_matmul_q4_batched_enc", header: "metal_shim.h".}
  proc metal_copy_into_enc_c(src, dst: MetalBufferHandle, srcOffsetElems, dstOffsetElems, n: cint): cint {.importc: "metal_copy_into_enc", header: "metal_shim.h".}
  proc metal_softmax_enc_c(x, y: MetalBufferHandle, rows, cols: cint): cint {.importc: "metal_softmax_enc", header: "metal_shim.h".}
  proc metal_rmsnorm_enc_c(x, gamma, y: MetalBufferHandle, rows, cols: cint, eps: cfloat): cint
    {.importc: "metal_rmsnorm_enc", header: "metal_shim.h".}
  proc metal_layernorm_enc_c(x, gamma, beta, y: MetalBufferHandle, rows, cols: cint, eps: cfloat): cint
    {.importc: "metal_layernorm_enc", header: "metal_shim.h".}
  proc metal_embedding_lookup_enc_c(table, indices, y: MetalBufferHandle, vocab, dim, numIndices: cint): cint
    {.importc: "metal_embedding_lookup_enc", header: "metal_shim.h".}
  proc metal_attention_fused_enc_c(q, k, v, o, sMatrix: MetalBufferHandle, B, H, S, D: cint, scale: cfloat): cint
    {.importc: "metal_attention_fused_enc", header: "metal_shim.h".}
  proc metal_attention_decode_gqa_enc_c(q, k, v, o, scores: MetalBufferHandle, H, S, D: cint, scale: cfloat): cint
    {.importc: "metal_attention_decode_gqa_enc", header: "metal_shim.h".}
  proc metal_attention_decode_all_heads_enc_c(q, kAll, vAll, o: MetalBufferHandle,
                                               H, HKV, D, S, maxSeqLen: cint): cint
    {.importc: "metal_attention_decode_all_heads_enc", header: "metal_shim.h".}
  proc metal_attention_prefill_flash_enc_c(q, kAll, vAll, o: MetalBufferHandle,
                                            T, H, HKV, D, pos, maxSeqLen: cint, scale: cfloat): cint
    {.importc: "metal_attention_prefill_flash_enc", header: "metal_shim.h".}
  proc metal_attention_prefill_flash_fp16_supported_c(): cint
    {.importc: "metal_attention_prefill_flash_fp16_supported", header: "metal_shim.h".}
  proc metal_kv_scatter_enc_c(src, dst: MetalBufferHandle, T, HKV, D, pos, maxSeqLen: cint): cint
    {.importc: "metal_kv_scatter_enc", header: "metal_shim.h".}
  proc metal_session_end_c(): cint {.importc: "metal_session_end", header: "metal_shim.h".}
  proc metal_get_last_session_error_c(): cstring {.importc: "metal_get_last_session_error", header: "metal_shim.h".}
  proc metal_session_commit_continue_c(): cint {.importc: "metal_session_commit_continue", header: "metal_shim.h".}
  proc metal_buffer_read_c(h: MetalBufferHandle, outData: ptr float32, n: cint): cint {.importc: "metal_buffer_read", header: "metal_shim.h".}
  proc metal_buffer_free_c(h: MetalBufferHandle) {.importc: "metal_buffer_free", header: "metal_shim.h".}
  # NEW: 100% sessionised - the remaining ops (sqrt/exp/log/pow, apflu,
  # apflu_backward, layernorm_backward, attention_backward, gather, argmax,
  # compare, dropout_mask, permute, split_slice, concat).
  proc metal_activation_enc_p_c(op: cint, x, y: MetalBufferHandle, n: cint, p: cfloat): cint
    {.importc: "metal_activation_enc_p", header: "metal_shim.h".}
  proc metal_scale_enc_c(x: MetalBufferHandle, scalar: cfloat, n: cint): cint
    {.importc: "metal_scale_enc", header: "metal_shim.h".}
  proc metal_activation_backward_enc_c(op: cint, x, dy, dx: MetalBufferHandle, n: cint): cint
    {.importc: "metal_activation_backward_enc", header: "metal_shim.h".}
  proc metal_layernorm_backward_enc_c(dy, x, gamma, beta, dx, dgamma, dbeta: MetalBufferHandle,
                                         rows, cols: cint, eps: cfloat): cint
      {.importc: "metal_layernorm_backward_enc", header: "metal_shim.h".}
  proc metal_attention_fused_backward_enc_c(q, k, v, sMatrix, dy, dq, dk, dv: MetalBufferHandle,
                                             B, H, S, D: cint, scale: cfloat): cint
    {.importc: "metal_attention_fused_backward_enc", header: "metal_shim.h".}
  proc metal_gather_enc_c(src, idx, dst: MetalBufferHandle, n: cint): cint
    {.importc: "metal_gather_enc", header: "metal_shim.h".}
  proc metal_argmax_enc_c(x, outVal, outIdx: MetalBufferHandle, outer, dimSize, inner: cint): cint
    {.importc: "metal_argmax_enc", header: "metal_shim.h".}
  proc metal_compare_enc_c(a, b, mask: MetalBufferHandle, op, n: cint): cint
    {.importc: "metal_compare_enc", header: "metal_shim.h".}
  proc metal_dropout_mask_enc_c(mask: MetalBufferHandle, seed: cuint, p, scale: cfloat, n: cint): cint
    {.importc: "metal_dropout_mask_enc", header: "metal_shim.h".}
  proc metal_permute_enc_c(src, dst, newStrides, oldStrides, dimsMap: MetalBufferHandle, ndim, n: cint): cint
    {.importc: "metal_permute_enc", header: "metal_shim.h".}
  proc metal_split_slice_enc_c(src, dst: MetalBufferHandle, dimTotal, thisSize, offset, inner, n: cint): cint
    {.importc: "metal_split_slice_enc", header: "metal_shim.h".}
  proc metal_concat_gather_enc_c(flatSrc, srcBase, srcDimSize, srcDimOffset: MetalBufferHandle,
                                  numTensors: cint, dst: MetalBufferHandle, totalDim, inner, n: cint): cint
    {.importc: "metal_concat_gather_enc", header: "metal_shim.h".}

  proc metalInit*(): bool =
    ## Calls ensureMetalInit() explicitly (dispatch_once inside - cheap to call
    ## repeatedly). MUST be called before metalUploadBytes/metalSessionUpload if
    ## no metalSessionBegin has run yet (exactly the resident-upload case
    ## while loading the model, before the decode loop).
    metal_init_c(kMetalKernelSrc.cstring) != 0

  proc metalSessionBegin*(): bool =
    var err: cstring = nil
    let ok = metal_session_begin_c(kMetalKernelSrc.cstring, addr err) != 0
    if err != nil: c_free(err)
    ok

  proc metalSessionUpload*(data: seq[float32]): MetalBufferHandle =
    discard metalInit()  # CHANGED: safe for a resident upload BEFORE the first session begin
    var d = data
    metal_upload_c(addr d[0], cint(d.len))

  proc metalSessionUploadBorrowed*(data: var seq[float32]): MetalBufferHandle =
    ## Session-only path. Caller owns `data` until session read/free completes;
    ## C tries a page-aligned Shared no-copy mapping and otherwise copies.
    if data.len == 0: return nil
    discard metalInit()
    metal_upload_borrowed_c(addr data[0], cint(data.len))

  proc metalSessionAllocScratch*(n: int): MetalBufferHandle =
    metal_alloc_scratch_c(cint(n))

  proc metalAllocHeap*(bytes: int): pointer =
    ## NEW: allocates one large region (device level, one fixed cost)
    ## and metalAllocFromHeap() suballocates many CHEAP child buffers from it - used
    ## for the resident KV (see ensureKvResident in nim_inference.nim) to
    ## avoid thousands of metalSessionAllocScratch calls (= one device allocation each)
    ## at startup.
    metal_alloc_heap_c(clong(bytes))

  proc metalAllocFromHeap*(heap: pointer, n: int): MetalBufferHandle =
    metal_alloc_from_heap_c(heap, cint(n))

  proc metalFreeHeap*(heap: pointer) =
    metal_free_heap_c(heap)

  proc metalSessionMatmulEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool =
    metal_matmul_enc_c(a, b, c, cint(m), cint(k), cint(n)) != 0

  proc metal_matmul_mps_nt_enc_c(a, b, c: MetalBufferHandle;
      m, k, n, stride, offset: cint): cint {.importc: "metal_matmul_mps_nt_enc", header: "metal_shim.h".}
  proc metal_mps_supported_c(): cint {.importc: "metal_mps_supported", header: "metal_shim.h".}
  proc metal_is_apple_silicon_c(): cint {.importc: "metal_is_apple_silicon", header: "metal_shim.h".}
  proc metalMpsSupported*(): bool = metal_mps_supported_c() != 0
  proc metalIsAppleSilicon*(): bool = metal_is_apple_silicon_c() != 0

  proc metalSessionMatmulMpsNTEnc*(a, b, c: MetalBufferHandle;
      m, k, n, stride, offset: int): bool =
    # Single-token decode is strictly native GEMV/compute, never MPS.
    if m <= 1: return false
    metal_matmul_mps_nt_enc_c(a, b, c, cint(m), cint(k), cint(n), cint(stride), cint(offset)) != 0

  proc metalSessionMatmulNTEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool =
    ## C(m,n) = A(m,k) @ B(n,k)^T - B KEEPS its original [n,k] layout (no
    ## CPU transpose first). Used by attention decode (K/V cache).
    metal_matmul_nt_enc_c(a, b, c, cint(m), cint(k), cint(n)) != 0

  proc metalSessionMatmulNTEnc*(a, b, c: MetalBufferHandle, m, k, n, bOffsetBytes: int): bool =
    ## B starts bOffsetBytes (a multiple of 4) into its buffer, e.g. a weight
    ## wrapped inside a model mmap by metalWrapHostRange.
    if bOffsetBytes == 0: return metalSessionMatmulNTEnc(a, b, c, m, k, n)
    metal_matmul_nt_off_enc_c(a, b, cint(bOffsetBytes), c, cint(m), cint(k), cint(n)) != 0

  proc metalUploadBytes*(data: openArray[uint8]): MetalBufferHandle =
    discard metalInit()
    if data.len == 0: return nil
    metal_upload_bytes_c(unsafeAddr data[0], cint(data.len))

  proc metalUploadPrivate*(data: openArray[float32]): MetalBufferHandle =
    ## Private-storage variant of metalSessionUpload - ONLY for resident
    ## fp32 weights (int8/int4-symmetric/tied embedding through beResidentUploadFp32)
    ## that the CPU never reads/writes after the upload.
    discard metalInit()
    if data.len == 0: return nil
    metal_upload_private_c(unsafeAddr data[0], cint(data.len))

  proc metalUploadBytesPrivate*(data: seq[uint8]): MetalBufferHandle =
    ## Private-storage variant of metalUploadBytes - for resident wq (packed Q4).
    discard metalInit()
    if data.len == 0: return nil
    var d = data
    metal_upload_bytes_private_c(addr d[0], cint(d.len))

  proc metalUploadBytesPrivatePtr*(data: pointer; bytes: int): MetalBufferHandle =
    ## metalUploadBytesPrivate straight from caller memory (e.g. a model mmap),
    ## without first copying it into a Nim seq.
    discard metalInit()
    if data == nil or bytes <= 0 or bytes > high(cint): return nil
    metal_upload_bytes_private_c(cast[ptr uint8](data), cint(bytes))

  proc metalWrapHostNoCopy*(data: pointer; bytes: int): MetalBufferHandle =
    ## See metal_wrap_host_nocopy: nil means copy with metalUploadBytesPrivatePtr.
    discard metalInit()
    if data == nil or bytes <= 0: return nil
    metal_wrap_host_nocopy_c(data, csize_t(bytes))

  proc getpagesize(): cint {.importc, header: "<unistd.h>".}

  proc metalWrapHostRange*(data: pointer; bytes: int): tuple[buf: MetalBufferHandle, offset: int] =
    ## metalWrapHostNoCopy for data at any address inside a mapping (a tensor
    ## in an unaligned .nimq): the buffer starts at the enclosing page and
    ## `offset` is the byte position of `data` inside it.
    if data == nil or bytes <= 0: return
    let offset = int(cast[uint](data) mod uint(getpagesize()))
    result.buf = metalWrapHostNoCopy(cast[pointer](cast[uint](data) - uint(offset)), offset + bytes)
    if result.buf != nil: result.offset = offset

  proc metalSessionMatmulQ4Enc*(a, wq, scales, zeros, c: MetalBufferHandle,
                                 m, k, n, groupSize, nGroupsPerRow: int; wqOffset = 0): bool =
    ## wqOffset: byte position of the packed weights inside wq.
    metal_matmul_q4_enc_c(a, wq, scales, zeros, c,
      cint(m), cint(k), cint(n), cint(groupSize), cint(nGroupsPerRow), cint(wqOffset)) != 0

  proc metal_prepare_q4_tuned_c(w, scales, zeros: MetalBufferHandle;
      k, n, groupSize, groupsPerRow, m, wqOffset: cint): cint
      {.importc: "metal_prepare_q4_tuned", header: "metal_shim.h".}
  proc metalPrepareQ4Tuned*(w, scales, zeros: MetalBufferHandle;
      k, n, groupSize, groupsPerRow, m: int; wqOffset = 0): bool =
    metal_prepare_q4_tuned_c(w, scales, zeros, cint(k), cint(n),
      cint(groupSize), cint(groupsPerRow), cint(m), cint(wqOffset)) != 0

  proc metalSessionMatmulQ4BatchedEnc*(a, wq, scales, zeros, c: MetalBufferHandle,
                                        m, k, n, groupSize, nGroupsPerRow: int; wqOffset = 0): bool =
    ## NEW: only for M>1 (prefill) - see matmul_q4_batched.metal.
    metal_matmul_q4_batched_enc_c(a, wq, scales, zeros, c,
      cint(m), cint(k), cint(n), cint(groupSize), cint(nGroupsPerRow), cint(wqOffset)) != 0

  proc metalSessionCopyIntoEnc*(src, dst: MetalBufferHandle, srcOffsetElems, dstOffsetElems, n: int): bool =
    metal_copy_into_enc_c(src, dst, cint(srcOffsetElems), cint(dstOffsetElems), cint(n)) != 0
  proc metalSessionCopyIntoEnc*(src, dst: MetalBufferHandle, dstOffsetElems, n: int): bool =
    ## Keeps the old overload (srcOffsetElems=0) so callers that already worked
    ## (e.g. merge oProj) need no change - 100% backward compatible.
    metalSessionCopyIntoEnc(src, dst, 0, dstOffsetElems, n)

  proc metalSessionSoftmaxEnc*(x, y: MetalBufferHandle, rows, cols: int): bool =
    metal_softmax_enc_c(x, y, cint(rows), cint(cols)) != 0

  proc metalSessionRmsnormEnc*(x, gamma, y: MetalBufferHandle, rows, cols: int, eps: float32): bool =
    metal_rmsnorm_enc_c(x, gamma, y, cint(rows), cint(cols), cfloat(eps)) != 0

  proc metalSessionLayernormEnc*(x, gamma, beta, y: MetalBufferHandle, rows, cols: int, eps: float32): bool =
    metal_layernorm_enc_c(x, gamma, beta, y, cint(rows), cint(cols), cfloat(eps)) != 0

  proc metalSessionEmbeddingLookupEnc*(table, indices, y: MetalBufferHandle, vocab, dim, numIndices: int): bool =
    metal_embedding_lookup_enc_c(table, indices, y, cint(vocab), cint(dim), cint(numIndices)) != 0

  proc metalSessionAttentionFusedEnc*(q, k, v, o, sMatrix: MetalBufferHandle, B, H, S, D: int, scale: float32): bool =
    metal_attention_fused_enc_c(q, k, v, o, sMatrix, cint(B), cint(H), cint(S), cint(D), cfloat(scale)) != 0

  proc metalSessionAttentionDecodeGqaEnc*(q, k, v, o, scores: MetalBufferHandle, H, S, D: int, scale: float32): bool =
    ## NEW: 1-query/head attention decode for one GQA group (see the full explanation
    ## in attention_decode_gqa_kernel in vecop_matmul.metal). Called once
    ## per KV-head pair in forwardDecode, replacing the manual
    ## matmul+softmax+matmul loop - noticeably fewer encodes per token.
    metal_attention_decode_gqa_enc_c(q, k, v, o, scores, cint(H), cint(S), cint(D), cfloat(scale)) != 0

  proc metalSessionAttentionDecodeAllHeadsEnc*(q, kAll, vAll, o: MetalBufferHandle,
                                                H, HKV, D, S, maxSeqLen: int): bool =
    ## NEW: FUSED attention decode for all H heads in one dispatch (see
    ## attention_decode_all_heads.metal). kAll/vAll MUST be buffers CONTIGUOUS
    ## per head (offset of head hkv = hkv*maxSeqLen*D) - use
    ## kResidentCombined/vResidentCombined from nim_inference.nim, NOT
    ## kResident/vResident (32 separate buffers suballocated from a heap).
    metal_attention_decode_all_heads_enc_c(q, kAll, vAll, o, cint(H), cint(HKV), cint(D),
                                            cint(S), cint(maxSeqLen)) != 0

  proc metalSessionFlashAttentionPrefillEnc*(q, kAll, vAll, o: MetalBufferHandle;
                                             T, H, HKV, D, pos, maxSeqLen: int; scale: float32): bool =
    ## Causal prefill flash attention (attention_prefill_flash.metal), D <= 128.
    metal_attention_prefill_flash_enc_c(q, kAll, vAll, o, cint(T), cint(H), cint(HKV), cint(D),
                                        cint(pos), cint(maxSeqLen), cfloat(scale)) != 0

  proc metalSessionFlashAttentionPrepare*(D: int): string =
    if D > 0 and D <= 128 and metal_attention_prefill_flash_fp16_supported_c() != 0:
      "scalar FP16"
    elif D > 0 and D <= 128: "scalar FP32"
    else: ""

  proc metalSessionKvScatterEnc*(src, dst: MetalBufferHandle; T, HKV, D, pos, maxSeqLen: int): bool =
    metal_kv_scatter_enc_c(src, dst, cint(T), cint(HKV), cint(D), cint(pos), cint(maxSeqLen)) != 0

  proc metalSessionActivationPEnc*(op: int, x, y: MetalBufferHandle, n: int, p: float32): bool =
    metal_activation_enc_p_c(cint(op), x, y, cint(n), cfloat(p)) != 0

  proc metalSessionScaleEnc*(x: MetalBufferHandle, scalar: float32, n: int): bool =
    metal_scale_enc_c(x, cfloat(scalar), cint(n)) != 0

  proc metalSessionActivationBackwardEnc*(op: int, x, dy, dx: MetalBufferHandle, n: int): bool =
    metal_activation_backward_enc_c(cint(op), x, dy, dx, cint(n)) != 0

  proc metalSessionLayernormBackwardEnc*(dy, x, gamma, beta, dx, dgamma, dbeta: MetalBufferHandle,
                                          rows, cols: int, eps: float32): bool =
    metal_layernorm_backward_enc_c(dy, x, gamma, beta, dx, dgamma, dbeta, cint(rows), cint(cols), cfloat(eps)) != 0

  proc metalSessionAttentionFusedBackwardEnc*(q, k, v, sMatrix, dy, dq, dk, dv: MetalBufferHandle,
                                               B, H, S, D: int, scale: float32): bool =
    metal_attention_fused_backward_enc_c(q, k, v, sMatrix, dy, dq, dk, dv, cint(B), cint(H), cint(S), cint(D), cfloat(scale)) != 0

  proc metalSessionGatherEnc*(src, idx, dst: MetalBufferHandle, n: int): bool =
    metal_gather_enc_c(src, idx, dst, cint(n)) != 0

  proc metalSessionArgmaxEnc*(x, outVal, outIdx: MetalBufferHandle, outer, dimSize, inner: int): bool =
    metal_argmax_enc_c(x, outVal, outIdx, cint(outer), cint(dimSize), cint(inner)) != 0

  proc metalSessionCompareEnc*(a, b, mask: MetalBufferHandle, op, n: int): bool =
    metal_compare_enc_c(a, b, mask, cint(op), cint(n)) != 0

  proc metalSessionDropoutMaskEnc*(mask: MetalBufferHandle, seed: uint32, p, scale: float32, n: int): bool =
    metal_dropout_mask_enc_c(mask, cuint(seed), cfloat(p), cfloat(scale), cint(n)) != 0

  proc metalSessionPermuteEnc*(src, dst, newStrides, oldStrides, dimsMap: MetalBufferHandle, ndim, n: int): bool =
    metal_permute_enc_c(src, dst, newStrides, oldStrides, dimsMap, cint(ndim), cint(n)) != 0

  proc metalSessionSplitSliceEnc*(src, dst: MetalBufferHandle, dimTotal, thisSize, offset, inner, n: int): bool =
    metal_split_slice_enc_c(src, dst, cint(dimTotal), cint(thisSize), cint(offset), cint(inner), cint(n)) != 0

  proc metalSessionConcatEnc*(flatSrc, srcBase, srcDimSize, srcDimOffset: MetalBufferHandle,
                               numTensors: int, dst: MetalBufferHandle, totalDim, inner, n: int): bool =
    metal_concat_gather_enc_c(flatSrc, srcBase, srcDimSize, srcDimOffset, cint(numTensors), dst, cint(totalDim), cint(inner), cint(n)) != 0

  proc metalSessionEnd*(): bool =
    metal_session_end_c() != 0

  proc metalGetLastSessionError*(): string =
    $metal_get_last_session_error_c()   ## '$' copy ra Nim string NGAY - an
    ## safe even if the original C pointer is overwritten by the next metal_session_end().

  proc metalRecoverQueue*(): bool =
    ## NEW: call when metalSessionEnd() returns false - recreates the MTLCommandQueue
    ## (see metal_shim.m::metal_recover_queue). After a command buffer
    ## REALLY fails (Timeout/StackOverflow/...), the old queue is "poisoned" and
    ## EVERY later buffer is rejected by the driver ("Ignored") - without
    ## this call EVERY later token reads GARBAGE logits (the buffers never
    ## ran on the GPU) without any clear error.
    metal_recover_queue_c() != 0

  proc metalSessionCommitContinue*(): bool =
    ## NEW: "commitAndContinue" - see the full explanation in
    ## metal_shim.m::metal_session_commit_continue. This is the ROOT FIX for GPU
    ## timeouts (following Apple's MPSCommandBuffer / the PyTorch MPS backend),
    ## instead of a real flush-and-wait (slow) or one huge buffer (watchdog kill).
    metal_session_commit_continue_c() != 0

  proc metalSessionRead*(h: MetalBufferHandle, n: int): seq[float32] =
    result = newSeq[float32](n)
    let ok = metal_buffer_read_c(h, addr result[0], cint(n))
    if ok == 0:
      raise newException(CatchableError, "metal_buffer_read failed")

  proc metalSessionFree*(h: MetalBufferHandle) =
    metal_buffer_free_c(h)

when not defined(macosx):
  type MetalBufferHandle* = pointer
  proc metalSessionBegin*(): bool = false
  proc metalSessionUpload*(data: seq[float32]): MetalBufferHandle = nil
  proc metalSessionUploadBorrowed*(data: var seq[float32]): MetalBufferHandle = nil
  proc metalSessionAllocScratch*(n: int): MetalBufferHandle = nil
  proc metalAllocHeap*(bytes: int): pointer = nil
  proc metalAllocFromHeap*(heap: pointer, n: int): MetalBufferHandle = nil
  proc metalFreeHeap*(heap: pointer) = discard
  proc metalSessionMatmulEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool = false
  proc metalSessionMatmulNTEnc*(a, b, c: MetalBufferHandle, m, k, n: int): bool = false
  proc metalInit*(): bool = false
  proc metalUploadBytes*(data: openArray[uint8]): MetalBufferHandle = nil
  proc metalUploadPrivate*(data: openArray[float32]): MetalBufferHandle = nil
  proc metalUploadBytesPrivate*(data: seq[uint8]): MetalBufferHandle = nil
  proc metalUploadBytesPrivatePtr*(data: pointer; bytes: int): MetalBufferHandle = nil
  proc metalWrapHostNoCopy*(data: pointer; bytes: int): MetalBufferHandle = nil
  proc metalWrapHostRange*(data: pointer; bytes: int): tuple[buf: MetalBufferHandle, offset: int] = discard
  proc metalSessionMatmulNTEnc*(a, b, c: MetalBufferHandle, m, k, n, bOffsetBytes: int): bool = false
  proc metalSessionMatmulQ4Enc*(a, wq, scales, zeros, c: MetalBufferHandle,
                                 m, k, n, groupSize, nGroupsPerRow: int; wqOffset = 0): bool = false
  proc metalSessionMatmulQ4BatchedEnc*(a, wq, scales, zeros, c: MetalBufferHandle,
                                        m, k, n, groupSize, nGroupsPerRow: int; wqOffset = 0): bool = false
  proc metalSessionCopyIntoEnc*(src, dst: MetalBufferHandle, srcOffsetElems, dstOffsetElems, n: int): bool = false
  proc metalSessionCopyIntoEnc*(src, dst: MetalBufferHandle, dstOffsetElems, n: int): bool = false
  proc metalSessionSoftmaxEnc*(x, y: MetalBufferHandle, rows, cols: int): bool = false
  proc metalSessionRmsnormEnc*(x, gamma, y: MetalBufferHandle, rows, cols: int, eps: float32): bool = false
  proc metalSessionLayernormEnc*(x, gamma, beta, y: MetalBufferHandle, rows, cols: int, eps: float32): bool = false
  proc metalSessionEmbeddingLookupEnc*(table, indices, y: MetalBufferHandle, vocab, dim, numIndices: int): bool = false
  proc metalSessionAttentionFusedEnc*(q, k, v, o, sMatrix: MetalBufferHandle, B, H, S, D: int, scale: float32): bool = false
  proc metalSessionAttentionDecodeGqaEnc*(q, k, v, o, scores: MetalBufferHandle, H, S, D: int, scale: float32): bool = false
  proc metalSessionAttentionDecodeAllHeadsEnc*(q, kAll, vAll, o: MetalBufferHandle,
                                                H, HKV, D, S, maxSeqLen: int): bool = false
  proc metalSessionFlashAttentionPrefillEnc*(q, kAll, vAll, o: MetalBufferHandle;
                                             T, H, HKV, D, pos, maxSeqLen: int; scale: float32): bool = false
  proc metalSessionKvScatterEnc*(src, dst: MetalBufferHandle; T, HKV, D, pos, maxSeqLen: int): bool = false
  proc metalSessionActivationPEnc*(op: int, x, y: MetalBufferHandle, n: int, p: float32): bool = false
  proc metalSessionScaleEnc*(x: MetalBufferHandle, scalar: float32, n: int): bool = false
  proc metalSessionActivationBackwardEnc*(op: int, x, dy, dx: MetalBufferHandle, n: int): bool = false
  proc metalSessionLayernormBackwardEnc*(dy, x, gamma, beta, dx, dgamma, dbeta: MetalBufferHandle,
                                          rows, cols: int, eps: float32): bool = false
  proc metalSessionAttentionFusedBackwardEnc*(q, k, v, sMatrix, dy, dq, dk, dv: MetalBufferHandle,
                                               B, H, S, D: int, scale: float32): bool = false
  proc metalSessionGatherEnc*(src, idx, dst: MetalBufferHandle, n: int): bool = false
  proc metalSessionArgmaxEnc*(x, outVal, outIdx: MetalBufferHandle, outer, dimSize, inner: int): bool = false
  proc metalSessionCompareEnc*(a, b, mask: MetalBufferHandle, op, n: int): bool = false
  proc metalSessionDropoutMaskEnc*(mask: MetalBufferHandle, seed: uint32, p, scale: float32, n: int): bool = false
  proc metalSessionPermuteEnc*(src, dst, newStrides, oldStrides, dimsMap: MetalBufferHandle, ndim, n: int): bool = false
  proc metalSessionSplitSliceEnc*(src, dst: MetalBufferHandle, dimTotal, thisSize, offset, inner, n: int): bool = false
  proc metalSessionConcatEnc*(flatSrc, srcBase, srcDimSize, srcDimOffset: MetalBufferHandle,
                               numTensors: int, dst: MetalBufferHandle, totalDim, inner, n: int): bool = false
  proc metalSessionEnd*(): bool = false
  proc metalGetLastSessionError*(): string = ""
  proc metalRecoverQueue*(): bool = false
  proc metalSessionCommitContinue*(): bool = false
  proc metalSessionRead*(h: MetalBufferHandle, n: int): seq[float32] = @[]
  proc metalSessionFree*(h: MetalBufferHandle) = discard

  proc metalAvailable*(): bool = false
  proc metalIsAppleGpu*(): bool = false
  proc metalVecOp*(op: string, a, b: seq[float32]): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalMatmul2*(a1, b1: seq[float32], m1, k1, n1: int,
                      a2, b2: seq[float32], m2, k2, n2: int):
                      tuple[c1, c2: seq[float32]] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalMatmulQ4*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                       m, k, n, groupSize, nGroupsPerRow: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalActivation*(op: string, x: seq[float32], p: float32 = 0.0'f32): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalGather*(src: seq[float32], idx: seq[int32]): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalArgmaxIndex*(x: seq[float32], outer, dimSize, inner: int): tuple[val: seq[float32], idx: seq[int32]] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalCompare*(a, b: seq[float32], op: int): seq[uint8] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalDropoutMask*(n: int, p: float32, seed: uint32): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalPermute*(src: seq[float32], newStrides, oldStrides, dimsMap: seq[int32], n: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalSplitSlice*(src: seq[float32], dimTotal, thisSize, offset, inner, n: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalConcat*(flatSrc: seq[float32], srcBase, srcDimSize, srcDimOffset: seq[int32],
                     totalDim, inner, n: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")
  proc metalAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
    raise newException(CatchableError, "The Metal backend is only available on macOS")

when defined(macosx):
  proc metalDeviceMaxTgThreads*(): cint {.importc: "metal_device_max_tg_threads", header: "metal_shim.h".}
  proc metalDeviceMaxTgBytes*(): cint {.importc: "metal_device_max_tg_bytes", header: "metal_shim.h".}
