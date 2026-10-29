## CPU backend for BybyLang / Nimformers.
##
## Thin, allocation-friendly wrapper around the runtime-dispatched C kernels
## in this directory.  The C constructor performs feature detection and sets
## up SSE4/AVX/AVX2+FMA/AVX-512/NEON tables; this module only marshals Nim
## sequences and exposes the CPU micro-VM.
##
## Environment overrides:
##   NIMFORMERS_CPU_ISA=scalar|sse|avx|avx2|avx512|auto
##   NIMFORMERS_CPU_SMALL=<element threshold for preferring SSE>
##   NIMFORMERS_CPU_AVX512_MIN=<minimum work for AVX-512>
##   NIMFORMERS_CPU_BLOCK_NC / _KC / _MC

import std/os

const
  CpuBackendDir = currentSourcePath().parentDir
  CpuHeader = "cpu_kernels.h"

{.passC: "-I" & CpuBackendDir.}
{.passC: "-O3".}
{.passC: "-std=c11".}
{.compile: CpuBackendDir / "cpu_kernels.c".}
{.compile: CpuBackendDir / "cpu_vm.c".}

type
  NimfCpuInfoC {.importc: "NimfCpuInfo", header: CpuHeader, bycopy.} = object
    hasSse4 {.importc: "has_sse4".}: cint
    hasAvx {.importc: "has_avx".}: cint
    hasAvx2 {.importc: "has_avx2".}: cint
    hasFma {.importc: "has_fma".}: cint
    hasAvx512 {.importc: "has_avx512".}: cint
    hasNeon {.importc: "has_neon".}: cint
    isa {.importc: "isa".}: cint
    smallThreshold {.importc: "small_threshold".}: cint
    avx512Threshold {.importc: "avx512_threshold".}: cint
    l1Bytes {.importc: "l1_bytes".}: csize_t
    l2Bytes {.importc: "l2_bytes".}: csize_t
    l3Bytes {.importc: "l3_bytes".}: csize_t

  CpuInfo* = object
    isa*: int
    isaName*: string
    hasSse4*, hasAvx*, hasAvx2*, hasFma*, hasAvx512*, hasNeon*: bool
    smallThreshold*, avx512Threshold*: int
    l1Bytes*, l2Bytes*, l3Bytes*: int

  CpuFeatures* = tuple[sse4, avx, avx2, fma, avx512, neon: bool]

var cpuLastError {.threadvar.}: string

proc nimf_cpu_init() {.importc, cdecl.}
proc nimf_cpu_info(): ptr NimfCpuInfoC {.importc, cdecl.}
proc nimf_cpu_current_isa(): cint {.importc, cdecl.}
proc nimf_cpu_backend_block_sizes(mc, nc, kc: ptr cint) {.importc, cdecl.}

proc cpuInit*() {.inline.} =
  nimf_cpu_init()

proc cpuInfo*(): CpuInfo =
  cpuInit()
  let p = nimf_cpu_info()
  result.isa = int(p.isa)
  result.isaName = case result.isa
    of 0: "scalar"
    of 1: "sse4"
    of 2: "avx"
    of 3: "avx2"
    of 4: "avx512"
    else: "unknown"
  result.hasSse4 = p.hasSse4 != 0
  result.hasAvx = p.hasAvx != 0
  result.hasAvx2 = p.hasAvx2 != 0
  result.hasFma = p.hasFma != 0
  result.hasAvx512 = p.hasAvx512 != 0
  result.hasNeon = p.hasNeon != 0
  result.smallThreshold = int(p.smallThreshold)
  result.avx512Threshold = int(p.avx512Threshold)
  result.l1Bytes = int(p.l1Bytes)
  result.l2Bytes = int(p.l2Bytes)
  result.l3Bytes = int(p.l3Bytes)

proc cpuFeatures*(): CpuFeatures =
  let i = cpuInfo()
  (i.hasSse4, i.hasAvx, i.hasAvx2, i.hasFma, i.hasAvx512, i.hasNeon)

proc cpuIsaName*(): string = cpuInfo().isaName
proc cpuLastErrorMessage*(): string = cpuLastError

proc cpuBlockSizes*(): tuple[mc, nc, kc: int] =
  cpuInit()
  var mc, nc, kc: cint
  nimf_cpu_backend_block_sizes(addr mc, addr nc, addr kc)
  (int(mc), int(nc), int(kc))

# ---------------------------------------------------------------------
# C entry points
# ---------------------------------------------------------------------

proc nimf_cpu_add(a, b, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_sub(a, b, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_mul(a, b, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_div(a, b, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_relu(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_sqrt(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_exp(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_log(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_sigmoid(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_tanh(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_neg(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_abs(x, dst: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_pow(x, dst: ptr float32, n: csize_t, p: cfloat) {.importc, cdecl.}
proc nimf_cpu_add_scalar(x, dst: ptr float32, n: csize_t, s: cfloat) {.importc, cdecl.}
proc nimf_cpu_mul_scalar(x, dst: ptr float32, n: csize_t, s: cfloat) {.importc, cdecl.}
proc nimf_cpu_apflu(x, dst: ptr float32, n: csize_t, alpha, beta: cfloat) {.importc, cdecl.}
proc nimf_cpu_apflu_backward(x, dy, dst: ptr float32, n: csize_t,
                             alpha, beta: cfloat) {.importc, cdecl.}
proc nimf_cpu_fused_add_act(a, b, dst: ptr float32, n: csize_t, act: cint) {.importc, cdecl.}
proc nimf_cpu_copy(dst, src: ptr float32, n: csize_t) {.importc, cdecl.}
proc nimf_cpu_scale(x: ptr float32, n: csize_t, s: cfloat) {.importc, cdecl.}
proc nimf_cpu_softmax(x, dst: ptr float32, rows, cols: cint) {.importc, cdecl.}
proc nimf_cpu_layernorm(x, gamma, beta, dst: ptr float32, rows, cols: cint,
                        eps: cfloat) {.importc, cdecl.}
proc nimf_cpu_rmsnorm(x, gamma, dst: ptr float32, rows, cols: cint,
                      eps: cfloat) {.importc, cdecl.}
proc nimf_cpu_layernorm_backward(dy, x, gamma, beta: ptr float32,
                                 rows, cols: cint, eps: cfloat,
                                 dx, dgamma, dbeta: ptr float32) {.importc, cdecl.}
proc nimf_cpu_matmul(a, b, c: ptr float32, M, K, N: cint) {.importc, cdecl.}
proc nimf_cpu_matmul_nt(a, b, c: ptr float32, M, K, N: cint) {.importc, cdecl.}
proc nimf_cpu_matmul_q4(a: ptr float32, wq: ptr uint8, scales, zeros, c: ptr float32,
                        M, K, N, groupSize, groupsPerRow: cint) {.importc, cdecl.}
proc nimf_cpu_attention(q, k, v, mask: ptr float32, maskLen: cint,
                        dst, scores: ptr float32,
                        B, H, S, D: cint, scale: cfloat) {.importc, cdecl.}
proc nimf_cpu_embedding(table: ptr float32, indices: ptr int32, dst: ptr float32,
                        numIdx, vocab, dim: cint) {.importc, cdecl.}
proc nimf_cpu_gather(src: ptr float32, idx: ptr int32, dst: ptr float32, n: cint) {.importc, cdecl.}
proc nimf_cpu_compare(a, b: ptr float32, mask: ptr uint8, n: cint, op: cint) {.importc, cdecl.}
proc nimf_cpu_dropout_mask(dst: ptr float32, n: cint, p: cfloat, seed: uint32) {.importc, cdecl.}
proc nimf_cpu_argmax(x, outVal: ptr float32, outIdx: ptr int32,
                     outer, dimSize, inner: cint) {.importc, cdecl.}
proc nimf_cpu_permute(src, dst: ptr float32, newStrides, oldStrides, dimsMap: ptr int32,
                      ndim, n: cint) {.importc, cdecl.}
proc nimf_cpu_split_slice(src, dst: ptr float32, dimTotal, thisSize, offset,
                          inner, n: cint) {.importc, cdecl.}
proc nimf_cpu_concat(flatSrc, dst: ptr float32, srcBase, srcDimSize,
                     srcDimOffset: ptr int32, numTensors, totalDim, inner,
                     n: cint) {.importc, cdecl.}
proc nimf_cpu_rope(x, cosT, sinT: ptr float32, T, H, D: cint) {.importc, cdecl.}
proc nimf_cpu_vm_run(code: ptr uint32, nWords: cint, mem: ptr float32,
                     memLen: cint, inF32: ptr float32, inLen: cint,
                     nThreads, tgFloats, tgSize: cint, err: cstring,
                     errLen: cint): cint {.importc, cdecl.}

# ---------------------------------------------------------------------
# Nim-friendly wrappers
# ---------------------------------------------------------------------

proc add*(a, b: openArray[float32]): seq[float32] =
  result = newSeq[float32](a.len)
  if a.len > 0: nimf_cpu_add(unsafeAddr a[0], unsafeAddr b[0], addr result[0], csize_t(a.len))

proc sub*(a, b: openArray[float32]): seq[float32] =
  result = newSeq[float32](a.len)
  if a.len > 0: nimf_cpu_sub(unsafeAddr a[0], unsafeAddr b[0], addr result[0], csize_t(a.len))

proc mul*(a, b: openArray[float32]): seq[float32] =
  result = newSeq[float32](a.len)
  if a.len > 0: nimf_cpu_mul(unsafeAddr a[0], unsafeAddr b[0], addr result[0], csize_t(a.len))

proc divide*(a, b: openArray[float32]): seq[float32] =
  result = newSeq[float32](a.len)
  if a.len > 0: nimf_cpu_div(unsafeAddr a[0], unsafeAddr b[0], addr result[0], csize_t(a.len))

proc relu*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_relu(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc sqrt*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_sqrt(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc exp*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_exp(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc log*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_log(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc sigmoid*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_sigmoid(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc tanh*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_tanh(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc neg*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_neg(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc abs*(x: openArray[float32]): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_abs(unsafeAddr x[0], addr result[0], csize_t(x.len))

proc pow*(x: openArray[float32], p: float32): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_pow(unsafeAddr x[0], addr result[0], csize_t(x.len), cfloat(p))

proc addScalar*(x: openArray[float32], s: float32): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_add_scalar(unsafeAddr x[0], addr result[0], csize_t(x.len), cfloat(s))

proc mulScalar*(x: openArray[float32], s: float32): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0: nimf_cpu_mul_scalar(unsafeAddr x[0], addr result[0], csize_t(x.len), cfloat(s))

proc apflu*(x: openArray[float32], alpha, beta: float32): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0:
    nimf_cpu_apflu(unsafeAddr x[0], addr result[0], csize_t(x.len), cfloat(alpha), cfloat(beta))

proc apfluBackward*(x, dy: openArray[float32], alpha, beta: float32): seq[float32] =
  result = newSeq[float32](x.len)
  if x.len > 0:
    nimf_cpu_apflu_backward(unsafeAddr x[0], unsafeAddr dy[0], addr result[0],
      csize_t(x.len), cfloat(alpha), cfloat(beta))

proc fusedAddAct*(a, b: openArray[float32], act: int): seq[float32] =
  result = newSeq[float32](a.len)
  if a.len > 0:
    nimf_cpu_fused_add_act(unsafeAddr a[0], unsafeAddr b[0], addr result[0],
      csize_t(a.len), cint(act))

proc copyInto*(dst: var seq[float32], src: openArray[float32]) =
  let n = min(dst.len, src.len)
  if n > 0: nimf_cpu_copy(addr dst[0], unsafeAddr src[0], csize_t(n))

proc scale*(x: var seq[float32], s: float32) =
  if x.len > 0: nimf_cpu_scale(addr x[0], csize_t(x.len), cfloat(s))

proc softmax*(x: openArray[float32], rows, cols: int): seq[float32] =
  result = newSeq[float32](rows * cols)
  if result.len > 0:
    nimf_cpu_softmax(unsafeAddr x[0], addr result[0], cint(rows), cint(cols))

proc layernorm*(x, gamma, beta: openArray[float32], rows, cols: int,
                eps: float32): seq[float32] =
  result = newSeq[float32](rows * cols)
  if result.len > 0:
    nimf_cpu_layernorm(unsafeAddr x[0], unsafeAddr gamma[0], unsafeAddr beta[0],
      addr result[0], cint(rows), cint(cols), cfloat(eps))

proc rmsnorm*(x, gamma: openArray[float32], rows, cols: int, eps: float32): seq[float32] =
  result = newSeq[float32](rows * cols)
  if result.len > 0:
    nimf_cpu_rmsnorm(unsafeAddr x[0], unsafeAddr gamma[0], addr result[0],
      cint(rows), cint(cols), cfloat(eps))

proc layernormBackward*(dy, x, gamma, beta: openArray[float32], rows, cols: int,
                        eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  result.dx = newSeq[float32](rows * cols)
  result.dgamma = newSeq[float32](cols)
  result.dbeta = newSeq[float32](cols)
  if rows * cols > 0:
    nimf_cpu_layernorm_backward(unsafeAddr dy[0], unsafeAddr x[0],
      unsafeAddr gamma[0], unsafeAddr beta[0], cint(rows), cint(cols),
      cfloat(eps), addr result.dx[0], addr result.dgamma[0], addr result.dbeta[0])

proc matmul*(a, b: openArray[float32], M, K, N: int): seq[float32] =
  result = newSeq[float32](M * N)
  if M > 0 and K > 0 and N > 0:
    nimf_cpu_matmul(unsafeAddr a[0], unsafeAddr b[0], addr result[0],
      cint(M), cint(K), cint(N))

proc matmulNT*(a, b: openArray[float32], M, K, N: int): seq[float32] =
  result = newSeq[float32](M * N)
  if M > 0 and K > 0 and N > 0:
    nimf_cpu_matmul_nt(unsafeAddr a[0], unsafeAddr b[0], addr result[0],
      cint(M), cint(K), cint(N))

proc matmulQ4*(a: openArray[float32], wq: openArray[uint8],
               scales, zeros: openArray[float32], M, K, N, groupSize,
               groupsPerRow: int): seq[float32] =
  result = newSeq[float32](M * N)
  if M > 0 and K > 0 and N > 0:
    nimf_cpu_matmul_q4(unsafeAddr a[0], unsafeAddr wq[0], unsafeAddr scales[0],
      unsafeAddr zeros[0], addr result[0], cint(M), cint(K), cint(N),
      cint(groupSize), cint(groupsPerRow))

proc attention*(q, k, v, mask: openArray[float32], maskLen, B, H, S, D: int,
                scale: float32): tuple[o, sMatrix: seq[float32]] =
  result.o = newSeq[float32](B * H * S * D)
  result.sMatrix = newSeq[float32](B * H * S * S)
  if result.o.len > 0 and result.sMatrix.len > 0:
    nimf_cpu_attention(unsafeAddr q[0], unsafeAddr k[0], unsafeAddr v[0],
      (if maskLen > 0: unsafeAddr mask[0] else: nil), cint(maskLen),
      addr result.o[0], addr result.sMatrix[0], cint(B), cint(H), cint(S),
      cint(D), cfloat(scale))

proc embeddingLookup*(table: openArray[float32], indices: openArray[int32],
                      vocab, dim: int): seq[float32] =
  result = newSeq[float32](indices.len * dim)
  if result.len > 0:
    nimf_cpu_embedding(unsafeAddr table[0], unsafeAddr indices[0], addr result[0],
      cint(indices.len), cint(vocab), cint(dim))

proc gather*(src: openArray[float32], idx: openArray[int32]): seq[float32] =
  result = newSeq[float32](idx.len)
  if idx.len > 0:
    nimf_cpu_gather(unsafeAddr src[0], unsafeAddr idx[0], addr result[0], cint(idx.len))

proc compare*(a, b: openArray[float32], op: int): seq[bool] =
  var mask = newSeq[uint8](a.len)
  result = newSeq[bool](a.len)
  if a.len > 0:
    nimf_cpu_compare(unsafeAddr a[0], unsafeAddr b[0], addr mask[0], cint(a.len), cint(op))
    for i in 0 ..< mask.len: result[i] = mask[i] != 0

proc dropoutMask*(n: int, p: float32, seed: uint32): seq[float32] =
  result = newSeq[float32](n)
  if n > 0: nimf_cpu_dropout_mask(addr result[0], cint(n), cfloat(p), seed)

proc argmaxIndex*(x: openArray[float32], outer, dimSize, inner: int):
    tuple[val: seq[float32], idx: seq[int32]] =
  result.val = newSeq[float32](outer * inner)
  result.idx = newSeq[int32](outer * inner)
  if result.val.len > 0:
    nimf_cpu_argmax(unsafeAddr x[0], addr result.val[0], addr result.idx[0],
      cint(outer), cint(dimSize), cint(inner))

proc permute*(src: openArray[float32], newStrides, oldStrides, dimsMap: openArray[int32],
              n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0:
    nimf_cpu_permute(unsafeAddr src[0], addr result[0], unsafeAddr newStrides[0],
      unsafeAddr oldStrides[0], unsafeAddr dimsMap[0], cint(newStrides.len), cint(n))

proc splitSlice*(src: openArray[float32], dimTotal, thisSize, offset, inner,
                 n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0:
    nimf_cpu_split_slice(unsafeAddr src[0], addr result[0], cint(dimTotal),
      cint(thisSize), cint(offset), cint(inner), cint(n))

proc concat*(flatSrc: openArray[float32], srcBase, srcDimSize, srcDimOffset: openArray[int32],
             totalDim, inner, n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0:
    nimf_cpu_concat(unsafeAddr flatSrc[0], addr result[0], unsafeAddr srcBase[0],
      unsafeAddr srcDimSize[0], unsafeAddr srcDimOffset[0], cint(srcBase.len),
      cint(totalDim), cint(inner), cint(n))

proc rope*(x: var seq[float32], cosT, sinT: openArray[float32], T, H, D: int) =
  if x.len > 0:
    nimf_cpu_rope(addr x[0], unsafeAddr cosT[0], unsafeAddr sinT[0],
      cint(T), cint(H), cint(D))

proc vmRun*(code: openArray[uint32], mem: var seq[float32],
            input: openArray[float32], nThreads, tgFloats, tgSize: int): bool =
  if code.len == 0 or mem.len == 0 or nThreads <= 0:
    cpuLastError = "invalid CPU VM arguments"
    return false
  var err = newString(512)
  let rc = nimf_cpu_vm_run(unsafeAddr code[0], cint(code.len), addr mem[0],
    cint(mem.len),
    (if input.len > 0: unsafeAddr input[0] else: nil), cint(input.len),
    cint(nThreads), cint(tgFloats), cint(tgSize), err.cstring, cint(err.len))
  if rc != 0:
    cpuLastError = $err
    return false
  cpuLastError.setLen(0)
  true
