# cuda_driver.nim - Runtime bindings to the CUDA Driver API (libcuda) loaded via dynlib.
# No build-time dependency: without CUDA/an NVIDIA GPU this module only reports
# "unavailable" instead of crashing -> a safe CPU fallback.
#
# CHANGES FROM THE OLD VERSION:
# - Context/module/stream/functions are created EXACTLY ONCE (lazily, on first use) and
#   live for the whole process, instead of cuCtxCreate + cuModuleLoadData per call.
# - One non-NULL CUstream, created once, reused for every kernel + copy.
# - H2D/D2H copies use the Async variants on that stream, with ONE cuStreamSynchronize
#   at the end of each proc (instead of the implicit sync of each cuMemcpyHtoD/DtoH).
# - Host staging buffers use pinned memory (cuMemAllocHost) instead of the normal heap,
#   so the CUDA driver does not copy into a hidden buffer before DMA to the GPU.
# - Device buffers and pinned host buffers come from size-keyed pools (bytes),
#   NO more cuMemAlloc/cuMemFree or cuMemAllocHost/cuMemFreeHost per call.
import std/dynlib
import std/os
import std/tables
type
  CUdeviceptr* = uint64
  CUstream* = pointer
  CUcontext* = pointer
  CUmodule* = pointer
  CUfunction* = pointer
  CUdevice* = int32
  CUresult* = int32

  CudaTensor* = object
    dptr*: CUdeviceptr
    bytes*: csize_t
    numel*: int
type
  CUdevice_attribute = int32

  CudaLib = object
    handle: LibHandle
    cuInit: proc(flags: uint32): CUresult {.cdecl.}
    cuDeviceGetCount: proc(count: ptr int32): CUresult {.cdecl.}
    cuDeviceGet: proc(dev: ptr CUdevice, ordinal: int32): CUresult {.cdecl.}
    cuDeviceGetAttribute: proc(pi: ptr int32, attrib: CUdevice_attribute, dev: CUdevice): CUresult {.cdecl.}
    cuCtxCreate: proc(pctx: ptr CUcontext, flags: uint32, dev: CUdevice): CUresult {.cdecl.}
    cuCtxDestroy: proc(ctx: CUcontext): CUresult {.cdecl.}
    cuModuleLoadData: proc(module: ptr CUmodule, image: cstring): CUresult {.cdecl.}
    cuModuleGetFunction: proc(hfunc: ptr CUfunction, hmod: CUmodule, name: cstring): CUresult {.cdecl.}
    cuMemAlloc: proc(dptr: ptr CUdeviceptr, bytesize: csize_t): CUresult {.cdecl.}
    cuMemFree: proc(dptr: CUdeviceptr): CUresult {.cdecl.}
    cuMemAllocHost: proc(pp: ptr pointer, bytesize: csize_t): CUresult {.cdecl.}
    cuMemFreeHost: proc(p: pointer): CUresult {.cdecl.}
    cuMemcpyHtoD: proc(dst: CUdeviceptr, src: pointer, byteCount: csize_t): CUresult {.cdecl.}
    cuMemcpyDtoH: proc(dst: pointer, src: CUdeviceptr, byteCount: csize_t): CUresult {.cdecl.}
    cuMemcpyHtoDAsync: proc(dst: CUdeviceptr, src: pointer, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}
    cuMemcpyDtoHAsync: proc(dst: pointer, src: CUdeviceptr, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}
    cuStreamCreate: proc(phStream: ptr CUstream, flags: uint32): CUresult {.cdecl.}
    cuStreamSynchronize: proc(stream: CUstream): CUresult {.cdecl.}
    cuLaunchKernel: proc(f: CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          sharedMemBytes: uint32, stream: CUstream, kernelParams: pointer,
                          extra: pointer): CUresult {.cdecl.}
    cuMemsetD8Async: proc(dst: CUdeviceptr, value: uint8, byteCount: csize_t, stream: CUstream): CUresult {.cdecl.}

const
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: CUdevice_attribute = 1

# --------------------------------------------------------------------------
# PTX kernels (equivalent to nvcc -ptx for 4 float32 elementwise ops), in
# the real file backends/cuda/kernels/vecop.ptx (NO escaped string in the
# .nim any more) -- 4 entry points: vecop_add/vecop_sub/vecop_mul/vecop_div,
# in one PTX module, loaded once with cuModuleLoadData at runtime,
# no nvcc needed on the target machine.
# --------------------------------------------------------------------------
const ptxSource = staticRead(currentSourcePath().parentDir() / "kernels" / "vecop.ptx")

proc fnNameFor(op: string): string =
  case op
  of "add": "vecop_add"
  of "sub": "vecop_sub"
  of "mul": "vecop_mul"
  of "div": "vecop_div"
  else: "vecop_add"

proc gpuCheck(cond: bool, msg: string) =
  ## Like doAssert but raises CatchableError instead of Defect, so gpubackend.nim
  ## can catch it and fall back to the CPU instead of crashing.
  if not cond:
    raise newException(CatchableError, msg)

var lib: CudaLib
var loaded = false

# ---- Global state, initialised once (ensureInit), alive until the process exits ----
var gInitialized = false
var gCtx: CUcontext
var gModule: CUmodule
var gStream: CUstream
var gFnCache = initTable[string, CUfunction]()

# ---- Device memory + pinned host memory pools, keyed by byte size ----
var gDevicePool = initTable[csize_t, seq[CUdeviceptr]]()
var gHostPool = initTable[csize_t, seq[pointer]]()

# ---- Block size tuned per GPU ----
var gOptimalBlockSize: uint32 = 256
var gBlockSizeInitialized = false

proc getOptimalBlockSize(): uint32 =
  if gBlockSizeInitialized: return gOptimalBlockSize
  var dev: CUdevice
  var maxThreads: int32
  if lib.cuDeviceGet(addr dev, 0) == 0:
    if lib.cuDeviceGetAttribute(addr maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev) == 0:
      if maxThreads >= 1024: gOptimalBlockSize = 512
      elif maxThreads >= 512: gOptimalBlockSize = 256
      else: gOptimalBlockSize = 128
  gBlockSizeInitialized = true
  result = gOptimalBlockSize

proc gridFor(n: int): uint32 =
  let blockSize = getOptimalBlockSize()
  uint32((n + int(blockSize) - 1) div int(blockSize))

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  # Common library names on Linux / Windows.
  for name in ["libcuda.so", "libcuda.so.1", "nvcuda.dll"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil:
    return false
  lib.cuInit = cast[typeof(lib.cuInit)](lib.handle.symAddr("cuInit"))
  lib.cuDeviceGetCount = cast[typeof(lib.cuDeviceGetCount)](lib.handle.symAddr("cuDeviceGetCount"))
  lib.cuDeviceGet = cast[typeof(lib.cuDeviceGet)](lib.handle.symAddr("cuDeviceGet"))
  lib.cuDeviceGetAttribute = cast[typeof(lib.cuDeviceGetAttribute)](lib.handle.symAddr("cuDeviceGetAttribute"))
  lib.cuCtxCreate = cast[typeof(lib.cuCtxCreate)](lib.handle.symAddr("cuCtxCreate_v2"))
  lib.cuCtxDestroy = cast[typeof(lib.cuCtxDestroy)](lib.handle.symAddr("cuCtxDestroy_v2"))
  lib.cuModuleLoadData = cast[typeof(lib.cuModuleLoadData)](lib.handle.symAddr("cuModuleLoadData"))
  lib.cuModuleGetFunction = cast[typeof(lib.cuModuleGetFunction)](lib.handle.symAddr("cuModuleGetFunction"))
  lib.cuMemAlloc = cast[typeof(lib.cuMemAlloc)](lib.handle.symAddr("cuMemAlloc_v2"))
  lib.cuMemFree = cast[typeof(lib.cuMemFree)](lib.handle.symAddr("cuMemFree_v2"))
  lib.cuMemAllocHost = cast[typeof(lib.cuMemAllocHost)](lib.handle.symAddr("cuMemAllocHost_v2"))
  lib.cuMemFreeHost = cast[typeof(lib.cuMemFreeHost)](lib.handle.symAddr("cuMemFreeHost"))
  lib.cuMemcpyHtoD = cast[typeof(lib.cuMemcpyHtoD)](lib.handle.symAddr("cuMemcpyHtoD_v2"))
  lib.cuMemcpyDtoH = cast[typeof(lib.cuMemcpyDtoH)](lib.handle.symAddr("cuMemcpyDtoH_v2"))
  lib.cuMemcpyHtoDAsync = cast[typeof(lib.cuMemcpyHtoDAsync)](lib.handle.symAddr("cuMemcpyHtoDAsync_v2"))
  lib.cuMemcpyDtoHAsync = cast[typeof(lib.cuMemcpyDtoHAsync)](lib.handle.symAddr("cuMemcpyDtoHAsync_v2"))
  lib.cuStreamCreate = cast[typeof(lib.cuStreamCreate)](lib.handle.symAddr("cuStreamCreate"))
  lib.cuStreamSynchronize = cast[typeof(lib.cuStreamSynchronize)](lib.handle.symAddr("cuStreamSynchronize"))
  lib.cuLaunchKernel = cast[typeof(lib.cuLaunchKernel)](lib.handle.symAddr("cuLaunchKernel"))
  lib.cuMemsetD8Async = cast[typeof(lib.cuMemsetD8Async)](lib.handle.symAddr("cuMemsetD8Async"))
  result = lib.cuInit != nil and lib.cuLaunchKernel != nil

proc cudaAvailable*(): bool =
  ## Checks whether the machine has an NVIDIA driver (libcuda) and at least one GPU.
  if not tryLoad(): return false
  if lib.cuInit(0) != 0: return false
  var count: int32 = 0
  if lib.cuDeviceGetCount(addr count) != 0: return false
  result = count > 0

proc ensureInit() =
  ## Creates the context + PTX module + stream EXACTLY ONCE for the process lifetime.
  ## Later calls return immediately - no repeated cuCtxCreate/cuModuleLoadData
  ## per operation as in the old version.
  if gInitialized: return
  gpuCheck(tryLoad(), "libcuda not found")
  var dev: CUdevice
  let initRes = lib.cuInit(0)
  gpuCheck(initRes == 0, "cuInit failed (error code: " & $initRes & ")")
  
  let devRes = lib.cuDeviceGet(addr dev, 0)
  gpuCheck(devRes == 0, "cuDeviceGet failed (error code: " & $devRes & ")")
  
  let ctxRes = lib.cuCtxCreate(addr gCtx, 0, dev)
  gpuCheck(ctxRes == 0, "cuCtxCreate failed (error code: " & $ctxRes & ")")
  
  let modRes = lib.cuModuleLoadData(addr gModule, ptxSource.cstring)
  gpuCheck(modRes == 0, "cuModuleLoadData failed (PTX) (error code: " & $modRes & "). This usually means the PTX version or format is incompatible with the installed CUDA driver.")
  
  let streamRes = lib.cuStreamCreate(addr gStream, 0)
  gpuCheck(streamRes == 0, "cuStreamCreate failed (error code: " & $streamRes & ")")
  gInitialized = true

proc getStream*(): CUstream =
  ensureInit()
  result = gStream

proc getFn(name: string): CUfunction =
  ## CUfunction cache by name - avoids repeated cuModuleGetFunction.
  if gFnCache.hasKey(name):
    return gFnCache[name]
  var fn: CUfunction
  gpuCheck(lib.cuModuleGetFunction(addr fn, gModule, name.cstring) == 0,
           "cuModuleGetFunction failed for " & name)
  gFnCache[name] = fn
  result = fn

proc getDeviceBuf*(bytes: csize_t): CUdeviceptr =
  ## Takes a device buffer of the same size from the pool if available, else cuMemAlloc.
  ## Exported (*) so cuda_runtime.nim shares one pool for resident CudaTensors
  ## (resident matmul also allocates here instead of a separate cudaMalloc pool,
  ## to avoid two separate pools for the same kind of buffer).
  if gDevicePool.hasKey(bytes) and gDevicePool[bytes].len > 0:
    return gDevicePool[bytes].pop()
  var dptr: CUdeviceptr
  gpuCheck(lib.cuMemAlloc(addr dptr, bytes) == 0, "cuMemAlloc failed")
  result = dptr

proc putDeviceBuf*(bytes: csize_t, dptr: CUdeviceptr) =
  ## Returns the buffer to the pool for reuse instead of cuMemFree right away.
  gDevicePool.mgetOrPut(bytes, @[]).add(dptr)

proc getHostBuf(bytes: csize_t): pointer =
  ## Takes a pinned host staging buffer from the pool, else cuMemAllocHost.
  if gHostPool.hasKey(bytes) and gHostPool[bytes].len > 0:
    return gHostPool[bytes].pop()
  var p: pointer
  gpuCheck(lib.cuMemAllocHost(addr p, bytes) == 0, "cuMemAllocHost failed")
  result = p

proc putHostBuf(bytes: csize_t, p: pointer) =
  gHostPool.mgetOrPut(bytes, @[]).add(p)
proc uploadAsync*(data: seq[float32]): CudaTensor =
  ensureInit()
  let bytes = csize_t(data.len * sizeof(float32))
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0, "uploadAsync H2D failed")
  return CudaTensor(dptr: dptr, bytes: bytes, numel: data.len)

proc uploadIndicesAsync*(data: seq[int32]): CudaTensor =
  ensureInit()
  let bytes = csize_t(data.len * sizeof(int32))
  let dptr = getDeviceBuf(bytes)
  gpuCheck(lib.cuMemcpyHtoDAsync(dptr, unsafeAddr data[0], bytes, gStream) == 0, "uploadIndicesAsync H2D failed")
  return CudaTensor(dptr: dptr, bytes: bytes, numel: data.len)

proc downloadSync*(t: CudaTensor): seq[float32] =
  ensureInit()
  result = newSeq[float32](t.numel)
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], t.dptr, t.bytes, gStream) == 0, "downloadSync D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "downloadSync sync failed")

proc freeResident*(t: var CudaTensor) =
  if t.dptr != 0:
    putDeviceBuf(t.bytes, t.dptr)
    t.dptr = 0
    t.bytes = 0
    t.numel = 0
# ============================================================
# VECTOR OPERATIONS
# ============================================================
proc cudaVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  ## Runs an elementwise op (add/sub/mul/div) on an NVIDIA GPU through the CUDA Driver API.
  ## Raises on any error so the caller (gpubackend) falls back to the CPU.
  ensureInit()
  let n = a.len
  result = newSeq[float32](n)
  let fn = getFn(fnNameFor(op))
  let bytes = csize_t(n * sizeof(float32))

  let dA = getDeviceBuf(bytes)
  let dB = getDeviceBuf(bytes)
  let dC = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dA); putDeviceBuf(bytes, dB); putDeviceBuf(bytes, dC)

  gpuCheck(lib.cuMemcpyHtoDAsync(dA, unsafeAddr a[0], bytes, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dB, unsafeAddr b[0], bytes, gStream) == 0, "H2D failed")

  var dAv = dA; var dBv = dB; var dCv = dC
  var nParam = int32(n)
  var params: array[4, pointer] = [cast[pointer](addr dAv), cast[pointer](addr dBv),
                                    cast[pointer](addr dCv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dC, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# ACTIVATIONS
# ============================================================
proc cudaActivation*(op: string, x: seq[float32]): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let fn = getFn("vecop_" & op)
  let bytes = csize_t(n * sizeof(float32))

  let dX = getDeviceBuf(bytes)
  let dY = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dX); putDeviceBuf(bytes, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")

  var dXv = dX; var dYv = dY
  var nParam = int32(n)
  var params: array[3, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# APFLU FORWARD
# ============================================================
proc cudaApflu*(x: seq[float32], alpha: float32, beta: float32): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let fn = getFn("vecop_apflu")
  let bytes = csize_t(n * sizeof(float32))

  let dX = getDeviceBuf(bytes)
  let dY = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dX); putDeviceBuf(bytes, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")

  var dXv = dX; var dYv = dY
  var nParam = int32(n)
  var aParam = alpha
  var bParam = beta
  var params: array[5, pointer] = [
    cast[pointer](addr dXv), cast[pointer](addr dYv),
    cast[pointer](addr nParam), cast[pointer](addr aParam),
    cast[pointer](addr bParam)
  ]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# APFLU BACKWARD
# ============================================================
proc cudaApfluBackward*(x, dy: seq[float32], alpha: float32, beta: float32): seq[float32] =
  ensureInit()
  let n = x.len
  result = newSeq[float32](n)
  let fn = getFn("vecop_apflu_backward")
  let bytes = csize_t(n * sizeof(float32))

  let dXin = getDeviceBuf(bytes)      # SỬA: dXin
  let dDy = getDeviceBuf(bytes)
  let dDxOut = getDeviceBuf(bytes)    # SỬA: dDxOut
  defer:
    putDeviceBuf(bytes, dXin); putDeviceBuf(bytes, dDy); putDeviceBuf(bytes, dDxOut)

  gpuCheck(lib.cuMemcpyHtoDAsync(dXin, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dDy, unsafeAddr dy[0], bytes, gStream) == 0, "H2D failed")

  var dXinV = dXin; var dDyV = dDy; var dDxOutV = dDxOut
  var nParam = int32(n)
  var aParam = alpha
  var bParam = beta
  var params: array[6, pointer] = [
    cast[pointer](addr dXinV), cast[pointer](addr dDyV), cast[pointer](addr dDxOutV),
    cast[pointer](addr nParam), cast[pointer](addr aParam),
    cast[pointer](addr bParam)
  ]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dDxOut, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")
# ============================================================
# SOFTMAX
# ============================================================
proc cudaSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  ensureInit()
  let n = rows * cols
  result = newSeq[float32](n)
  let fn = getFn("softmax_kernel")
  let bytes = csize_t(n * sizeof(float32))

  let dX = getDeviceBuf(bytes)
  let dY = getDeviceBuf(bytes)
  defer:
    putDeviceBuf(bytes, dX); putDeviceBuf(bytes, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytes, gStream) == 0, "H2D failed")

  var dXv = dX; var dYv = dY
  var rParam = int32(rows)
  var cParam = int32(cols)
  var params: array[4, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv),
                                    cast[pointer](addr rParam), cast[pointer](addr cParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytes, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# LAYERNORM FORWARD
# ============================================================
proc cudaLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  ensureInit()
  let n = rows * cols
  result = newSeq[float32](n)
  let fn = getFn("layernorm_kernel")

  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let dX = getDeviceBuf(bytesX)
  let dGamma = getDeviceBuf(bytesC)
  let dBeta = getDeviceBuf(bytesC)
  let dY = getDeviceBuf(bytesX)
  defer:
    putDeviceBuf(bytesX, dX); putDeviceBuf(bytesC, dGamma); putDeviceBuf(bytesC, dBeta); putDeviceBuf(bytesX, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dX, unsafeAddr x[0], bytesX, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dGamma, unsafeAddr gamma[0], bytesC, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dBeta, unsafeAddr beta[0], bytesC, gStream) == 0, "H2D failed")

  var dXv = dX; var dGv = dGamma; var dBv = dBeta; var dYv = dY
  var rParam = int32(rows)
  var cParam = int32(cols)
  var eParam = float32(eps)
  var params: array[7, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dGv), cast[pointer](addr dBv),
                                    cast[pointer](addr dYv), cast[pointer](addr rParam), cast[pointer](addr cParam),
                                    cast[pointer](addr eParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytesX, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

proc cudaLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  ensureInit()
  let n = rows * cols
  var dx = newSeq[float32](n)
  var dgamma = newSeq[float32](cols)
  var dbeta = newSeq[float32](cols)

  let fn = getFn("layernorm_backward_kernel")

  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))

  let dDyBuf = getDeviceBuf(bytesX)      # SỬA: dDyBuf
  let dXBuf = getDeviceBuf(bytesX)       # SỬA: dXBuf
  let dGammaBuf = getDeviceBuf(bytesC)   # SỬA: dGammaBuf
  let dBetaBuf = getDeviceBuf(bytesC)    # SỬA: dBetaBuf
  let dDxBuf = getDeviceBuf(bytesX)      # SỬA: dDxBuf
  let dDgammaBuf = getDeviceBuf(bytesC)  # SỬA: dDgammaBuf
  let dDbetaBuf = getDeviceBuf(bytesC)   # SỬA: dDbetaBuf
  defer:
    putDeviceBuf(bytesX, dDyBuf); putDeviceBuf(bytesX, dXBuf)
    putDeviceBuf(bytesC, dGammaBuf); putDeviceBuf(bytesC, dBetaBuf)
    putDeviceBuf(bytesX, dDxBuf); putDeviceBuf(bytesC, dDgammaBuf)
    putDeviceBuf(bytesC, dDbetaBuf)

  gpuCheck(lib.cuMemcpyHtoDAsync(dDyBuf, unsafeAddr dy[0], bytesX, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dXBuf, unsafeAddr x[0], bytesX, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dGammaBuf, unsafeAddr gamma[0], bytesC, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dBetaBuf, unsafeAddr beta[0], bytesC, gStream) == 0, "H2D failed")

  var dDyV = dDyBuf; var dXV = dXBuf; var dGammaV = dGammaBuf; var dBetaV = dBetaBuf
  var dDxV = dDxBuf; var dDgammaV = dDgammaBuf; var dDbetaV = dDbetaBuf
  var rParam = int32(rows)
  var cParam = int32(cols)
  var eParam = float32(eps)
  var params: array[10, pointer] = [
    cast[pointer](addr dDyV), cast[pointer](addr dXV),
    cast[pointer](addr dGammaV), cast[pointer](addr dBetaV),
    cast[pointer](addr dDxV), cast[pointer](addr dDgammaV),
    cast[pointer](addr dDbetaV),
    cast[pointer](addr rParam), cast[pointer](addr cParam),
    cast[pointer](addr eParam)
  ]
  # FIX: layernorm_backward_kernel is a "1 thread = 1 independent id" kernel
  # (like OpenCL get_global_id, threads in a block do not cooperate),
  # unlike layernorm_kernel (forward), which is "1 block = 1 row, cooperating threads".
  # It used getOptimalBlockSize() (128/256/512 threads/block), so each block
  # computed only ids in [0, blockSize) and they OVERWROTE each other -> wrong results /
  # races between blocks -> a silently broken CUDA context -> D2H failed on the next call.
  gpuCheck(lib.cuLaunchKernel(fn, uint32(max(rows, cols)), 1, 1, 1, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr dx[0], dDxBuf, bytesX, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr dgamma[0], dDgammaBuf, bytesC, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr dbeta[0], dDbetaBuf, bytesC, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")
  result = (dx, dgamma, dbeta)
# ============================================================
# EMBEDDING LOOKUP
# ============================================================
proc cudaEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  ensureInit()
  let num = indices.len
  result = newSeq[float32](num * dim)
  let fn = getFn("embedding_lookup_kernel")

  let bytesTable = csize_t(vocab * dim * sizeof(float32))
  let bytesIndices = csize_t(num * sizeof(int32))
  let bytesY = csize_t(num * dim * sizeof(float32))
  let dTable = getDeviceBuf(bytesTable)
  let dIndices = getDeviceBuf(bytesIndices)
  let dY = getDeviceBuf(bytesY)
  defer:
    putDeviceBuf(bytesTable, dTable); putDeviceBuf(bytesIndices, dIndices); putDeviceBuf(bytesY, dY)

  gpuCheck(lib.cuMemcpyHtoDAsync(dTable, unsafeAddr table[0], bytesTable, gStream) == 0, "H2D failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dIndices, unsafeAddr indices[0], bytesIndices, gStream) == 0, "H2D failed")

  var dTv = dTable; var dIdxv = dIndices; var dYv = dY
  var vParam = int32(vocab)
  var dParam = int32(dim)
  var numParam = int32(num)
  var params: array[6, pointer] = [cast[pointer](addr dTv), cast[pointer](addr dIdxv), cast[pointer](addr dYv),
                                    cast[pointer](addr vParam), cast[pointer](addr dParam), cast[pointer](addr numParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(num), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result[0], dY, bytesY, gStream) == 0, "D2H failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "cuStreamSynchronize failed")

# ============================================================
# ATTENTION FUSED FORWARD
# ============================================================
proc cudaAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  
  result.o = newSeq[float32](qkvLen)
  result.s_matrix = newSeq[float32](sLen)
  
  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))
  
  let dQ = getDeviceBuf(bytesQKV)
  let dK = getDeviceBuf(bytesQKV)
  let dV = getDeviceBuf(bytesQKV)
  let dOut = getDeviceBuf(bytesQKV)
  let dS = getDeviceBuf(bytesS)
  defer:
    putDeviceBuf(bytesQKV, dQ)
    putDeviceBuf(bytesQKV, dK)
    putDeviceBuf(bytesQKV, dV)
    putDeviceBuf(bytesQKV, dOut)
    putDeviceBuf(bytesS, dS)
  
  gpuCheck(lib.cuMemcpyHtoDAsync(dQ, unsafeAddr q[0], bytesQKV, gStream) == 0, "H2D Q failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dK, unsafeAddr k[0], bytesQKV, gStream) == 0, "H2D K failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dV, unsafeAddr v[0], bytesQKV, gStream) == 0, "H2D V failed")
  
  let fn = getFn("attention_fused_kernel")
  
  var bQ = dQ; var bK = dK; var bV = dV; var bOut = dOut; var bS = dS
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = float32(scale)
  
  var params: array[10, pointer] = [
    cast[pointer](addr bQ), cast[pointer](addr bK), cast[pointer](addr bV),
    cast[pointer](addr bOut), cast[pointer](addr bS),
    cast[pointer](addr bArg), cast[pointer](addr hArg),
    cast[pointer](addr sArg), cast[pointer](addr dArg),
    cast[pointer](addr scArg)
  ]
  
  let blockSize = getOptimalBlockSize()
  gpuCheck(lib.cuLaunchKernel(fn, uint32(B*H), uint32(S), 1, blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "attention_fused_kernel launch failed")
  
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.o[0], dOut, bytesQKV, gStream) == 0, "D2H O failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.s_matrix[0], dS, bytesS, gStream) == 0, "D2H S failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "stream sync failed")

# ============================================================
# ATTENTION FUSED BACKWARD
# ============================================================
proc cudaAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  ensureInit()
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  
  result.dq = newSeq[float32](qkvLen)
  result.dk = newSeq[float32](qkvLen)
  result.dv = newSeq[float32](qkvLen)
  
  let bytesQKV = csize_t(qkvLen * sizeof(float32))
  let bytesS = csize_t(sLen * sizeof(float32))
  
  let dQ = getDeviceBuf(bytesQKV)
  let dK = getDeviceBuf(bytesQKV)
  let dV = getDeviceBuf(bytesQKV)
  let dS = getDeviceBuf(bytesS)
  let dDy = getDeviceBuf(bytesQKV)
  let dDq = getDeviceBuf(bytesQKV)
  let dDk = getDeviceBuf(bytesQKV)
  let dDv = getDeviceBuf(bytesQKV)
  defer:
    putDeviceBuf(bytesQKV, dQ); putDeviceBuf(bytesQKV, dK)
    putDeviceBuf(bytesQKV, dV); putDeviceBuf(bytesS, dS)
    putDeviceBuf(bytesQKV, dDy); putDeviceBuf(bytesQKV, dDq)
    putDeviceBuf(bytesQKV, dDk); putDeviceBuf(bytesQKV, dDv)
  
  gpuCheck(lib.cuMemcpyHtoDAsync(dQ, unsafeAddr q[0], bytesQKV, gStream) == 0, "H2D Q failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dK, unsafeAddr k[0], bytesQKV, gStream) == 0, "H2D K failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dV, unsafeAddr v[0], bytesQKV, gStream) == 0, "H2D V failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dS, unsafeAddr s_matrix[0], bytesS, gStream) == 0, "H2D S failed")
  gpuCheck(lib.cuMemcpyHtoDAsync(dDy, unsafeAddr dy[0], bytesQKV, gStream) == 0, "H2D dy failed")
  
  gpuCheck(lib.cuMemsetD8Async(dDq, 0, bytesQKV, gStream) == 0, "zero init dq failed")
  gpuCheck(lib.cuMemsetD8Async(dDk, 0, bytesQKV, gStream) == 0, "zero init dk failed")
  gpuCheck(lib.cuMemsetD8Async(dDv, 0, bytesQKV, gStream) == 0, "zero init dv failed")
  
  let fn = getFn("attention_fused_backward_kernel")
  
  var bQ = dQ; var bK = dK; var bV = dV; var bS = dS; var bDy = dDy
  var bDq = dDq; var bDk = dDk; var bDv = dDv
  var bArg = int32(B); var hArg = int32(H); var sArg = int32(S); var dArg = int32(D); var scArg = float32(scale)
  
  var params: array[13, pointer] = [
    cast[pointer](addr bQ), cast[pointer](addr bK), cast[pointer](addr bV),
    cast[pointer](addr bS), cast[pointer](addr bDy),
    cast[pointer](addr bDq), cast[pointer](addr bDk), cast[pointer](addr bDv),
    cast[pointer](addr bArg), cast[pointer](addr hArg),
    cast[pointer](addr sArg), cast[pointer](addr dArg),
    cast[pointer](addr scArg)
  ]
  
  let blockSize = getOptimalBlockSize()
  gpuCheck(lib.cuLaunchKernel(fn, uint32(B*H), uint32(S), 1, blockSize, 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "attention_fused_backward_kernel launch failed")
  
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.dq[0], dDq, bytesQKV, gStream) == 0, "D2H dq failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.dk[0], dDk, bytesQKV, gStream) == 0, "D2H dk failed")
  gpuCheck(lib.cuMemcpyDtoHAsync(addr result.dv[0], dDv, bytesQKV, gStream) == 0, "D2H dv failed")
  gpuCheck(lib.cuStreamSynchronize(gStream) == 0, "stream sync failed")
# ─────────────────────────────────────────────────────────────
# CUDA Resident Vector Ops
# ─────────────────────────────────────────────────────────────

proc cudaVecOpR*(op: string, a, b: CudaTensor): CudaTensor =
  ensureInit()
  gpuCheck(a.numel == b.numel, "cudaVecOpR: size mismatch")
  let n = a.numel
  let bytes = a.bytes
  let dC = getDeviceBuf(bytes)
  
  let fn = getFn(fnNameFor(op))
  var dAv = a.dptr; var dBv = b.dptr; var dCv = dC
  var nParam = int32(n)
  var params: array[4, pointer] = [cast[pointer](addr dAv), cast[pointer](addr dBv),
                                    cast[pointer](addr dCv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaVecOpR launch failed")
  return CudaTensor(dptr: dC, bytes: bytes, numel: n)

proc cudaActivationR*(op: string, x: CudaTensor): CudaTensor =
  ensureInit()
  let n = x.numel
  let bytes = x.bytes
  let dY = getDeviceBuf(bytes)
  
  let fn = getFn("vecop_" & op)
  var dXv = x.dptr; var dYv = dY
  var nParam = int32(n)
  var params: array[3, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv), cast[pointer](addr nParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(n), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaActivationR launch failed")
  return CudaTensor(dptr: dY, bytes: bytes, numel: n)

proc cudaSoftmaxR*(x: CudaTensor, rows, cols: int): CudaTensor =
  ensureInit()
  let n = rows * cols
  let bytes = csize_t(n * sizeof(float32))
  let dY = getDeviceBuf(bytes)
  
  let fn = getFn("softmax_kernel")
  var dXv = x.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols)
  var params: array[4, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dYv),
                                    cast[pointer](addr rParam), cast[pointer](addr cParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaSoftmaxR launch failed")
  return CudaTensor(dptr: dY, bytes: bytes, numel: n)

proc cudaLayernormR*(x, gamma, beta: CudaTensor, rows, cols: int, eps: float32): CudaTensor =
  ensureInit()
  let n = rows * cols
  let bytesX = csize_t(n * sizeof(float32))
  let bytesC = csize_t(cols * sizeof(float32))
  let dY = getDeviceBuf(bytesX)
  
  let fn = getFn("layernorm_kernel")
  var dXv = x.dptr; var dGv = gamma.dptr; var dBv = beta.dptr; var dYv = dY
  var rParam = int32(rows); var cParam = int32(cols); var eParam = float32(eps)
  var params: array[7, pointer] = [cast[pointer](addr dXv), cast[pointer](addr dGv), cast[pointer](addr dBv),
                                    cast[pointer](addr dYv), cast[pointer](addr rParam), cast[pointer](addr cParam),
                                    cast[pointer](addr eParam)]
  gpuCheck(lib.cuLaunchKernel(fn, uint32(rows), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaLayernormR launch failed")
  return CudaTensor(dptr: dY, bytes: bytesX, numel: n)

proc cudaEmbeddingLookupR*(table, indices: CudaTensor, numIdx, vocab, dim: int): CudaTensor =
  ensureInit()
  let bytesY = csize_t(numIdx * dim * sizeof(float32))
  let dY = getDeviceBuf(bytesY)
  
  let fn = getFn("embedding_lookup_kernel")
  var dTv = table.dptr; var dIdxv = indices.dptr; var dYv = dY
  var vParam = int32(vocab); var dParam = int32(dim); var numParam = int32(numIdx)
  var params: array[6, pointer] = [cast[pointer](addr dTv), cast[pointer](addr dIdxv), cast[pointer](addr dYv),
                                    cast[pointer](addr vParam), cast[pointer](addr dParam), cast[pointer](addr numParam)]
  gpuCheck(lib.cuLaunchKernel(fn, gridFor(numIdx), 1, 1, getOptimalBlockSize(), 1, 1, 0, gStream,
                                addr params[0], nil) == 0, "cudaEmbeddingLookupR launch failed")
  return CudaTensor(dptr: dY, bytes: bytesY, numel: numIdx * dim)