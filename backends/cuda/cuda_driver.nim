# cuda_driver.nim - Runtime bindings to the CUDA Driver API (libcuda) loaded via dynlib.
# Không phụ thuộc lúc build: nếu máy không có CUDA/GPU NVIDIA, module này chỉ trả về
# "không khả dụng" thay vì làm sập chương trình -> cho phép fallback CPU an toàn.
import std/dynlib
import std/os

type
  CUresult = int32
  CUdevice = int32
  CUcontext = pointer
  CUmodule = pointer
  CUfunction = pointer
  CUdeviceptr = uint64

  CudaLib = object
    handle: LibHandle
    cuInit: proc(flags: uint32): CUresult {.cdecl.}
    cuDeviceGetCount: proc(count: ptr int32): CUresult {.cdecl.}
    cuDeviceGet: proc(dev: ptr CUdevice, ordinal: int32): CUresult {.cdecl.}
    cuCtxCreate: proc(pctx: ptr CUcontext, flags: uint32, dev: CUdevice): CUresult {.cdecl.}
    cuCtxDestroy: proc(ctx: CUcontext): CUresult {.cdecl.}
    cuModuleLoadData: proc(module: ptr CUmodule, image: cstring): CUresult {.cdecl.}
    cuModuleGetFunction: proc(hfunc: ptr CUfunction, hmod: CUmodule, name: cstring): CUresult {.cdecl.}
    cuMemAlloc: proc(dptr: ptr CUdeviceptr, bytesize: csize_t): CUresult {.cdecl.}
    cuMemFree: proc(dptr: CUdeviceptr): CUresult {.cdecl.}
    cuMemcpyHtoD: proc(dst: CUdeviceptr, src: pointer, byteCount: csize_t): CUresult {.cdecl.}
    cuMemcpyDtoH: proc(dst: pointer, src: CUdeviceptr, byteCount: csize_t): CUresult {.cdecl.}
    cuLaunchKernel: proc(f: CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ,
                          sharedMemBytes: uint32, stream: pointer, kernelParams: pointer,
                          extra: pointer): CUresult {.cdecl.}

# --------------------------------------------------------------------------
# PTX kernels (tương đương nvcc -ptx cho 4 phép elementwise float32), nằm ở
# file thật backends/cuda/kernels/vecop.ptx (KHÔNG còn escape-string trong
# .nim nữa) -- 4 entry point riêng: vecop_add/vecop_sub/vecop_mul/vecop_div,
# gộp trong cùng 1 module PTX, nạp 1 lần bằng cuModuleLoadData tại runtime,
# không cần nvcc trên máy chạy.
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
  ## Giống doAssert nhưng raise CatchableError thay vì Defect, để gpubackend.nim
  ## có thể bắt lỗi và fallback CPU thay vì làm crash chương trình.
  if not cond:
    raise newException(CatchableError, msg)

var lib: CudaLib
var loaded = false

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  # Tên thư viện phổ biến trên Linux / Windows.
  for name in ["libcuda.so", "libcuda.so.1", "nvcuda.dll"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil:
    return false
  lib.cuInit = cast[typeof(lib.cuInit)](lib.handle.symAddr("cuInit"))
  lib.cuDeviceGetCount = cast[typeof(lib.cuDeviceGetCount)](lib.handle.symAddr("cuDeviceGetCount"))
  lib.cuDeviceGet = cast[typeof(lib.cuDeviceGet)](lib.handle.symAddr("cuDeviceGet"))
  lib.cuCtxCreate = cast[typeof(lib.cuCtxCreate)](lib.handle.symAddr("cuCtxCreate_v2"))
  lib.cuCtxDestroy = cast[typeof(lib.cuCtxDestroy)](lib.handle.symAddr("cuCtxDestroy_v2"))
  lib.cuModuleLoadData = cast[typeof(lib.cuModuleLoadData)](lib.handle.symAddr("cuModuleLoadData"))
  lib.cuModuleGetFunction = cast[typeof(lib.cuModuleGetFunction)](lib.handle.symAddr("cuModuleGetFunction"))
  lib.cuMemAlloc = cast[typeof(lib.cuMemAlloc)](lib.handle.symAddr("cuMemAlloc_v2"))
  lib.cuMemFree = cast[typeof(lib.cuMemFree)](lib.handle.symAddr("cuMemFree_v2"))
  lib.cuMemcpyHtoD = cast[typeof(lib.cuMemcpyHtoD)](lib.handle.symAddr("cuMemcpyHtoD_v2"))
  lib.cuMemcpyDtoH = cast[typeof(lib.cuMemcpyDtoH)](lib.handle.symAddr("cuMemcpyDtoH_v2"))
  lib.cuLaunchKernel = cast[typeof(lib.cuLaunchKernel)](lib.handle.symAddr("cuLaunchKernel"))
  result = lib.cuInit != nil and lib.cuLaunchKernel != nil

proc cudaAvailable*(): bool =
  ## Dò xem máy có driver NVIDIA (libcuda) và ít nhất 1 GPU hay không.
  if not tryLoad(): return false
  if lib.cuInit(0) != 0: return false
  var count: int32 = 0
  if lib.cuDeviceGetCount(addr count) != 0: return false
  result = count > 0

proc cudaVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  ## Chạy phép toán elementwise (add/sub/mul/div) trên GPU NVIDIA qua CUDA Driver API.
  ## Raise nếu có lỗi bất kỳ, để tầng gọi (gpubackend) fallback về CPU.
  if not tryLoad():
    raise newException(CatchableError, "libcuda not found")
  let n = a.len
  result = newSeq[float32](n)
  var dev: CUdevice
  var ctx: CUcontext
  gpuCheck(lib.cuInit(0) == 0, "cuInit failed")
  gpuCheck(lib.cuDeviceGet(addr dev, 0) == 0, "cuDeviceGet failed")
  gpuCheck(lib.cuCtxCreate(addr ctx, 0, dev) == 0, "cuCtxCreate failed")
  defer: discard lib.cuCtxDestroy(ctx)

  var module: CUmodule
  gpuCheck(lib.cuModuleLoadData(addr module, ptxSource.cstring) == 0, "cuModuleLoadData failed (PTX)")
  var fn: CUfunction
  gpuCheck(lib.cuModuleGetFunction(addr fn, module, fnNameFor(op).cstring) == 0, "cuModuleGetFunction failed")

  let bytes = csize_t(n * sizeof(float32))
  var dA, dB, dC: CUdeviceptr
  gpuCheck(lib.cuMemAlloc(addr dA, bytes) == 0, "GPU operation failed")
  gpuCheck(lib.cuMemAlloc(addr dB, bytes) == 0, "GPU operation failed")
  gpuCheck(lib.cuMemAlloc(addr dC, bytes) == 0, "GPU operation failed")
  defer:
    discard lib.cuMemFree(dA)
    discard lib.cuMemFree(dB)
    discard lib.cuMemFree(dC)

  var aVar = a
  var bVar = b
  gpuCheck(lib.cuMemcpyHtoD(dA, addr aVar[0], bytes) == 0, "GPU operation failed")
  gpuCheck(lib.cuMemcpyHtoD(dB, addr bVar[0], bytes) == 0, "GPU operation failed")

  var nParam = int32(n)
  var params: array[4, pointer] = [cast[pointer](addr dA), cast[pointer](addr dB),
                                    cast[pointer](addr dC), cast[pointer](addr nParam)]
  let threads: uint32 = 256
  let blocks: uint32 = uint32((n + 255) div 256)
  gpuCheck(lib.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nil,
                               addr params[0], nil) == 0, "cuLaunchKernel failed")
  gpuCheck(lib.cuMemcpyDtoH(addr result[0], dC, bytes) == 0, "GPU operation failed")
