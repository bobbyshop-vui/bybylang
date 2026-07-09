# opencl_api.nim - Runtime bindings to OpenCL (libOpenCL / OpenCL.framework) qua dynlib.
# Biên dịch kernel OpenCL C từ chuỗi nguồn ngay lúc chạy (clBuildProgram), không cần
# công cụ build ngoài nào cả -> chạy được trên bất kỳ GPU/CPU nào có driver OpenCL.
import std/dynlib
import std/os

type
  cl_int = int32
  cl_uint = uint32
  cl_platform_id = pointer
  cl_device_id = pointer
  cl_context = pointer
  cl_command_queue = pointer
  cl_program = pointer
  cl_kernel = pointer
  cl_mem = pointer

  OclLib = object
    handle: LibHandle
    clGetPlatformIDs: proc(num_entries: cl_uint, platforms: ptr cl_platform_id, num_platforms: ptr cl_uint): cl_int {.cdecl.}
    clGetDeviceIDs: proc(platform: cl_platform_id, device_type: uint64, num_entries: cl_uint,
                          devices: ptr cl_device_id, num_devices: ptr cl_uint): cl_int {.cdecl.}
    clCreateContext: proc(properties: pointer, num_devices: cl_uint, devices: ptr cl_device_id,
                           pfn_notify: pointer, user_data: pointer, errcode_ret: ptr cl_int): cl_context {.cdecl.}
    clCreateCommandQueue: proc(context: cl_context, device: cl_device_id, properties: uint64,
                                errcode_ret: ptr cl_int): cl_command_queue {.cdecl.}
    clCreateBuffer: proc(context: cl_context, flags: uint64, size: csize_t, host_ptr: pointer,
                          errcode_ret: ptr cl_int): cl_mem {.cdecl.}
    clCreateProgramWithSource: proc(context: cl_context, count: cl_uint, strings: ptr cstring,
                                     lengths: ptr csize_t, errcode_ret: ptr cl_int): cl_program {.cdecl.}
    clBuildProgram: proc(program: cl_program, num_devices: cl_uint, device_list: ptr cl_device_id,
                          options: cstring, pfn_notify: pointer, user_data: pointer): cl_int {.cdecl.}
    clCreateKernel: proc(program: cl_program, kernel_name: cstring, errcode_ret: ptr cl_int): cl_kernel {.cdecl.}
    clSetKernelArg: proc(kernel: cl_kernel, arg_index: cl_uint, arg_size: csize_t, arg_value: pointer): cl_int {.cdecl.}
    clEnqueueNDRangeKernel: proc(command_queue: cl_command_queue, kernel: cl_kernel, work_dim: cl_uint,
                                  global_work_offset: ptr csize_t, global_work_size: ptr csize_t,
                                  local_work_size: ptr csize_t, num_events_in_wait_list: cl_uint,
                                  event_wait_list: pointer, event: pointer): cl_int {.cdecl.}
    clEnqueueWriteBuffer: proc(command_queue: cl_command_queue, buffer: cl_mem, blocking_write: uint32,
                                offset, size: csize_t, ptrData: pointer, num_events: cl_uint,
                                wait_list: pointer, event: pointer): cl_int {.cdecl.}
    clEnqueueReadBuffer: proc(command_queue: cl_command_queue, buffer: cl_mem, blocking_read: uint32,
                               offset, size: csize_t, ptrData: pointer, num_events: cl_uint,
                               wait_list: pointer, event: pointer): cl_int {.cdecl.}
    clFinish: proc(command_queue: cl_command_queue): cl_int {.cdecl.}
    clReleaseMemObject: proc(memobj: cl_mem): cl_int {.cdecl.}

const kernelSource = staticRead(currentSourcePath().parentDir() / "kernels" / "vecop_matmul.cl")

const CL_DEVICE_TYPE_ALL: uint64 = 0xFFFFFFFF'u64
const CL_MEM_READ_ONLY: uint64 = 4
const CL_MEM_WRITE_ONLY: uint64 = 2
const CL_TRUE: uint32 = 1

proc gpuCheck(cond: bool, msg: string) =
  ## Giống doAssert nhưng raise CatchableError thay vì Defect, để gpubackend.nim
  ## có thể bắt lỗi và fallback CPU thay vì làm crash chương trình.
  if not cond:
    raise newException(CatchableError, msg)

var lib: OclLib
var loaded = false

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  for name in ["libOpenCL.so", "libOpenCL.so.1", "OpenCL.dll", "/System/Library/Frameworks/OpenCL.framework/OpenCL"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil: return false
  lib.clGetPlatformIDs = cast[typeof(lib.clGetPlatformIDs)](lib.handle.symAddr("clGetPlatformIDs"))
  lib.clGetDeviceIDs = cast[typeof(lib.clGetDeviceIDs)](lib.handle.symAddr("clGetDeviceIDs"))
  lib.clCreateContext = cast[typeof(lib.clCreateContext)](lib.handle.symAddr("clCreateContext"))
  lib.clCreateCommandQueue = cast[typeof(lib.clCreateCommandQueue)](lib.handle.symAddr("clCreateCommandQueue"))
  lib.clCreateBuffer = cast[typeof(lib.clCreateBuffer)](lib.handle.symAddr("clCreateBuffer"))
  lib.clCreateProgramWithSource = cast[typeof(lib.clCreateProgramWithSource)](lib.handle.symAddr("clCreateProgramWithSource"))
  lib.clBuildProgram = cast[typeof(lib.clBuildProgram)](lib.handle.symAddr("clBuildProgram"))
  lib.clCreateKernel = cast[typeof(lib.clCreateKernel)](lib.handle.symAddr("clCreateKernel"))
  lib.clSetKernelArg = cast[typeof(lib.clSetKernelArg)](lib.handle.symAddr("clSetKernelArg"))
  lib.clEnqueueNDRangeKernel = cast[typeof(lib.clEnqueueNDRangeKernel)](lib.handle.symAddr("clEnqueueNDRangeKernel"))
  lib.clEnqueueWriteBuffer = cast[typeof(lib.clEnqueueWriteBuffer)](lib.handle.symAddr("clEnqueueWriteBuffer"))
  lib.clEnqueueReadBuffer = cast[typeof(lib.clEnqueueReadBuffer)](lib.handle.symAddr("clEnqueueReadBuffer"))
  lib.clFinish = cast[typeof(lib.clFinish)](lib.handle.symAddr("clFinish"))
  lib.clReleaseMemObject = cast[typeof(lib.clReleaseMemObject)](lib.handle.symAddr("clReleaseMemObject"))
  result = lib.clGetPlatformIDs != nil and lib.clCreateContext != nil

proc openclAvailable*(): bool =
  ## Dò xem có driver OpenCL nào (GPU hoặc CPU) trên máy hay không.
  if not tryLoad(): return false
  var platform: cl_platform_id
  var nPlat: cl_uint
  if lib.clGetPlatformIDs(1, addr platform, addr nPlat) != 0 or nPlat == 0: return false
  var device: cl_device_id
  var nDev: cl_uint
  result = lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, addr device, addr nDev) == 0 and nDev > 0

proc openclVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  ## Chạy phép toán elementwise trên GPU/CPU qua OpenCL. Raise nếu lỗi để fallback CPU thuần.
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  let n = a.len
  result = newSeq[float32](n)

  var platform: cl_platform_id
  var nPlat: cl_uint
  gpuCheck(lib.clGetPlatformIDs(1, addr platform, addr nPlat) == 0 and nPlat > 0, "no OpenCL platform")
  var device: cl_device_id
  var nDev: cl_uint
  gpuCheck(lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, addr device, addr nDev) == 0 and nDev > 0, "no OpenCL device")

  var err: cl_int
  let ctx = lib.clCreateContext(nil, 1, addr device, nil, nil, addr err)
  gpuCheck(err == 0 and ctx != nil, "clCreateContext failed")
  let queue = lib.clCreateCommandQueue(ctx, device, 0, addr err)
  gpuCheck(err == 0 and queue != nil, "clCreateCommandQueue failed")

  let bytes = csize_t(n * sizeof(float32))
  let bufA = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufB = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufC = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  defer:
    discard lib.clReleaseMemObject(bufA)
    discard lib.clReleaseMemObject(bufB)
    discard lib.clReleaseMemObject(bufC)

  var aVar = a
  var bVar = b
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytes, addr aVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytes, addr bVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  let prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  let kname = "vecop_" & op
  let kernel = lib.clCreateKernel(prog, kname.cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for " & kname)

  var bA = bufA
  var bB = bufB
  var bC = bufC
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bA) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bB) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bC) == 0, "GPU operation failed")

  var globalSize = csize_t(n)
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 1, nil, addr globalSize, nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytes, addr result[0], 0, nil, nil) == 0, "GPU operation failed")

proc openclMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
  ## C(m x n) = A(m x k) * B(k x n), row-major. Kernel naive (không tối ưu
  ## tile/shared-memory) chạy trên NDRange 2 chiều [m, n] -> đúng trên mọi
  ## GPU/CPU có driver OpenCL, không phụ thuộc đời phần cứng cụ thể.
  if not tryLoad():
    raise newException(CatchableError, "libOpenCL not found")
  result = newSeq[float32](m * n)

  var platform: cl_platform_id
  var nPlat: cl_uint
  gpuCheck(lib.clGetPlatformIDs(1, addr platform, addr nPlat) == 0 and nPlat > 0, "no OpenCL platform")
  var device: cl_device_id
  var nDev: cl_uint
  gpuCheck(lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, addr device, addr nDev) == 0 and nDev > 0, "no OpenCL device")

  var err: cl_int
  let ctx = lib.clCreateContext(nil, 1, addr device, nil, nil, addr err)
  gpuCheck(err == 0 and ctx != nil, "clCreateContext failed")
  let queue = lib.clCreateCommandQueue(ctx, device, 0, addr err)
  gpuCheck(err == 0 and queue != nil, "clCreateCommandQueue failed")

  let bytesA = csize_t(m * k * sizeof(float32))
  let bytesB = csize_t(k * n * sizeof(float32))
  let bytesC = csize_t(m * n * sizeof(float32))
  let bufA = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesA, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufB = lib.clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytesB, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  let bufC = lib.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytesC, nil, addr err)
  gpuCheck(err == 0, "GPU operation failed")
  defer:
    discard lib.clReleaseMemObject(bufA)
    discard lib.clReleaseMemObject(bufB)
    discard lib.clReleaseMemObject(bufC)

  var aVar = a
  var bVar = b
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, bytesA, addr aVar[0], 0, nil, nil) == 0, "GPU operation failed")
  gpuCheck(lib.clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, bytesB, addr bVar[0], 0, nil, nil) == 0, "GPU operation failed")

  var srcPtr = kernelSource.cstring
  let prog = lib.clCreateProgramWithSource(ctx, 1, addr srcPtr, nil, addr err)
  gpuCheck(err == 0 and prog != nil, "clCreateProgramWithSource failed")
  discard lib.clBuildProgram(prog, 1, addr device, nil, nil, nil)

  let kernel = lib.clCreateKernel(prog, "matmul_naive".cstring, addr err)
  gpuCheck(err == 0 and kernel != nil, "clCreateKernel failed for matmul_naive")

  var bA = bufA
  var bB = bufB
  var bC = bufC
  var mArg = int32(m)
  var kArg = int32(k)
  var nArg = int32(n)
  gpuCheck(lib.clSetKernelArg(kernel, 0, csize_t(sizeof(cl_mem)), addr bA) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 1, csize_t(sizeof(cl_mem)), addr bB) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 2, csize_t(sizeof(cl_mem)), addr bC) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 3, csize_t(sizeof(int32)), addr mArg) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 4, csize_t(sizeof(int32)), addr kArg) == 0, "GPU operation failed")
  gpuCheck(lib.clSetKernelArg(kernel, 5, csize_t(sizeof(int32)), addr nArg) == 0, "GPU operation failed")

  var globalSize2D = [csize_t(m), csize_t(n)]
  gpuCheck(lib.clEnqueueNDRangeKernel(queue, kernel, 2, nil, addr globalSize2D[0], nil, 0, nil, nil) == 0, "clEnqueueNDRangeKernel failed")
  discard lib.clFinish(queue)
  gpuCheck(lib.clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bytesC, addr result[0], 0, nil, nil) == 0, "GPU operation failed")
