## session_vm_dispatch.nim — cầu nối Nim ↔ backend encoder (Metal / Vulkan)
## cho micro_vm_kernel.
##
## Metal: metal_enc_* thật (metal_shim.m).
## Vulkan: multi-buffer JIT session API (vulkanJitBegin / vulkanBindBuf /
##         vulkanKernelDispatch) + 1-buffer vulkanSessionJitMicroVmEnc.
## CUDA đã bị loại — mọi path non-macOS đi qua Vulkan.

when defined(macosx):
  type MetalBufferHandle* = pointer

  proc metal_enc_pipeline(kernelName: cstring): cint {.
    importc: "metal_enc_pipeline", header: "metal_shim.h".}
  proc metal_enc_bind_buffer(index: cint; buf: MetalBufferHandle; offsetBytes: cint): cint {.
    importc: "metal_enc_bind_buffer", header: "metal_shim.h".}
  proc metal_enc_bind_int(index: cint; v: cint): cint {.
    importc: "metal_enc_bind_int", header: "metal_shim.h".}
  proc metal_enc_bind_float(index: cint; v: cfloat): cint {.
    importc: "metal_enc_bind_float", header: "metal_shim.h".}
  proc metal_enc_dispatch_threads(threadsX, threadsY, threadsZ, tgX, tgY, tgZ: cint): cint {.
    importc: "metal_enc_dispatch_threads", header: "metal_shim.h".}
  proc metal_enc_barrier(): cint {.
    importc: "metal_enc_barrier", header: "metal_shim.h".}
  proc metal_enc_set_threadgroup_bytes(index, nBytes: cint): cint {.
    importc: "metal_enc_set_threadgroup_bytes", header: "metal_shim.h".}
  proc metal_enc_simd_width(): cint {.
    importc: "metal_enc_simd_width", header: "metal_shim.h".}
  proc metal_enc_max_tg_threads(): cint {.
    importc: "metal_enc_max_tg_threads", header: "metal_shim.h".}

  proc sessionPipeline*(kernelName: string): bool =
    metal_enc_pipeline(kernelName.cstring) != 0

  proc sessionBindBuffer*(index: int; buf: MetalBufferHandle; offsetBytes = 0): bool =
    metal_enc_bind_buffer(index.cint, buf, offsetBytes.cint) != 0

  proc sessionBindInt*(index: int; v: int): bool =
    metal_enc_bind_int(index.cint, v.cint) != 0

  proc sessionBindFloat*(index: int; v: float32): bool =
    metal_enc_bind_float(index.cint, v.cfloat) != 0

  proc sessionDispatch*(nThreads: int; tgSize = 0): bool =
    ## Every current micro-VM dispatch may write a buffer that the next
    ## dispatch reads (Mamba state stages, Linear++ state/projection, GEMV and
    ## LoRA). Metal preserves command order but Intel GPUs need an explicit
    ## buffer barrier for visibility between compute dispatches. Centralize
    ## it here so a newly composed block cannot accidentally omit the hazard
    ## fence. This is an encoder barrier, not a CPU/GPU wait.
    if metal_enc_dispatch_threads(nThreads.cint, 1, 1, tgSize.cint, 1, 1) == 0:
      return false
    metal_enc_barrier() != 0

  proc sessionBarrier*(): bool =
    metal_enc_barrier() != 0

  proc sessionSetThreadgroupBytes*(index: int; nBytes: int): bool =
    metal_enc_set_threadgroup_bytes(index.cint, nBytes.cint) != 0

  proc sessionSimdWidth*(): int =
    ## MOI: doc threadExecutionWidth THAT cua pipeline dang duoc bind
    ## (sessionPipeline() da goi truoc do) - dung de dispatch dung so
    ## thread/simdgroup thay vi doan cung 32 (xem giai thich day du o
    ## fp4MatmulMetalSource trong lora.nim va fp4GemvGpuInto). PHAI goi SAU
    ## sessionPipeline(name), truoc khi dispatch.
    int(metal_enc_simd_width())

  proc sessionMaxTgThreads*(): int =
    ## MOI (FIX token rac bug #9): doc maxTotalThreadsPerThreadgroup THAT cua
    ## pipeline dang bind. Metal KHONG cho 1 threadgroup vuot qua so nay;
    ## metal_enc_dispatch_threads se AM THAM ha tgSize xuong maxTG neu ta xin
    ## nhieu hon -> moi kernel dung threadgroup memory + barrier (reduction)
    ## se bi chia thanh NHIEU threadgroup, moi cai reduce rieng tren vung nho
    ## RIENG -> ket qua RAC ma khong bao loi. PHAI goi SAU sessionPipeline().
    int(metal_enc_max_tg_threads())

else:
  type MetalBufferHandle* = pointer
  proc sessionPipeline*(kernelName: string): bool = true
  proc sessionBindBuffer*(index: int; buf: pointer; offsetBytes = 0): bool = true
  proc sessionBindInt*(index: int; v: int): bool = true
  proc sessionBindFloat*(index: int; v: float32): bool = true
  proc sessionDispatch*(nThreads: int; tgSize = 0): bool = true
  proc sessionBarrier*(): bool = true
  proc sessionSetThreadgroupBytes*(index: int; nBytes: int): bool = true
  proc sessionSimdWidth*(): int = 32
  proc sessionMaxTgThreads*(): int = 1024

# ── Vulkan multi-buffer session wrappers (available on all platforms) ──
# Inference encode path: pipeline → bind × N → dispatch, parallel to Metal.
# Path tương đối trong vendor/bybylang: micro_kernel → backends/vulkan
import ../backends/vulkan/vulkan_driver as vkDrv

proc sessionPipelineVk*(name, glslSrc: string): bool =
  vkDrv.vulkanJitBegin(name, glslSrc)

proc sessionBindBufferVk*(idx: int; buf: vkDrv.VulkanBufferHandle; offElems = 0): bool =
  vkDrv.vulkanBindBuf(idx, buf, offElems)

proc sessionDispatchVk*(nThreads, tgSize: int; tgBytes = 0): bool =
  vkDrv.vulkanKernelDispatch(nThreads, tgSize, tgBytes)

proc sessionSetThreadgroupBytesVk*(index, nBytes: int): bool =
  vkDrv.vulkanSetThreadgroupBytes(index, nBytes)

proc sessionMaxTgThreadsVk*(): int =
  vkDrv.vulkanMaxTgThreads()

# ════════════════════════════════════════════════════════════════════
# API CHUNG encode (Metal + Vulkan) — caller chi can isVulkan + name/glsl/handle.
# Shim native van tach; day la lop gom de encodeVm / encodeVmKernel khong nhai if.
# ════════════════════════════════════════════════════════════════════

proc sessionEncPipeline*(isVulkan: bool; name: string; glslSrc = ""): bool =
  ## Vulkan BAT BUOC glslSrc; Metal chi can ten pipeline da JIT.
  if isVulkan:
    if glslSrc.len == 0: return false
    return sessionPipelineVk(name, glslSrc)
  sessionPipeline(name)

proc sessionEncBindVkOrMetal*(isVulkan: bool; idx: int;
                              metalBuf: pointer;
                              vkBuf: vkDrv.VulkanBufferHandle;
                              offElems = 0): bool =
  ## Metal: offset = offElems * 4 (byte). Vulkan: offElems (float index).
  if isVulkan:
    return sessionBindBufferVk(idx, vkBuf, offElems)
  when defined(macosx):
    return sessionBindBuffer(idx, metalBuf, offElems * 4)
  else:
    return true

proc sessionEncDispatch*(isVulkan: bool; nThreads, tgSize: int; tgBytes = 0): bool =
  if isVulkan:
    return sessionDispatchVk(nThreads, tgSize, tgBytes)
  sessionDispatch(nThreads, tgSize)

proc sessionEncMaxTg*(isVulkan: bool): int =
  if isVulkan: return sessionMaxTgThreadsVk()
  sessionMaxTgThreads()

proc sessionEncTgBytes*(isVulkan: bool; index, nBytes: int): bool =
  if isVulkan: return sessionSetThreadgroupBytesVk(index, nBytes)
  sessionSetThreadgroupBytes(index, nBytes)

