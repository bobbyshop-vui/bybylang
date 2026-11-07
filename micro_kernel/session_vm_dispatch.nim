## session_vm_dispatch.nim — bridge between Nim and the backend encoders (Metal / Vulkan)
## cho micro_vm_kernel.
##
## Metal: the real metal_enc_* (metal_shim.m).
## Vulkan: multi-buffer JIT session API (vulkanJitBegin / vulkanBindBuf /
##         vulkanKernelDispatch) + 1-buffer vulkanSessionJitMicroVmEnc.
## CUDA was removed — every non-macOS path goes through Vulkan.

import ../probe

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
    result = metal_enc_pipeline(kernelName.cstring) != 0
    if result and unlikely(firProbeOn): probeEncoderPipeline(kernelName, "metal")

  proc sessionBindBuffer*(index: int; buf: MetalBufferHandle; offsetBytes = 0): bool =
    result = metal_enc_bind_buffer(index.cint, buf, offsetBytes.cint) != 0
    if result and unlikely(firProbeOn): probeEncoderBind(index, buf, max(0, offsetBytes))

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
    if unlikely(firProbeOn):
      let actualTg = if tgSize > 0: min(tgSize, int(metal_enc_max_tg_threads()))
                     else: int(metal_enc_simd_width())
      probeEncoderDispatch(nThreads, actualTg)
    metal_enc_barrier() != 0

  proc sessionBarrier*(): bool =
    metal_enc_barrier() != 0

  proc sessionSetThreadgroupBytes*(index: int; nBytes: int): bool =
    result = metal_enc_set_threadgroup_bytes(index.cint, nBytes.cint) != 0
    if result and unlikely(firProbeOn): probeEncoderTgBytes(nBytes)

  proc sessionSimdWidth*(): int =
    ## NEW: reads the REAL threadExecutionWidth of the bound pipeline
    ## (sessionPipeline() must have run) - used to dispatch the right number of
    ## threads/simdgroups instead of hardcoding 32 (see the full explanation in
    ## fp4MatmulMetalSource in lora.nim and fp4GemvGpuInto). MUST be called AFTER
    ## sessionPipeline(name), before dispatch.
    int(metal_enc_simd_width())

  proc sessionMaxTgThreads*(): int =
    ## NEW (fix for garbage-token bug #9): reads the REAL maxTotalThreadsPerThreadgroup of
    ## the bound pipeline. Metal does NOT allow a threadgroup larger than this;
    ## metal_enc_dispatch_threads SILENTLY lowers tgSize to maxTG if we ask for
    ## more -> any kernel using threadgroup memory + barriers (reductions)
    ## gets split into SEVERAL threadgroups, each reducing its OWN memory
    ## -> GARBAGE results without any error. MUST be called AFTER sessionPipeline().
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
# Relative path inside vendor/bybylang: micro_kernel → backends/vulkan
import ../backends/vulkan/vulkan_driver as vkDrv

proc sessionPipelineVk*(name, glslSrc: string): bool =
  result = vkDrv.vulkanJitBegin(name, glslSrc)
  if result and unlikely(firProbeOn): probeEncoderPipeline(name, "vulkan")

proc sessionBindBufferVk*(idx: int; buf: vkDrv.VulkanBufferHandle; offElems = 0): bool =
  result = vkDrv.vulkanBindBuf(idx, buf, offElems)
  if result and unlikely(firProbeOn):
    probeEncoderBind(idx, cast[pointer](buf.buf), offElems*4, int(buf.sizeBytes))

proc sessionDispatchVk*(nThreads, tgSize: int; tgBytes = 0): bool =
  result = vkDrv.vulkanKernelDispatch(nThreads, tgSize, tgBytes)
  if result and unlikely(firProbeOn): probeEncoderDispatch(nThreads, tgSize, tgBytes)

proc sessionSetThreadgroupBytesVk*(index, nBytes: int): bool =
  result = vkDrv.vulkanSetThreadgroupBytes(index, nBytes)
  if result and unlikely(firProbeOn): probeEncoderTgBytes(nBytes)

proc sessionMaxTgThreadsVk*(): int =
  vkDrv.vulkanMaxTgThreads()

# ════════════════════════════════════════════════════════════════════
# SHARED encode API (Metal + Vulkan) — the caller only needs isVulkan + name/glsl/handle.
# The native shims stay separate; this layer keeps encodeVm / encodeVmKernel free of if-chains.
# ════════════════════════════════════════════════════════════════════

proc sessionEncPipeline*(isVulkan: bool; name: string; glslSrc = ""): bool =
  ## Vulkan REQUIRES glslSrc; Metal only needs the name of an already JIT-compiled pipeline.
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

