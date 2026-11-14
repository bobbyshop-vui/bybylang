## Vulkan/MoltenVK compute sessions and resident buffers.
## Optional KHR cooperative-matrix inference GEMM uses FP16 tiles and FP32
## accumulation. Device features/limits are queried before creating the device.
## CUDA libraries are not required by this backend.

import std/dynlib
import std/os
import std/math
import std/[osproc, hashes, strutils, sets, algorithm, monotimes, times, json]
import vulkan_caps
import ../../probe
export vulkan_caps.VulkanCaps

type
  VkInstance* = pointer
  VkPhysicalDevice* = pointer
  VkDevice* = pointer
  VkQueue* = pointer
  VkBuffer* = pointer
  VkDeviceMemory* = pointer
  VkShaderModule* = pointer
  VkPipeline* = pointer
  VkPipelineLayout* = pointer
  VkDescriptorSetLayout* = pointer
  VkDescriptorPool* = pointer
  VkDescriptorSet* = pointer
  VkCommandPool* = pointer
  VkCommandBuffer* = pointer
  VkFence* = pointer
  VkQueryPool* = pointer
  VkPipelineCache* = pointer
  VkResult* = int32

  # ---- Minimal Vulkan structs needed for step 1 (fields in the same order as
  # the real vulkan_core.h - A WRONG ORDER FAILS SILENTLY, not at compile time,
  # because a Nim object is passed with its declared layout, which matches the C struct only when
  # field order + types match EXACTLY). ----
  VkApplicationInfo* = object
    sType*: uint32          # VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
    pNext*: pointer
    pApplicationName*: cstring
    applicationVersion*: uint32
    pEngineName*: cstring
    engineVersion*: uint32
    apiVersion*: uint32

  VkInstanceCreateInfo* = object
    sType*: uint32          # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
    pNext*: pointer
    flags*: uint32
    pApplicationInfo*: ptr VkApplicationInfo
    enabledLayerCount*: uint32
    ppEnabledLayerNames*: ptr cstring
    enabledExtensionCount*: uint32
    ppEnabledExtensionNames*: ptr cstring

  VkQueueFamilyProperties* = object
    queueFlags*: uint32
    queueCount*: uint32
    timestampValidBits*: uint32
    minImageTransferGranularityWidth*: uint32
    minImageTransferGranularityHeight*: uint32
    minImageTransferGranularityDepth*: uint32

  VkDeviceQueueCreateInfo* = object
    sType*: uint32           # VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
    pNext*: pointer
    flags*: uint32
    queueFamilyIndex*: uint32
    queueCount*: uint32
    pQueuePriorities*: ptr float32

  VkDeviceCreateInfo* = object
    sType*: uint32            # VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
    pNext*: pointer
    flags*: uint32
    queueCreateInfoCount*: uint32
    pQueueCreateInfos*: ptr VkDeviceQueueCreateInfo
    enabledLayerCount*: uint32
    ppEnabledLayerNames*: ptr cstring
    enabledExtensionCount*: uint32
    ppEnabledExtensionNames*: ptr cstring
    pEnabledFeatures*: pointer

  VkBufferCreateInfo* = object
    sType*: uint32             # VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
    pNext*: pointer
    flags*: uint32
    size*: uint64
    usage*: uint32
    sharingMode*: uint32
    queueFamilyIndexCount*: uint32
    pQueueFamilyIndices*: ptr uint32

  VkMemoryRequirements* = object
    size*: uint64
    alignment*: uint64
    memoryTypeBits*: uint32

  VkMemoryType* = object
    propertyFlags*: uint32
    heapIndex*: uint32

  VkMemoryHeap* = object
    size*: uint64
    flags*: uint32

  VkPhysicalDeviceMemoryProperties* = object
    memoryTypeCount*: uint32
    memoryTypes*: array[32, VkMemoryType]
    memoryHeapCount*: uint32
    memoryHeaps*: array[16, VkMemoryHeap]

  VkMemoryAllocateInfo* = object
    sType*: uint32              # VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
    pNext*: pointer
    allocationSize*: uint64
    memoryTypeIndex*: uint32

  VkShaderModuleCreateInfo* = object
    sType*: uint32               # VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16
    pNext*: pointer
    flags*: uint32
    codeSize*: uint             # really csize_t; uint matches a 64-bit size_t
    pCode*: ptr uint32           # pointer to the SPIR-V bytecode (multiple of 4 bytes)

  VkDescriptorSetLayoutBinding* = object
    binding*: uint32
    descriptorType*: uint32
    descriptorCount*: uint32
    stageFlags*: uint32
    pImmutableSamplers*: pointer

  VkDescriptorSetLayoutCreateInfo* = object
    sType*: uint32                # VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
    pNext*: pointer
    flags*: uint32
    bindingCount*: uint32
    pBindings*: ptr VkDescriptorSetLayoutBinding

  VkPipelineLayoutCreateInfo* = object
    sType*: uint32                 # VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
    pNext*: pointer
    flags*: uint32
    setLayoutCount*: uint32
    pSetLayouts*: ptr VkDescriptorSetLayout
    pushConstantRangeCount*: uint32
    pPushConstantRanges*: pointer

  VkPipelineShaderStageCreateInfo* = object
    sType*: uint32                  # VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
    pNext*: pointer
    flags*: uint32
    stage*: uint32                  # VK_SHADER_STAGE_COMPUTE_BIT = 0x20
    module*: VkShaderModule
    pName*: cstring
    pSpecializationInfo*: pointer

  VkSpecializationMapEntry* = object
    constantID*: uint32
    offset*: uint32
    size*: uint

  VkSpecializationInfo* = object
    mapEntryCount*: uint32
    pMapEntries*: ptr VkSpecializationMapEntry
    dataSize*: uint
    pData*: pointer

  VkComputePipelineCreateInfo* = object
    sType*: uint32                   # VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
    pNext*: pointer
    flags*: uint32
    stage*: VkPipelineShaderStageCreateInfo
    layout*: VkPipelineLayout
    basePipelineHandle*: VkPipeline
    basePipelineIndex*: int32

  VkDescriptorPoolSize* = object
    descriptorType*: uint32
    descriptorCount*: uint32

  VkDescriptorPoolCreateInfo* = object
    sType*: uint32                    # VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33
    pNext*: pointer
    flags*: uint32
    maxSets*: uint32
    poolSizeCount*: uint32
    pPoolSizes*: ptr VkDescriptorPoolSize

  VkDescriptorSetAllocateInfo* = object
    sType*: uint32                     # VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34
    pNext*: pointer
    descriptorPool*: VkDescriptorPool
    descriptorSetCount*: uint32
    pSetLayouts*: ptr VkDescriptorSetLayout

  VkDescriptorBufferInfo* = object
    buffer*: VkBuffer
    offset*: uint64
    range*: uint64

  VkWriteDescriptorSet* = object
    sType*: uint32                      # VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35
    pNext*: pointer
    dstSet*: VkDescriptorSet
    dstBinding*: uint32
    dstArrayElement*: uint32
    descriptorCount*: uint32
    descriptorType*: uint32
    pImageInfo*: pointer
    pBufferInfo*: ptr VkDescriptorBufferInfo
    pTexelBufferView*: pointer

  VkCommandPoolCreateInfo* = object
    sType*: uint32                       # VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
    pNext*: pointer
    flags*: uint32
    queueFamilyIndex*: uint32

  VkCommandBufferAllocateInfo* = object
    sType*: uint32                        # VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
    pNext*: pointer
    commandPool*: VkCommandPool
    level*: uint32
    commandBufferCount*: uint32

  VkCommandBufferBeginInfo* = object
    sType*: uint32                         # VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
    pNext*: pointer
    flags*: uint32
    pInheritanceInfo*: pointer

  VkSubmitInfo* = object
    sType*: uint32                          # VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
    pNext*: pointer
    waitSemaphoreCount*: uint32
    pWaitSemaphores*: pointer
    pWaitDstStageMask*: pointer
    commandBufferCount*: uint32
    pCommandBuffers*: ptr VkCommandBuffer
    signalSemaphoreCount*: uint32
    pSignalSemaphores*: pointer

  VkFenceCreateInfo* = object
    sType*: uint32                           # VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
    pNext*: pointer
    flags*: uint32

  VkQueryPoolCreateInfo* = object
    sType*: uint32                           # VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO = 11
    pNext*: pointer
    flags*: uint32
    queryType*: uint32                       # VK_QUERY_TYPE_TIMESTAMP = 2
    queryCount*: uint32
    pipelineStatistics*: uint32

  VkPipelineCacheCreateInfo* = object
    sType*: uint32                         # VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO = 17
    pNext*: pointer
    flags*: uint32
    initialDataSize*: uint
    pInitialData*: pointer

  VkPushConstantRange* = object
    stageFlags*: uint32
    offset*: uint32
    size*: uint32

  VkBufferCopy* = object
    srcOffset*: uint64
    dstOffset*: uint64
    size*: uint64

  VkMemoryBarrier* = object
    sType*: uint32
    pNext*: pointer
    srcAccessMask*: uint32
    dstAccessMask*: uint32

const
  VK_STRUCTURE_TYPE_APPLICATION_INFO: uint32 = 0
  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: uint32 = 1
  VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: uint32 = 2
  VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: uint32 = 3
  VK_STRUCTURE_TYPE_SUBMIT_INFO: uint32 = 4
  VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: uint32 = 5
  VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: uint32 = 8
  VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO: uint32 = 11
  VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO: uint32 = 17
  VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: uint32 = 12
  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: uint32 = 18
  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: uint32 = 29
  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: uint32 = 30
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: uint32 = 32
  VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: uint32 = 33
  VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: uint32 = 34
  VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: uint32 = 35
  VK_STRUCTURE_TYPE_MEMORY_BARRIER: uint32 = 46
  VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: uint32 = 16
  VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: uint32 = 39
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: uint32 = 40
  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: uint32 = 42

  VK_QUEUE_COMPUTE_BIT: uint32 = 0x00000002
  VK_SHADER_STAGE_COMPUTE_BIT: uint32 = 0x00000020
  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: uint32 = 0x00000020
  VK_BUFFER_USAGE_TRANSFER_SRC_BIT: uint32 = 0x00000001
  VK_BUFFER_USAGE_TRANSFER_DST_BIT: uint32 = 0x00000002
  VK_SHARING_MODE_EXCLUSIVE: uint32 = 0
  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: uint32 = 0x00000001
  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: uint32 = 0x00000002
  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: uint32 = 0x00000004
  VK_MEMORY_PROPERTY_HOST_CACHED_BIT: uint32 = 0x00000008
  VK_PIPELINE_STAGE_HOST_BIT: uint32 = 0x00004000
  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: uint32 = 0x00000001
  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: uint32 = 0x00002000
  VK_ACCESS_HOST_READ_BIT: uint32 = 0x00002000
  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: uint32 = 7
  VK_PIPELINE_BIND_POINT_COMPUTE: uint32 = 1
  VK_COMMAND_BUFFER_LEVEL_PRIMARY: uint32 = 0
  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: uint32 = 0x00000001
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: uint32 = 0x00000800
  VK_ACCESS_SHADER_READ_BIT: uint32 = 0x00000020
  VK_ACCESS_SHADER_WRITE_BIT: uint32 = 0x00000040
  VK_PIPELINE_STAGE_TRANSFER_BIT: uint32 = 0x00001000
  VK_ACCESS_TRANSFER_READ_BIT: uint32 = 0x00000800
  VK_ACCESS_TRANSFER_WRITE_BIT: uint32 = 0x00001000
  VK_QUERY_TYPE_TIMESTAMP: uint32 = 2
  VK_QUERY_RESULT_64_BIT: uint32 = 0x00000001
  VK_SUCCESS*: VkResult = 0

  # MoltenVK/macOS REQUIRES these 2 portability flags when creating the instance (from
  # Vulkan 1.3.216 on macOS) - without them VK_ERROR_INCOMPATIBLE_DRIVER.
  VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR: uint32 = 0x00000001
  KHR_PORTABILITY_ENUMERATION_EXTENSION: cstring = "VK_KHR_portability_enumeration"

type
  VulkanLib = object
    handle: LibHandle
    vkGetInstanceProcAddr: proc(instance: VkInstance, name: cstring): pointer {.cdecl.}
    vkCreateInstance: proc(pCreateInfo: ptr VkInstanceCreateInfo, pAllocator: pointer, pInstance: ptr VkInstance): VkResult {.cdecl.}
    vkEnumeratePhysicalDevices: proc(instance: VkInstance, pCount: ptr uint32, pDevices: ptr VkPhysicalDevice): VkResult {.cdecl.}
    vkGetPhysicalDeviceQueueFamilyProperties: proc(dev: VkPhysicalDevice, pCount: ptr uint32, pProps: ptr VkQueueFamilyProperties) {.cdecl.}
    vkCreateDevice: proc(physDev: VkPhysicalDevice, pCreateInfo: ptr VkDeviceCreateInfo, pAllocator: pointer, pDevice: ptr VkDevice): VkResult {.cdecl.}
    vkGetDeviceQueue: proc(dev: VkDevice, queueFamilyIndex: uint32, queueIndex: uint32, pQueue: ptr VkQueue) {.cdecl.}
    vkGetPhysicalDeviceMemoryProperties: proc(dev: VkPhysicalDevice, pProps: ptr VkPhysicalDeviceMemoryProperties) {.cdecl.}
    vkCreateBuffer: proc(dev: VkDevice, pCreateInfo: ptr VkBufferCreateInfo, pAllocator: pointer, pBuffer: ptr VkBuffer): VkResult {.cdecl.}
    vkGetBufferMemoryRequirements: proc(dev: VkDevice, buffer: VkBuffer, pReq: ptr VkMemoryRequirements) {.cdecl.}
    vkAllocateMemory: proc(dev: VkDevice, pAllocInfo: ptr VkMemoryAllocateInfo, pAllocator: pointer, pMemory: ptr VkDeviceMemory): VkResult {.cdecl.}
    vkBindBufferMemory: proc(dev: VkDevice, buffer: VkBuffer, memory: VkDeviceMemory, offset: uint64): VkResult {.cdecl.}
    vkMapMemory: proc(dev: VkDevice, memory: VkDeviceMemory, offset: uint64, size: uint64, flags: uint32, ppData: ptr pointer): VkResult {.cdecl.}
    vkUnmapMemory: proc(dev: VkDevice, memory: VkDeviceMemory) {.cdecl.}
    vkCreateShaderModule: proc(dev: VkDevice, pCreateInfo: ptr VkShaderModuleCreateInfo, pAllocator: pointer, pModule: ptr VkShaderModule): VkResult {.cdecl.}
    vkCreateDescriptorSetLayout: proc(dev: VkDevice, pCreateInfo: ptr VkDescriptorSetLayoutCreateInfo, pAllocator: pointer, pLayout: ptr VkDescriptorSetLayout): VkResult {.cdecl.}
    vkCreatePipelineLayout: proc(dev: VkDevice, pCreateInfo: ptr VkPipelineLayoutCreateInfo, pAllocator: pointer, pLayout: ptr VkPipelineLayout): VkResult {.cdecl.}
    vkCreateComputePipelines: proc(dev: VkDevice, cache: pointer, count: uint32, pInfos: ptr VkComputePipelineCreateInfo, pAllocator: pointer, pPipelines: ptr VkPipeline): VkResult {.cdecl.}
    vkCreatePipelineCache: proc(dev: VkDevice, pCreateInfo: ptr VkPipelineCacheCreateInfo, pAllocator: pointer, pCache: ptr VkPipelineCache): VkResult {.cdecl.}
    vkDestroyPipelineCache: proc(dev: VkDevice, cache: VkPipelineCache, pAllocator: pointer) {.cdecl.}
    vkGetPipelineCacheData: proc(dev: VkDevice, cache: VkPipelineCache, pDataSize: ptr uint, pData: pointer): VkResult {.cdecl.}
    vkCreateDescriptorPool: proc(dev: VkDevice, pCreateInfo: ptr VkDescriptorPoolCreateInfo, pAllocator: pointer, pPool: ptr VkDescriptorPool): VkResult {.cdecl.}
    vkAllocateDescriptorSets: proc(dev: VkDevice, pAllocInfo: ptr VkDescriptorSetAllocateInfo, pSets: ptr VkDescriptorSet): VkResult {.cdecl.}
    vkUpdateDescriptorSets: proc(dev: VkDevice, writeCount: uint32, pWrites: ptr VkWriteDescriptorSet, copyCount: uint32, pCopies: pointer) {.cdecl.}
    vkCreateCommandPool: proc(dev: VkDevice, pCreateInfo: ptr VkCommandPoolCreateInfo, pAllocator: pointer, pPool: ptr VkCommandPool): VkResult {.cdecl.}
    vkAllocateCommandBuffers: proc(dev: VkDevice, pAllocInfo: ptr VkCommandBufferAllocateInfo, pBuffers: ptr VkCommandBuffer): VkResult {.cdecl.}
    vkBeginCommandBuffer: proc(cmdBuf: VkCommandBuffer, pBeginInfo: ptr VkCommandBufferBeginInfo): VkResult {.cdecl.}
    vkEndCommandBuffer: proc(cmdBuf: VkCommandBuffer): VkResult {.cdecl.}
    vkCmdBindPipeline: proc(cmdBuf: VkCommandBuffer, bindPoint: uint32, pipeline: VkPipeline) {.cdecl.}
    vkCmdBindDescriptorSets: proc(cmdBuf: VkCommandBuffer, bindPoint: uint32, layout: VkPipelineLayout, firstSet: uint32, count: uint32, pSets: ptr VkDescriptorSet, dynCount: uint32, pDynOffsets: pointer) {.cdecl.}
    vkCmdDispatch: proc(cmdBuf: VkCommandBuffer, x, y, z: uint32) {.cdecl.}
    vkCmdPushConstants: proc(cmdBuf: VkCommandBuffer, layout: VkPipelineLayout, stageFlags: uint32, offset: uint32, size: uint32, pValues: pointer) {.cdecl.}
    vkCmdPipelineBarrier: proc(cmdBuf: VkCommandBuffer, srcStageMask, dstStageMask: uint32,
      dependencyFlags: uint32, memoryBarrierCount: uint32, pMemoryBarriers: ptr VkMemoryBarrier,
      bufferMemoryBarrierCount: uint32, pBufferMemoryBarriers: pointer,
      imageMemoryBarrierCount: uint32, pImageMemoryBarriers: pointer) {.cdecl.}
    vkCreateFence: proc(dev: VkDevice, pCreateInfo: ptr VkFenceCreateInfo, pAllocator: pointer, pFence: ptr VkFence): VkResult {.cdecl.}
    vkCreateQueryPool: proc(dev: VkDevice, pCreateInfo: ptr VkQueryPoolCreateInfo, pAllocator: pointer, pQueryPool: ptr VkQueryPool): VkResult {.cdecl.}
    vkDestroyQueryPool: proc(dev: VkDevice, queryPool: VkQueryPool, pAllocator: pointer) {.cdecl.}
    vkCmdResetQueryPool: proc(cmdBuf: VkCommandBuffer, queryPool: VkQueryPool, firstQuery, queryCount: uint32) {.cdecl.}
    vkCmdWriteTimestamp: proc(cmdBuf: VkCommandBuffer, pipelineStage: uint32, queryPool: VkQueryPool, query: uint32) {.cdecl.}
    vkGetQueryPoolResults: proc(dev: VkDevice, queryPool: VkQueryPool, firstQuery, queryCount: uint32,
      dataSize: uint, pData: pointer, stride: uint64, flags: uint32): VkResult {.cdecl.}
    vkWaitForFences: proc(dev: VkDevice, count: uint32, pFences: ptr VkFence, waitAll: uint32, timeout: uint64): VkResult {.cdecl.}
    vkResetFences: proc(dev: VkDevice, count: uint32, pFences: ptr VkFence): VkResult {.cdecl.}
    vkQueueSubmit: proc(queue: VkQueue, count: uint32, pSubmits: ptr VkSubmitInfo, fence: VkFence): VkResult {.cdecl.}
    vkDeviceWaitIdle: proc(dev: VkDevice): VkResult {.cdecl.}
    vkDestroyBuffer: proc(dev: VkDevice, buffer: VkBuffer, pAllocator: pointer) {.cdecl.}
    vkFreeMemory: proc(dev: VkDevice, memory: VkDeviceMemory, pAllocator: pointer) {.cdecl.}
    vkDestroyShaderModule: proc(dev: VkDevice, module: VkShaderModule, pAllocator: pointer) {.cdecl.}
    vkDestroyPipeline: proc(dev: VkDevice, pipeline: VkPipeline, pAllocator: pointer) {.cdecl.}
    vkDestroyPipelineLayout: proc(dev: VkDevice, layout: VkPipelineLayout, pAllocator: pointer) {.cdecl.}
    vkDestroyDescriptorSetLayout: proc(dev: VkDevice, layout: VkDescriptorSetLayout, pAllocator: pointer) {.cdecl.}
    vkDestroyDescriptorPool: proc(dev: VkDevice, pool: VkDescriptorPool, pAllocator: pointer) {.cdecl.}
    vkDestroyFence: proc(dev: VkDevice, fence: VkFence, pAllocator: pointer) {.cdecl.}
    vkFreeCommandBuffers: proc(dev: VkDevice, pool: VkCommandPool, count: uint32, pBufs: ptr VkCommandBuffer) {.cdecl.}
    vkFreeDescriptorSets: proc(dev: VkDevice, pool: VkDescriptorPool, count: uint32, pSets: ptr VkDescriptorSet): VkResult {.cdecl.}
    vkCmdCopyBuffer: proc(cmdBuf: VkCommandBuffer, src, dst: VkBuffer, regionCount: uint32, pRegions: pointer) {.cdecl.}
    vkResetCommandBuffer: proc(cmdBuf: VkCommandBuffer, flags: uint32): VkResult {.cdecl.}

type
  VulkanBufferHandle* = object
    ## Vulkan NEEDS BOTH (VkBuffer + VkDeviceMemory) to use a buffer -
    ## unlike Metal/CUDA/OpenCL with a single handle - wrapped in one struct
    ## to fit the shared SessionHandle interface (one field per backend).
    buf*: VkBuffer
    mem*: VkDeviceMemory
    sizeBytes*: uint64
    ## deviceOnly: VRAM without a host mapping (dGPU); reads and uploads go
    ## through a staging copy. pooled: returned to the buffer pool on free.
    deviceOnly*: bool
    pooled*: bool
    allocBytes*: uint64
    mapped*: pointer   ## persistent host mapping of a pooled staging buffer

proc gpuCheck(cond: bool, msg: string) =
  ## Same pattern as cuda_driver.nim - raises CatchableError (not
  ## Defect) so the layer above can catch it and fall back instead of crashing the process.
  if not cond:
    raise newException(CatchableError, msg)

proc gpuCheckVk(result: VkResult, msg: string) =
  ## Check a VkResult and raise CatchableError with the numeric VkResult code
  ## included in the message for debugging.
  if result != VK_SUCCESS:
    let name =
      case result
      of 0: "VK_SUCCESS"
      of -1: "VK_ERROR_OUT_OF_HOST_MEMORY"
      of -2: "VK_ERROR_OUT_OF_DEVICE_MEMORY"
      of -3: "VK_ERROR_INITIALIZATION_FAILED"
      of -4: "VK_ERROR_DEVICE_LOST"
      of -5: "VK_ERROR_MEMORY_MAP_FAILED"
      of -6: "VK_ERROR_LAYER_NOT_PRESENT"
      of -7: "VK_ERROR_EXTENSION_NOT_PRESENT"
      of -8: "VK_ERROR_FEATURE_NOT_PRESENT"
      else: "VkResult(" & $result & ")"
    raise newException(CatchableError, msg & " [" & name & " raw=" & $result & "]")

var lib: VulkanLib
var loaded = false
var gInitialized = false
var gInstance: VkInstance
var gPhysicalDevice: VkPhysicalDevice
var gDeviceCaps: VulkanCaps
var gCoopEnabled = false
var gFp8CoopEnabled = false
## Prefill (M > 1) GEMMs must stay on the tensor cores: set on a dGPU with
## KHR_cooperative_matrix (see BYBY_VULKAN_TENSOR_CORES at device creation).
var gCoopRequired = false
var gFp8Fp4Logged = false
var gHostImportEnabled = false
## Session scratch/upload buffers in VRAM (dGPU). Host-visible system memory
## there means every kernel reads its activations across PCIe.
var gScratchDeviceLocal = false
var vkGetMemoryHostPointerPropertiesEXT: proc(dev: VkDevice; handleType: uint32;
  host: pointer; props: pointer): VkResult {.cdecl.}
var gFp16Fp4Logged = false
var gDevice: VkDevice
var gQueue: VkQueue
var gQueueFamilyIndex: uint32
var gQueueTimestampValidBits: uint32
var gCommandPool: VkCommandPool

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  # MoltenVK can be installed under one of several names depending on the install (the Vulkan SDK for
  # macOS usually installs libMoltenVK.dylib + libvulkan.1.dylib as the loader pointing
  # to it - both common names are tried). On Linux the standard loader is
  # libvulkan.so.1 (package vulkan-loader/libvulkan1) - that name USED TO BE MISSING,
  # so tryLoad() always failed on Linux even with a real Vulkan driver.
  for name in ["vulkan-1.dll", "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib",
               "libvulkan.so.1", "libvulkan.so"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil:
    return false
  lib.vkGetInstanceProcAddr = cast[typeof(lib.vkGetInstanceProcAddr)](lib.handle.symAddr("vkGetInstanceProcAddr"))
  if lib.vkGetInstanceProcAddr == nil:
    return false
  # vkCreateInstance is the ONLY function resolved DIRECTLY through the library's symAddr
  # (no instance needed yet) - all others MUST be resolved through
  # vkGetInstanceProcAddr(instance, ...) AFTER a real instance exists (as
  # the Vulkan loader spec requires - NOT optional; without it some functions
  # return nil on a real ICD driver even though a direct dlsym "seems" to give an address).
  lib.vkCreateInstance = cast[typeof(lib.vkCreateInstance)](lib.handle.symAddr("vkCreateInstance"))
  result = lib.vkCreateInstance != nil

proc loadInstanceProc[T](name: string): T =
  let p = lib.vkGetInstanceProcAddr(gInstance, name.cstring)
  gpuCheck(p != nil, "vkGetInstanceProcAddr returned nil for function: " & name)
  result = cast[T](p)

proc ensureInit() =
  if gInitialized: return
  gpuCheck(tryLoad(), "Vulkan loader library not found/loadable (libvulkan.so.1 on Linux, libvulkan.1.dylib/libMoltenVK.dylib on macOS) - check that a Vulkan driver/ICD (e.g. package vulkan-loader or vulkan-icd-loader) or MoltenVK/the Vulkan SDK is installed, and that LD_LIBRARY_PATH/DYLD_LIBRARY_PATH points to its directory")

  # ---- 1. Create the VkInstance (with the portability flag MoltenVK requires) ----
  var appInfo = VkApplicationInfo(
    sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: nil,
    pApplicationName: "bybylang_vulkan",
    applicationVersion: 1,
    pEngineName: "bybylang",
    engineVersion: 1,
    apiVersion: 0x00403000'u32  # VK_API_VERSION_1_3 = (1<<22)|(3<<12)|0, tinh tay: (1 shl 22) or (3 shl 12)
  )
  var extNames = [KHR_PORTABILITY_ENUMERATION_EXTENSION]
  var instInfo = VkInstanceCreateInfo(
    sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: nil,
    flags: VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    pApplicationInfo: addr appInfo,
    enabledLayerCount: 0,
    ppEnabledLayerNames: nil,
    enabledExtensionCount: 1,
    ppEnabledExtensionNames: addr extNames[0]
  )
  var inst: VkInstance
  var rInst = lib.vkCreateInstance(addr instInfo, nil, addr inst)

  # RETRY WITHOUT THE PORTABILITY FLAG: VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
  # + the "VK_KHR_portability_enumeration" extension belong to the VULKAN LOADER LAYER
  # (so the loader also lists "non-conformant" ICDs such as MoltenVK). If this machine
  # has NO real Vulkan loader (e.g. libvulkan.1.dylib missing) and tryLoad()
  # had to load libMoltenVK.dylib DIRECTLY (a bare ICD, no loader),
  # that bare ICD does NOT understand this flag/extension -> it returns exactly
  # VK_ERROR_INCOMPATIBLE_DRIVER (-9) although the GPU driver is fine. In that
  # case the instance is created a second time WITHOUT the portability flag/extension.
  if rInst != VK_SUCCESS:
    var instInfoNoPortability = VkInstanceCreateInfo(
      sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      pNext: nil,
      flags: 0,
      pApplicationInfo: addr appInfo,
      enabledLayerCount: 0,
      ppEnabledLayerNames: nil,
      enabledExtensionCount: 0,
      ppEnabledExtensionNames: nil
    )
    rInst = lib.vkCreateInstance(addr instInfoNoPortability, nil, addr inst)

  gpuCheckVk(rInst, "vkCreateInstance failed both times (with and without the " &
    "portability extension). If -9=VK_ERROR_INCOMPATIBLE_DRIVER: this machine lacks a real Vulkan loader/ICD " &
    "(no longer an instance-creation logic error) - install: 'brew install molten-vk vulkan-loader' " &
    "and set VK_ICD_FILENAMES to the MoltenVK_icd.json file (see 'brew --prefix molten-vk')")
  gInstance = inst

  # ---- 2. Resolve the remaining functions THROUGH vkGetInstanceProcAddr(instance,...) ----
  lib.vkEnumeratePhysicalDevices = loadInstanceProc[typeof(lib.vkEnumeratePhysicalDevices)]("vkEnumeratePhysicalDevices")
  lib.vkGetPhysicalDeviceQueueFamilyProperties = loadInstanceProc[typeof(lib.vkGetPhysicalDeviceQueueFamilyProperties)]("vkGetPhysicalDeviceQueueFamilyProperties")
  lib.vkCreateDevice = loadInstanceProc[typeof(lib.vkCreateDevice)]("vkCreateDevice")
  lib.vkGetDeviceQueue = loadInstanceProc[typeof(lib.vkGetDeviceQueue)]("vkGetDeviceQueue")
  lib.vkGetPhysicalDeviceMemoryProperties = loadInstanceProc[typeof(lib.vkGetPhysicalDeviceMemoryProperties)]("vkGetPhysicalDeviceMemoryProperties")
  lib.vkCreateBuffer = loadInstanceProc[typeof(lib.vkCreateBuffer)]("vkCreateBuffer")
  lib.vkGetBufferMemoryRequirements = loadInstanceProc[typeof(lib.vkGetBufferMemoryRequirements)]("vkGetBufferMemoryRequirements")
  lib.vkAllocateMemory = loadInstanceProc[typeof(lib.vkAllocateMemory)]("vkAllocateMemory")
  lib.vkBindBufferMemory = loadInstanceProc[typeof(lib.vkBindBufferMemory)]("vkBindBufferMemory")
  lib.vkMapMemory = loadInstanceProc[typeof(lib.vkMapMemory)]("vkMapMemory")
  lib.vkUnmapMemory = loadInstanceProc[typeof(lib.vkUnmapMemory)]("vkUnmapMemory")
  lib.vkCreateShaderModule = loadInstanceProc[typeof(lib.vkCreateShaderModule)]("vkCreateShaderModule")
  lib.vkCreateDescriptorSetLayout = loadInstanceProc[typeof(lib.vkCreateDescriptorSetLayout)]("vkCreateDescriptorSetLayout")
  lib.vkCreatePipelineLayout = loadInstanceProc[typeof(lib.vkCreatePipelineLayout)]("vkCreatePipelineLayout")
  lib.vkCreateComputePipelines = loadInstanceProc[typeof(lib.vkCreateComputePipelines)]("vkCreateComputePipelines")
  lib.vkCreatePipelineCache = loadInstanceProc[typeof(lib.vkCreatePipelineCache)]("vkCreatePipelineCache")
  lib.vkDestroyPipelineCache = loadInstanceProc[typeof(lib.vkDestroyPipelineCache)]("vkDestroyPipelineCache")
  lib.vkGetPipelineCacheData = loadInstanceProc[typeof(lib.vkGetPipelineCacheData)]("vkGetPipelineCacheData")
  lib.vkCreateDescriptorPool = loadInstanceProc[typeof(lib.vkCreateDescriptorPool)]("vkCreateDescriptorPool")
  lib.vkAllocateDescriptorSets = loadInstanceProc[typeof(lib.vkAllocateDescriptorSets)]("vkAllocateDescriptorSets")
  lib.vkUpdateDescriptorSets = loadInstanceProc[typeof(lib.vkUpdateDescriptorSets)]("vkUpdateDescriptorSets")
  lib.vkCreateCommandPool = loadInstanceProc[typeof(lib.vkCreateCommandPool)]("vkCreateCommandPool")
  lib.vkAllocateCommandBuffers = loadInstanceProc[typeof(lib.vkAllocateCommandBuffers)]("vkAllocateCommandBuffers")
  lib.vkBeginCommandBuffer = loadInstanceProc[typeof(lib.vkBeginCommandBuffer)]("vkBeginCommandBuffer")
  lib.vkEndCommandBuffer = loadInstanceProc[typeof(lib.vkEndCommandBuffer)]("vkEndCommandBuffer")
  lib.vkCmdBindPipeline = loadInstanceProc[typeof(lib.vkCmdBindPipeline)]("vkCmdBindPipeline")
  lib.vkCmdBindDescriptorSets = loadInstanceProc[typeof(lib.vkCmdBindDescriptorSets)]("vkCmdBindDescriptorSets")
  lib.vkCmdDispatch = loadInstanceProc[typeof(lib.vkCmdDispatch)]("vkCmdDispatch")
  lib.vkCmdPushConstants = loadInstanceProc[typeof(lib.vkCmdPushConstants)]("vkCmdPushConstants")
  lib.vkCmdPipelineBarrier = loadInstanceProc[typeof(lib.vkCmdPipelineBarrier)]("vkCmdPipelineBarrier")
  lib.vkCreateFence = loadInstanceProc[typeof(lib.vkCreateFence)]("vkCreateFence")
  lib.vkCreateQueryPool = loadInstanceProc[typeof(lib.vkCreateQueryPool)]("vkCreateQueryPool")
  lib.vkDestroyQueryPool = loadInstanceProc[typeof(lib.vkDestroyQueryPool)]("vkDestroyQueryPool")
  lib.vkCmdResetQueryPool = loadInstanceProc[typeof(lib.vkCmdResetQueryPool)]("vkCmdResetQueryPool")
  lib.vkCmdWriteTimestamp = loadInstanceProc[typeof(lib.vkCmdWriteTimestamp)]("vkCmdWriteTimestamp")
  lib.vkGetQueryPoolResults = loadInstanceProc[typeof(lib.vkGetQueryPoolResults)]("vkGetQueryPoolResults")
  lib.vkWaitForFences = loadInstanceProc[typeof(lib.vkWaitForFences)]("vkWaitForFences")
  lib.vkResetFences = loadInstanceProc[typeof(lib.vkResetFences)]("vkResetFences")
  lib.vkQueueSubmit = loadInstanceProc[typeof(lib.vkQueueSubmit)]("vkQueueSubmit")
  lib.vkDeviceWaitIdle = loadInstanceProc[typeof(lib.vkDeviceWaitIdle)]("vkDeviceWaitIdle")
  lib.vkDestroyBuffer = loadInstanceProc[typeof(lib.vkDestroyBuffer)]("vkDestroyBuffer")
  lib.vkFreeMemory = loadInstanceProc[typeof(lib.vkFreeMemory)]("vkFreeMemory")
  lib.vkDestroyShaderModule = loadInstanceProc[typeof(lib.vkDestroyShaderModule)]("vkDestroyShaderModule")
  lib.vkDestroyPipeline = loadInstanceProc[typeof(lib.vkDestroyPipeline)]("vkDestroyPipeline")
  lib.vkDestroyPipelineLayout = loadInstanceProc[typeof(lib.vkDestroyPipelineLayout)]("vkDestroyPipelineLayout")
  lib.vkDestroyDescriptorSetLayout = loadInstanceProc[typeof(lib.vkDestroyDescriptorSetLayout)]("vkDestroyDescriptorSetLayout")
  lib.vkDestroyDescriptorPool = loadInstanceProc[typeof(lib.vkDestroyDescriptorPool)]("vkDestroyDescriptorPool")
  lib.vkDestroyFence = loadInstanceProc[typeof(lib.vkDestroyFence)]("vkDestroyFence")
  lib.vkFreeCommandBuffers = loadInstanceProc[typeof(lib.vkFreeCommandBuffers)]("vkFreeCommandBuffers")
  lib.vkFreeDescriptorSets = loadInstanceProc[typeof(lib.vkFreeDescriptorSets)]("vkFreeDescriptorSets")
  lib.vkCmdCopyBuffer = loadInstanceProc[typeof(lib.vkCmdCopyBuffer)]("vkCmdCopyBuffer")
  lib.vkResetCommandBuffer = loadInstanceProc[typeof(lib.vkResetCommandBuffer)]("vkResetCommandBuffer")

  # ---- 3. Prefer a discrete GPU; allow explicit device index selection. ----
  var devCount: uint32 = 0
  gpuCheck(lib.vkEnumeratePhysicalDevices(gInstance, addr devCount, nil) == VK_SUCCESS, "vkEnumeratePhysicalDevices (dem) that bai")
  gpuCheck(devCount > 0, "No Vulkan physical device found")
  var devices = newSeq[VkPhysicalDevice](devCount)
  gpuCheck(lib.vkEnumeratePhysicalDevices(gInstance, addr devCount, addr devices[0]) == VK_SUCCESS, "vkEnumeratePhysicalDevices (listing) failed")
  var selected = -1
  let requested = getEnv("BYBY_VULKAN_DEVICE")
  if requested.len > 0:
    selected = parseInt(requested)
    gpuCheck(selected >= 0 and selected < devices.len, "BYBY_VULKAN_DEVICE index out of range")
  var bestScore = -1
  for i, physical in devices:
    let caps = queryVulkanCaps(lib.vkGetInstanceProcAddr, gInstance, physical)
    let score = (if caps.properties.deviceType == 2: 100 else: 10) +
                (if caps.properties.vendorID == 0x10de: 10 else: 0)
    if (selected >= 0 and i == selected) or (selected < 0 and score > bestScore):
      gPhysicalDevice = physical
      gDeviceCaps = caps
      bestScore = score

  # ---- 4. Find a queue family with compute support ----
  var qCount: uint32 = 0
  lib.vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevice, addr qCount, nil)
  gpuCheck(qCount > 0, "The physical device has no queue family")
  var qProps = newSeq[VkQueueFamilyProperties](qCount)
  lib.vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevice, addr qCount, addr qProps[0])
  var foundQueue = false
  for i in 0 ..< int(qCount):
    if (qProps[i].queueFlags and VK_QUEUE_COMPUTE_BIT) != 0:
      gQueueFamilyIndex = uint32(i)
      gQueueTimestampValidBits = qProps[i].timestampValidBits
      foundQueue = true
      break
  gpuCheck(foundQueue, "No queue family supporting compute found")

  # ---- 5. Create the logical device + get the queue ----
  var prio: float32 = 1.0
  var qInfo = VkDeviceQueueCreateInfo(
    sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: nil, flags: 0,
    queueFamilyIndex: gQueueFamilyIndex,
    queueCount: 1,
    pQueuePriorities: addr prio
  )
  # Enable only extensions/features actually reported by the chosen device.
  var devExtNames: seq[cstring]
  if "VK_KHR_portability_subset" in gDeviceCaps.extensions:
    devExtNames.add cstring("VK_KHR_portability_subset")
  # Zero-copy model mmaps only where the GPU shares system memory (iGPU); a
  # dGPU would read every weight over PCIe on each token. Added before the
  # cooperative-matrix extension, which the retry below removes from the end.
  gHostImportEnabled = gDeviceCaps.hostImportAlignment > 0 and
    gDeviceCaps.properties.deviceType == 1'u32 and
    getEnv("BYBY_VULKAN_MMAP_IMPORT", "1") != "0"
  if gHostImportEnabled: devExtNames.add cstring("VK_EXT_external_memory_host")
  gFp8CoopEnabled = gDeviceCaps.float8E4M3CooperativeMatrix and
    getEnv("BYBY_VULKAN_TENSOR_CORES", "1") != "0" and findExe("glslc").len > 0
  gCoopEnabled = (gDeviceCaps.cooperativeMatrix or gFp8CoopEnabled) and
    getEnv("BYBY_VULKAN_TENSOR_CORES", "1") != "0" and findExe("glslc").len > 0
  if gCoopEnabled: devExtNames.add cstring("VK_KHR_cooperative_matrix")
  if gFp8CoopEnabled: devExtNames.add cstring("VK_EXT_shader_float8")
  var float8Features = VkFloat8Features(sType: 1000567000'u32,
    shaderFloat8: uint32(gFp8CoopEnabled),
    shaderFloat8CooperativeMatrix: uint32(gFp8CoopEnabled))
  var coopFeatures = VkCoopFeatures(sType: CoopFeaturesType,
    cooperativeMatrix: uint32(gCoopEnabled),
    pNext: (if gFp8CoopEnabled: addr float8Features else: nil))
  var memoryFeatures = VkMemoryModelFeatures(sType: MemoryModelType,
    vulkanMemoryModel: uint32(gCoopEnabled),
    pNext: (if gCoopEnabled: addr coopFeatures else: nil))
  var storageFeatures = VkStorage8Features(sType: Storage8Type,
    storageBuffer8BitAccess: uint32(gDeviceCaps.storage8), pNext: addr memoryFeatures)
  var floatFeatures = VkFloat16Int8Features(sType: Float16Int8Type,
    shaderFloat16: uint32(gDeviceCaps.float16), shaderInt8: uint32(gDeviceCaps.int8),
    pNext: addr storageFeatures)
  var devInfo = VkDeviceCreateInfo(
    sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: addr floatFeatures, flags: 0,
    queueCreateInfoCount: 1, pQueueCreateInfos: addr qInfo,
    enabledExtensionCount: uint32(devExtNames.len),
    ppEnabledExtensionNames: (if devExtNames.len > 0: addr devExtNames[0] else: nil))
  var dev: VkDevice
  var rDev = lib.vkCreateDevice(gPhysicalDevice, addr devInfo, nil, addr dev)
  if rDev != VK_SUCCESS and gCoopEnabled:
    # A driver rejecting optional MMA features still gets the ordinary path.
    gCoopEnabled = false
    gFp8CoopEnabled = false
    devExtNames.setLen(devExtNames.len - 1)
    memoryFeatures.vulkanMemoryModel = 0
    memoryFeatures.pNext = nil
    devInfo.enabledExtensionCount = uint32(devExtNames.len)
    devInfo.ppEnabledExtensionNames = if devExtNames.len > 0: addr devExtNames[0] else: nil
    rDev = lib.vkCreateDevice(gPhysicalDevice, addr devInfo, nil, addr dev)
  gpuCheck(rDev == VK_SUCCESS, "vkCreateDevice failed, VkResult=" & $rDev)
  gDevice = dev
  # BYBY_VULKAN_TENSOR_CORES: 1 (default) uses them and, on a dGPU, never lets
  # a prefill GEMM fall back to a scalar kernel while a tensor-core kernel
  # passes parity; force does the same on any device that has them; tune
  # times them against the scalar kernels (the fastest wins); 0 turns them off.
  let tensorCoreMode = getEnv("BYBY_VULKAN_TENSOR_CORES", "1")
  gCoopRequired = gCoopEnabled and tensorCoreMode != "tune" and
    (tensorCoreMode == "force" or gDeviceCaps.properties.deviceType == 2'u32)
  if gHostImportEnabled:
    vkGetMemoryHostPointerPropertiesEXT = cast[typeof(vkGetMemoryHostPointerPropertiesEXT)](
      lib.vkGetInstanceProcAddr(gInstance, "vkGetMemoryHostPointerPropertiesEXT"))
    gHostImportEnabled = vkGetMemoryHostPointerPropertiesEXT != nil
  lib.vkGetDeviceQueue(gDevice, gQueueFamilyIndex, 0, addr gQueue)

  # ---- 6. Command pool (reused for every later compute command, like
  # gSessionEncoder on Metal) ----
  var poolInfo = VkCommandPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: nil, flags: 0, queueFamilyIndex: gQueueFamilyIndex
  )
  gpuCheck(lib.vkCreateCommandPool(gDevice, addr poolInfo, nil, addr gCommandPool) == VK_SUCCESS, "vkCreateCommandPool that bai")

  gInitialized = true
  let weightMemory = if useDeviceLocalWeights(gDeviceCaps.properties.deviceType,
      getEnv("BYBY_VULKAN_DEVICE_LOCAL", "auto")): "device-local/VRAM"
    else: "host-visible/shared"
  # BYBY_VULKAN_SCRATCH: auto (VRAM on a dGPU), device (always), host (never).
  let scratchMode = getEnv("BYBY_VULKAN_SCRATCH", "auto")
  gScratchDeviceLocal = scratchMode == "device" or
    (scratchMode == "auto" and gDeviceCaps.properties.deviceType == 2'u32)
  stderr.writeLine "[vulkan] device=" & gDeviceCaps.deviceName &
    " type=" & deviceClass(gDeviceCaps.properties.deviceType) &
    " residentWeights=" & weightMemory &
    " scratch=" & (if gScratchDeviceLocal: "device-local/VRAM" else: "host-visible") &
    " subgroup=" & $gDeviceCaps.subgroupSize & " maxThreads=" & $gDeviceCaps.maxThreads &
    " sharedBytes=" & $gDeviceCaps.maxSharedBytes &
    " mmapImport=" & (if gHostImportEnabled: $gDeviceCaps.hostImportAlignment else: "off") &
    " shaderFloat16=" & $gDeviceCaps.float16 &
    " matrix=" & (if gFp8CoopEnabled: "KHR_cooperative_matrix FP8-E4M3/FP32 + FP16/FP32"
      elif gCoopEnabled: "KHR_cooperative_matrix FP16/FP32" else: "scalar") &
    (if gCoopRequired: " prefill=tensor-cores-only" else: "") &
    " units=" & (if gDeviceCaps.computeUnits > 0:
        $gDeviceCaps.computeUnits & " (" & gDeviceCaps.computeUnitSource &
        ", " & $gDeviceCaps.warpsPerComputeUnit & " subgroups/unit)"
      else: "unknown")
  stderr.writeLine "[vulkan] Initialised: instance+device+queue+commandPool ready (queueFamily=" & $gQueueFamilyIndex & ")"

proc vulkanDeviceCaps*(): VulkanCaps =
  ensureInit()
  gDeviceCaps

proc vulkanTensorCoreAvailable*(): bool =
  ensureInit()
  gCoopEnabled

proc vulkanTensorCoreRequired*(): bool =
  ## True when every prefill GEMM (M > 1) must run on the tensor cores.
  ensureInit()
  gCoopRequired

proc coopMinRows(): int =
  ## Smallest M a cooperative GEMM takes. Below 8 rows its 16x16 tiles are
  ## mostly idle, which only loses to the scalar kernels where those may run.
  if gCoopRequired: 2 else: 8

proc findMemoryType(typeFilter: uint32, properties: uint32): uint32 =
  var memProps: VkPhysicalDeviceMemoryProperties
  lib.vkGetPhysicalDeviceMemoryProperties(gPhysicalDevice, addr memProps)
  # Plain VRAM requests skip the small host-visible BAR heap (256 MB without
  # resizable BAR) when the driver lists it before ordinary device memory.
  if properties == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
    for i in 0 ..< int(memProps.memoryTypeCount):
      let flags = memProps.memoryTypes[i].propertyFlags
      if (typeFilter and (1'u32 shl uint32(i))) != 0 and
         (flags and properties) == properties and
         (flags and VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0:
        return uint32(i)
  for i in 0 ..< int(memProps.memoryTypeCount):
    if (typeFilter and (1'u32 shl uint32(i))) != 0 and
       (memProps.memoryTypes[i].propertyFlags and properties) == properties:
      return uint32(i)
  gpuCheck(false, "No suitable memory type found (typeFilter=" & $typeFilter & ", needed properties=" & $properties & ")")

proc createHostVisibleBuffer(sizeBytes: uint64, usage: uint32; memoryFlags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT): tuple[buf: VkBuffer, mem: VkDeviceMemory] =
  ## Buffer host-visible + host-coherent.
  ## MoltenVK/Vulkan reject size==0 (VK_ERROR_OUT_OF_DEVICE_MEMORY "size 0 bytes").
  ## Use a 4-byte placeholder when caller asks for empty; sizeBytes field on
  ## VulkanBufferHandle still records the logical 0 so readback stays empty.
  var allocSize = sizeBytes
  if allocSize == 0:
    allocSize = 4
  var bufInfo = VkBufferCreateInfo(
    sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: nil, flags: 0,
    size: allocSize, usage: usage or VK_BUFFER_USAGE_TRANSFER_SRC_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    sharingMode: VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: 0, pQueueFamilyIndices: nil
  )
  var buf: VkBuffer
  let cr = lib.vkCreateBuffer(gDevice, addr bufInfo, nil, addr buf)
  gpuCheckVk(cr, "vkCreateBuffer that bai (sizeBytes=" & $sizeBytes & ", allocSize=" & $allocSize & ")")
  var req: VkMemoryRequirements
  lib.vkGetBufferMemoryRequirements(gDevice, buf, addr req)
  if req.size == 0:
    raise newException(CatchableError,
      "vkGetBufferMemoryRequirements returned size 0 (requested sizeBytes=" & $sizeBytes & ")")
  let memTypeIdx = findMemoryType(req.memoryTypeBits, memoryFlags)
  var allocInfo = VkMemoryAllocateInfo(
    sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, pNext: nil,
    allocationSize: req.size, memoryTypeIndex: memTypeIdx
  )
  var mem: VkDeviceMemory
  let ar = lib.vkAllocateMemory(gDevice, addr allocInfo, nil, addr mem)
  gpuCheckVk(ar, "vkAllocateMemory that bai (req.size=" & $req.size & ", sizeBytes=" & $sizeBytes & ")")
  gpuCheck(lib.vkBindBufferMemory(gDevice, buf, mem, 0) == VK_SUCCESS, "vkBindBufferMemory that bai")
  result = (buf, mem)

proc uploadToBuffer(mem: VkDeviceMemory, data: pointer, sizeBytes: uint64) =
  var mapped: pointer
  gpuCheck(lib.vkMapMemory(gDevice, mem, 0, sizeBytes, 0, addr mapped) == VK_SUCCESS, "vkMapMemory (upload) that bai")
  copyMem(mapped, data, int(sizeBytes))
  lib.vkUnmapMemory(gDevice, mem)

proc readFromBuffer(mem: VkDeviceMemory, data: pointer, sizeBytes: uint64) =
  var mapped: pointer
  gpuCheck(lib.vkMapMemory(gDevice, mem, 0, sizeBytes, 0, addr mapped) == VK_SUCCESS, "vkMapMemory (read) that bai")
  copyMem(data, mapped, int(sizeBytes))
  lib.vkUnmapMemory(gDevice, mem)

# ---------------------------------------------------------------------
# MINIMAL TEST: vecAdd(a,b) = c, used to CONFIRM the whole pipeline
# instance->device->buffer->shader->pipeline->dispatch->readback works
# CORRECTLY on real hardware before writing any more complex Q4 kernel.
# The GLSL source is kernels/vecadd.comp - it MUST be compiled to
# SPIR-V with glslc/glslangValidator (included in the Vulkan SDK) and
# the .spv placed in the SAME kernels/ directory as vecadd.spv:
#
#     glslc kernels/vecadd.comp -o kernels/vecadd.spv
#
# or:
#
#     glslangValidator -V kernels/vecadd.comp -o kernels/vecadd.spv
# ---------------------------------------------------------------------
proc vulkanVecAddTest*(a, b: seq[float32]): seq[float32] =
  ensureInit()
  gpuCheck(a.len == b.len, "vulkanVecAddTest: a and b must have the same length")
  let n = a.len
  let bytes = uint64(n * sizeof(float32))

  let (bufA, memA) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufB, memB) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufC, memC) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  var aVar = a
  var bVar = b
  uploadToBuffer(memA, addr aVar[0], bytes)
  uploadToBuffer(memB, addr bVar[0], bytes)

  # ---- Load the precompiled SPIR-V shader (read at Nim COMPILE time, like
  # Metal/CUDA staticRead - NOT read from disk at runtime) ----
  let spirvPath = currentSourcePath().parentDir() / "kernels" / "vecadd.spv"
  gpuCheck(fileExists(spirvPath),
    "Missing " & spirvPath & " - compile it first: glslc kernels/vecadd.comp -o kernels/vecadd.spv")
  let spirvBytes = readFile(spirvPath)
  gpuCheck(spirvBytes.len mod 4 == 0, "Invalid SPIR-V file (length not a multiple of 4 bytes)")

  var shaderInfo = VkShaderModuleCreateInfo(
    sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext: nil, flags: 0,
    codeSize: uint(spirvBytes.len),
    pCode: cast[ptr uint32](unsafeAddr spirvBytes[0])
  )
  var shaderModule: VkShaderModule
  gpuCheck(lib.vkCreateShaderModule(gDevice, addr shaderInfo, nil, addr shaderModule) == VK_SUCCESS, "vkCreateShaderModule that bai")

  # ---- Descriptor set layout: 3 binding (a,b,c) storage buffer ----
  var bindings: array[3, VkDescriptorSetLayoutBinding]
  for i in 0 ..< 3:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i), descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1, stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers: nil
    )
  var setLayoutInfo = VkDescriptorSetLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    bindingCount: 3, pBindings: addr bindings[0]
  )
  var setLayout: VkDescriptorSetLayout
  gpuCheck(lib.vkCreateDescriptorSetLayout(gDevice, addr setLayoutInfo, nil, addr setLayout) == VK_SUCCESS, "vkCreateDescriptorSetLayout that bai")

  var pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    setLayoutCount: 1, pSetLayouts: addr setLayout,
    pushConstantRangeCount: 0, pPushConstantRanges: nil
  )
  var pipelineLayout: VkPipelineLayout
  gpuCheck(lib.vkCreatePipelineLayout(gDevice, addr pipelineLayoutInfo, nil, addr pipelineLayout) == VK_SUCCESS, "vkCreatePipelineLayout that bai")

  var stageInfo = VkPipelineShaderStageCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, pNext: nil, flags: 0,
    stage: VK_SHADER_STAGE_COMPUTE_BIT, module: shaderModule, pName: "main",
    pSpecializationInfo: nil
  )
  var pipelineInfo = VkComputePipelineCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, pNext: nil, flags: 0,
    stage: stageInfo, layout: pipelineLayout,
    basePipelineHandle: nil, basePipelineIndex: -1
  )
  var pipeline: VkPipeline
  gpuCheck(lib.vkCreateComputePipelines(gDevice, nil, 1, addr pipelineInfo, nil, addr pipeline) == VK_SUCCESS, "vkCreateComputePipelines that bai")

  # ---- Descriptor pool + set, pointing the 3 buffers at the 3 bindings ----
  var poolSize = VkDescriptorPoolSize(descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 3)
  var descPoolInfo = VkDescriptorPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, pNext: nil, flags: 0,
    maxSets: 1, poolSizeCount: 1, pPoolSizes: addr poolSize
  )
  var descPool: VkDescriptorPool
  gpuCheck(lib.vkCreateDescriptorPool(gDevice, addr descPoolInfo, nil, addr descPool) == VK_SUCCESS, "vkCreateDescriptorPool that bai")

  var descAllocInfo = VkDescriptorSetAllocateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pNext: nil,
    descriptorPool: descPool, descriptorSetCount: 1, pSetLayouts: addr setLayout
  )
  var descSet: VkDescriptorSet
  gpuCheck(lib.vkAllocateDescriptorSets(gDevice, addr descAllocInfo, addr descSet) == VK_SUCCESS, "vkAllocateDescriptorSets that bai")

  var bufInfos = [
    VkDescriptorBufferInfo(buffer: bufA, offset: 0, range: bytes),
    VkDescriptorBufferInfo(buffer: bufB, offset: 0, range: bytes),
    VkDescriptorBufferInfo(buffer: bufC, offset: 0, range: bytes)
  ]
  var writes: array[3, VkWriteDescriptorSet]
  for i in 0 ..< 3:
    writes[i] = VkWriteDescriptorSet(
      sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext: nil,
      dstSet: descSet, dstBinding: uint32(i), dstArrayElement: 0,
      descriptorCount: 1, descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      pImageInfo: nil, pBufferInfo: addr bufInfos[i], pTexelBufferView: nil
    )
  lib.vkUpdateDescriptorSets(gDevice, 3, addr writes[0], 0, nil)

  # ---- Command buffer: bind pipeline + descriptor set, dispatch, submit ----
  var cmdAllocInfo = VkCommandBufferAllocateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext: nil,
    commandPool: gCommandPool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount: 1
  )
  var cmdBuf: VkCommandBuffer
  gpuCheck(lib.vkAllocateCommandBuffers(gDevice, addr cmdAllocInfo, addr cmdBuf) == VK_SUCCESS, "vkAllocateCommandBuffers that bai")

  var beginInfo = VkCommandBufferBeginInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: nil,
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo: nil
  )
  gpuCheck(lib.vkBeginCommandBuffer(cmdBuf, addr beginInfo) == VK_SUCCESS, "vkBeginCommandBuffer that bai")
  lib.vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
  var descSetVar = descSet
  lib.vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, addr descSetVar, 0, nil)
  # local_size_x = 64 in the shader (see vecadd.comp) - enough groups to cover n elements.
  let groupCount = uint32((n + 63) div 64)
  lib.vkCmdDispatch(cmdBuf, groupCount, 1, 1)
  gpuCheck(lib.vkEndCommandBuffer(cmdBuf) == VK_SUCCESS, "vkEndCommandBuffer that bai")

  var fenceInfo = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: nil, flags: 0)
  var fence: VkFence
  gpuCheck(lib.vkCreateFence(gDevice, addr fenceInfo, nil, addr fence) == VK_SUCCESS, "vkCreateFence that bai")

  var cmdBufVar = cmdBuf
  var submitInfo = VkSubmitInfo(
    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext: nil,
    waitSemaphoreCount: 0, pWaitSemaphores: nil, pWaitDstStageMask: nil,
    commandBufferCount: 1, pCommandBuffers: addr cmdBufVar,
    signalSemaphoreCount: 0, pSignalSemaphores: nil
  )
  gpuCheck(lib.vkQueueSubmit(gQueue, 1, addr submitInfo, fence) == VK_SUCCESS, "vkQueueSubmit that bai")
  gpuCheck(lib.vkWaitForFences(gDevice, 1, addr fence, 1, high(uint64)) == VK_SUCCESS, "vkWaitForFences that bai/timeout")

  result = newSeq[float32](n)
  readFromBuffer(memC, addr result[0], bytes)
  stderr.writeLine "[vulkan] vecAddTest xong, n=" & $n

# ---------------------------------------------------------------------
# REAL Q4 MATMUL KERNEL - an exact mirror of the Metal/CUDA/OpenCL algorithm.
# The "immediate" variant (no session) - each call uploads/dispatches/reads back itself,
# like metal_matmul_q4 (the non-session variant), NOT yet the session API that batches
# many commands like metal_matmul_q4_enc - that is the next step (step 2/3),
# once this immediate variant RUNS CORRECTLY.
#
# The shader MUST be compiled first:
#     glslc kernels/matmul_q4.comp -o kernels/matmul_q4.spv
# ---------------------------------------------------------------------
const MMQ4_NDST = 2  # CHANGED: 8 -> 2 (mirrors the MEASURED Metal result -
  # matmul_q4_kernel_ndst2_paired, fully explained in
  # vecop_matmul.metal and kernels/matmul_q4.comp). MUST stay in sync with BOTH places
  # using this value: #define MMQ4_NDST in kernels/matmul_q4.comp (the
  # two do not update each other - a GLSL macro and a Nim const are two
  # separate compile worlds). If they differ, rowBlocks are WRONG -> WRONG
  # results/crash. NOT YET RUN ON A REAL VULKAN GPU.

proc vulkanMatmulQ4*(a: seq[float32], wq: seq[uint8], scales, zeros: seq[float32],
                      M, K, N, groupSize, nGroupsPerRow: int): seq[float32] =
  ensureInit()
  let bytesA = uint64(a.len * sizeof(float32))
  let bytesWq = uint64(wq.len * sizeof(uint8))
  let bytesScales = uint64(scales.len * sizeof(float32))
  let bytesZeros = uint64(zeros.len * sizeof(float32))
  let bytesC = uint64(M * N * sizeof(float32))

  let (bufA, memA) = createHostVisibleBuffer(bytesA, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufWq, memWq) = createHostVisibleBuffer(bytesWq, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufScales, memScales) = createHostVisibleBuffer(bytesScales, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufZeros, memZeros) = createHostVisibleBuffer(bytesZeros, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufC, memC) = createHostVisibleBuffer(bytesC, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)

  var aVar = a; var wqVar = wq; var scalesVar = scales; var zerosVar = zeros
  uploadToBuffer(memA, addr aVar[0], bytesA)
  uploadToBuffer(memWq, addr wqVar[0], bytesWq)
  uploadToBuffer(memScales, addr scalesVar[0], bytesScales)
  uploadToBuffer(memZeros, addr zerosVar[0], bytesZeros)

  let spirvPath = currentSourcePath().parentDir() / "kernels" / "matmul_q4.spv"
  gpuCheck(fileExists(spirvPath),
    "Missing " & spirvPath & " - compile it first: glslc kernels/matmul_q4.comp -o kernels/matmul_q4.spv")
  let spirvBytes = readFile(spirvPath)
  gpuCheck(spirvBytes.len mod 4 == 0, "Invalid SPIR-V file")

  var shaderInfo = VkShaderModuleCreateInfo(
    sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext: nil, flags: 0,
    codeSize: uint(spirvBytes.len), pCode: cast[ptr uint32](unsafeAddr spirvBytes[0])
  )
  var shaderModule: VkShaderModule
  gpuCheck(lib.vkCreateShaderModule(gDevice, addr shaderInfo, nil, addr shaderModule) == VK_SUCCESS, "vkCreateShaderModule that bai")

  var bindings: array[5, VkDescriptorSetLayoutBinding]
  for i in 0 ..< 5:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i), descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1, stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers: nil
    )
  var setLayoutInfo = VkDescriptorSetLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    bindingCount: 5, pBindings: addr bindings[0]
  )
  var setLayout: VkDescriptorSetLayout
  gpuCheck(lib.vkCreateDescriptorSetLayout(gDevice, addr setLayoutInfo, nil, addr setLayout) == VK_SUCCESS, "vkCreateDescriptorSetLayout that bai")

  # push constants: 6 int32 (M,K,N,groupSize,nGroupsPerRow,wqOff) = 24 bytes, matching
  # struct PushConsts in matmul_q4.comp - A WRONG ORDER/TYPE HERE MAKES THE SHADER READ
  # WRONG VALUES (no compile error, silently wrong results).
  var pcRange = VkPushConstantRange(stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, offset: 0, size: 24)
  var pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    setLayoutCount: 1, pSetLayouts: addr setLayout,
    pushConstantRangeCount: 1, pPushConstantRanges: addr pcRange
  )
  var pipelineLayout: VkPipelineLayout
  gpuCheck(lib.vkCreatePipelineLayout(gDevice, addr pipelineLayoutInfo, nil, addr pipelineLayout) == VK_SUCCESS, "vkCreatePipelineLayout that bai")

  var stageInfo = VkPipelineShaderStageCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, pNext: nil, flags: 0,
    stage: VK_SHADER_STAGE_COMPUTE_BIT, module: shaderModule, pName: "main", pSpecializationInfo: nil
  )
  var pipelineInfo = VkComputePipelineCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, pNext: nil, flags: 0,
    stage: stageInfo, layout: pipelineLayout, basePipelineHandle: nil, basePipelineIndex: -1
  )
  var pipeline: VkPipeline
  gpuCheck(lib.vkCreateComputePipelines(gDevice, nil, 1, addr pipelineInfo, nil, addr pipeline) == VK_SUCCESS, "vkCreateComputePipelines that bai")

  var poolSize = VkDescriptorPoolSize(descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 5)
  var descPoolInfo = VkDescriptorPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, pNext: nil, flags: 0,
    maxSets: 1, poolSizeCount: 1, pPoolSizes: addr poolSize
  )
  var descPool: VkDescriptorPool
  gpuCheck(lib.vkCreateDescriptorPool(gDevice, addr descPoolInfo, nil, addr descPool) == VK_SUCCESS, "vkCreateDescriptorPool that bai")

  var descAllocInfo = VkDescriptorSetAllocateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pNext: nil,
    descriptorPool: descPool, descriptorSetCount: 1, pSetLayouts: addr setLayout
  )
  var descSet: VkDescriptorSet
  gpuCheck(lib.vkAllocateDescriptorSets(gDevice, addr descAllocInfo, addr descSet) == VK_SUCCESS, "vkAllocateDescriptorSets that bai")

  var bufInfos = [
    VkDescriptorBufferInfo(buffer: bufA, offset: 0, range: bytesA),
    VkDescriptorBufferInfo(buffer: bufWq, offset: 0, range: bytesWq),
    VkDescriptorBufferInfo(buffer: bufScales, offset: 0, range: bytesScales),
    VkDescriptorBufferInfo(buffer: bufZeros, offset: 0, range: bytesZeros),
    VkDescriptorBufferInfo(buffer: bufC, offset: 0, range: bytesC)
  ]
  var writes: array[5, VkWriteDescriptorSet]
  for i in 0 ..< 5:
    writes[i] = VkWriteDescriptorSet(
      sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext: nil,
      dstSet: descSet, dstBinding: uint32(i), dstArrayElement: 0,
      descriptorCount: 1, descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      pImageInfo: nil, pBufferInfo: addr bufInfos[i], pTexelBufferView: nil
    )
  lib.vkUpdateDescriptorSets(gDevice, 5, addr writes[0], 0, nil)

  var cmdAllocInfo = VkCommandBufferAllocateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext: nil,
    commandPool: gCommandPool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount: 1
  )
  var cmdBuf: VkCommandBuffer
  gpuCheck(lib.vkAllocateCommandBuffers(gDevice, addr cmdAllocInfo, addr cmdBuf) == VK_SUCCESS, "vkAllocateCommandBuffers that bai")

  var beginInfo = VkCommandBufferBeginInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: nil,
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo: nil
  )
  gpuCheck(lib.vkBeginCommandBuffer(cmdBuf, addr beginInfo) == VK_SUCCESS, "vkBeginCommandBuffer that bai")
  lib.vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
  var descSetVar = descSet
  lib.vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, addr descSetVar, 0, nil)

  # push constants: int32s MUST be in the order of struct PushConsts in the .comp
  var pcData = [int32(M), int32(K), int32(N), int32(groupSize), int32(nGroupsPerRow), 0'i32]
  lib.vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, addr pcData[0])

  let rowBlocks = uint32((N + MMQ4_NDST - 1) div MMQ4_NDST)
  lib.vkCmdDispatch(cmdBuf, rowBlocks, uint32(M), 1)
  gpuCheck(lib.vkEndCommandBuffer(cmdBuf) == VK_SUCCESS, "vkEndCommandBuffer that bai")

  var fenceInfo = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: nil, flags: 0)
  var fence: VkFence
  gpuCheck(lib.vkCreateFence(gDevice, addr fenceInfo, nil, addr fence) == VK_SUCCESS, "vkCreateFence that bai")

  var cmdBufVar = cmdBuf
  var submitInfo = VkSubmitInfo(
    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext: nil,
    waitSemaphoreCount: 0, pWaitSemaphores: nil, pWaitDstStageMask: nil,
    commandBufferCount: 1, pCommandBuffers: addr cmdBufVar,
    signalSemaphoreCount: 0, pSignalSemaphores: nil
  )
  gpuCheck(lib.vkQueueSubmit(gQueue, 1, addr submitInfo, fence) == VK_SUCCESS, "vkQueueSubmit that bai")
  gpuCheck(lib.vkWaitForFences(gDevice, 1, addr fence, 1, high(uint64)) == VK_SUCCESS, "vkWaitForFences that bai/timeout")

  result = newSeq[float32](M * N)
  readFromBuffer(memC, addr result[0], bytesC)

# =======================================================================
# FULL SESSION API - mirrors the Metal session architecture (gSessionEncoder)
# but uses a VkCommandBuffer as the "session" unit: sessionBegin() opens +
# starts recording a command buffer, the *Enc() functions RECORD commands into it
# (NOT submitted yet), sessionEnd() ends recording + submits + WAITS for completion
# (fence) - the SAME lesson learned on Metal: sessionRead() is only safe AFTER
# sessionEnd() has completed, buffers must NOT be read in the middle of an
# uncommitted session.
# =======================================================================
import std/tables

type
  PipelineEntry = object
    pipeline: VkPipeline
    layout: VkPipelineLayout
    setLayout: VkDescriptorSetLayout
    numBindings: int
    pushConstSize: uint32

var gPipelineCache = initTable[string, PipelineEntry]()
var gVkPipelineCacheObject: VkPipelineCache
var gVkPipelineCacheLoaded = false
var gVkPipelineCachePath = ""
var gSessionCmdBuf: VkCommandBuffer
var gSessionActive = false
var gSessionCleanupPools: seq[VkDescriptorPool] = @[]
# Internal-only session buffers (multiBuf dummy slots). Freed in
# vulkanSessionEnd. MUST NOT track buffers returned to callers that still
# read/free after End (vulkanMatmul/VecOp/Activation: AllocScratch → End →
# Read → Free) — that caused SIGSEGV use-after-free on first matmul.
var gSessionInternalBufs: seq[VulkanBufferHandle] = @[]
## A buffer freed with sessionFree() WHILE a session is open: the command buffer being recorded still references it,
## destroying it right away => vkQueueSubmit uses a destroyed VkBuffer => SIGSEGV (Metal is ref-counted, so fine there).
## Destruction is deferred until vulkanSessionEnd has waited on the fence.
var gSessionDeferredFree: seq[VulkanBufferHandle] = @[]
var gVkLastError: string = ""   ## error description of the last failed vulkanSessionEnd

proc vkPipelineCachePath(): string =
  var uuid = ""
  for b in gDeviceCaps.properties.pipelineCacheUUID: uuid.add(toHex(int(b), 2))
  let override = getEnv("BYBY_VULKAN_JIT_DIR")
  let cacheDir = if override.len > 0: override else: getTempDir() / "nimformers-vulkan-jit"
  createDir(cacheDir)
  cacheDir / ("pipeline-cache-vk" & uuid & "-drv" &
    $gDeviceCaps.properties.driverVersion & ".bin")

proc vkSavePipelineCache() {.noconv.} =
  if gVkPipelineCacheObject == nil or gVkPipelineCachePath.len == 0: return
  var size: uint
  if lib.vkGetPipelineCacheData(gDevice, gVkPipelineCacheObject, addr size, nil) != VK_SUCCESS or size == 0:
    return
  if size > 256'u * 1024'u * 1024'u: return
  var data = newString(int(size))
  if lib.vkGetPipelineCacheData(gDevice, gVkPipelineCacheObject, addr size, addr data[0]) != VK_SUCCESS:
    return
  data.setLen(int(size))
  try:
    writeFile(gVkPipelineCachePath, data)
  except CatchableError:
    discard

proc ensureVkPipelineCache() =
  if gVkPipelineCacheLoaded: return
  gVkPipelineCacheLoaded = true
  gVkPipelineCachePath = vkPipelineCachePath()
  var initial = ""
  if fileExists(gVkPipelineCachePath):
    try:
      let size = getFileSize(gVkPipelineCachePath)
      if size > 0 and size <= 256'i64 * 1024 * 1024:
        initial = readFile(gVkPipelineCachePath)
    except CatchableError:
      initial.setLen(0)
  var info = VkPipelineCacheCreateInfo(sType: VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    initialDataSize: uint(initial.len),
    pInitialData: (if initial.len > 0: cast[pointer](unsafeAddr initial[0]) else: nil))
  if lib.vkCreatePipelineCache(gDevice, addr info, nil, addr gVkPipelineCacheObject) != VK_SUCCESS and
      initial.len > 0:
    initial.setLen(0)
    info.initialDataSize = 0
    info.pInitialData = nil
    discard lib.vkCreatePipelineCache(gDevice, addr info, nil, addr gVkPipelineCacheObject)
  if gVkPipelineCacheObject != nil:
    addQuitProc(vkSavePipelineCache)
    if getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") != "0":
      stderr.writeLine "[vulkan] pipeline cache " & (if initial.len > 0: "loaded " & $initial.len & " bytes" else: "created empty")

# ---- Buffer pool (dGPU scratch) ----
# A decode step allocates the same few hundred scratch buffers every token.
# vkAllocateMemory/vkFreeMemory per buffer costs far more than the kernels on
# a dGPU, so freed buffers wait here (keyed by kind and rounded size) until
# the next allocation of that size. Buffers freed inside a session reach the
# pool only after its fence, like gSessionDeferredFree.
type PoolKind = enum pkDevice, pkUpload, pkReadback

var gBufferPool = initTable[(PoolKind, uint64), seq[VulkanBufferHandle]]()
var gPoolBytes = 0'u64
proc poolLimitMb(): int =
  try: max(0, parseInt(getEnv("BYBY_VULKAN_POOL_MB", "2048")))
  except ValueError: 2048
let gPoolLimitBytes = uint64(poolLimitMb()) * 1024 * 1024
## Upload staging recorded into the open session; released after its fence.
var gSessionStaging: seq[VulkanBufferHandle] = @[]
## Transfer hazards since the last transfer barrier: copies recorded back to
## back share one barrier unless one touches a buffer another copy wrote/read.
var gXferWritten = initHashSet[VkBuffer]()
var gXferRead = initHashSet[VkBuffer]()
var gXferPending = false
## Optional timestamps used by BYBY_DECODE_PROFILE on Vulkan. Off during
## ordinary inference; enabled only when BYBY_VULKAN_TIMESTAMPS=1.
var gVkSessionQueryPool: VkQueryPool
var gVkSessionTimingActive = false
var gVkStatsEnabled = false
var gVkGpuSecondsTotal = 0.0
var gVkEndCommandSeconds, gVkFenceCreateSeconds = 0.0
var gVkQueueSubmitSeconds, gVkFenceWaitSeconds, gVkCleanupSeconds = 0.0
var gVkSessionEnds, gVkQueueSubmits = 0
var gVkDispatches, gVkBarriers, gVkCopies, gVkDescriptorSets = 0

proc vulkanSessionGpuSeconds*(): float = gVkGpuSecondsTotal

proc vulkanSessionTimingReset*() =
  gVkStatsEnabled = true
  gVkGpuSecondsTotal = 0.0
  gVkEndCommandSeconds = 0.0
  gVkFenceCreateSeconds = 0.0
  gVkQueueSubmitSeconds = 0.0
  gVkFenceWaitSeconds = 0.0
  gVkCleanupSeconds = 0.0
  gVkSessionEnds = 0
  gVkQueueSubmits = 0
  gVkDispatches = 0
  gVkBarriers = 0
  gVkCopies = 0
  gVkDescriptorSets = 0

proc vulkanSessionTimingReport*(steps: int): string =
  if not gVkStatsEnabled: return ""
  let n = float(max(1, steps))
  result = "sessions=" & formatFloat(float(gVkSessionEnds) / n, ffDecimal, 1) &
    ", dispatches=" & formatFloat(float(gVkDispatches) / n, ffDecimal, 1) &
    ", descriptorSets=" & formatFloat(float(gVkDescriptorSets) / n, ffDecimal, 1) &
    ", copies=" & formatFloat(float(gVkCopies) / n, ffDecimal, 1) &
    ", barriers=" & formatFloat(float(gVkBarriers) / n, ffDecimal, 1) &
    ", gpu=" & formatFloat(gVkGpuSecondsTotal * 1000 / n, ffDecimal, 2) & " ms/token" &
    ", vkEnd=" & formatFloat(gVkEndCommandSeconds * 1000 / n, ffDecimal, 2) & " ms/token" &
    ", fenceCreate=" & formatFloat(gVkFenceCreateSeconds * 1000 / n, ffDecimal, 2) & " ms/token" &
    ", queueSubmit=" & formatFloat(gVkQueueSubmitSeconds * 1000 / n, ffDecimal, 2) & " ms/token" &
    ", fenceWait=" & formatFloat(gVkFenceWaitSeconds * 1000 / n, ffDecimal, 2) & " ms/token" &
    ", cleanup=" & formatFloat(gVkCleanupSeconds * 1000 / n, ffDecimal, 2) & " ms/token"
  if gQueueTimestampValidBits == 0 or
      gDeviceCaps.properties.limits.timestampComputeAndGraphics == 0 or
      gDeviceCaps.properties.limits.timestampPeriod <= 0:
    result &= ", device timestamps unavailable"
  elif getEnv("BYBY_VULKAN_TIMESTAMPS", "0") != "1":
    result &= ", device timestamps disabled"
  gVkStatsEnabled = false

proc poolBucket(bytes: uint64): uint64 =
  ## Power-of-two buckets up to 1 MiB, then whole MiB: prefill-sized buffers
  ## of nearby lengths reuse each other.
  result = 256
  if bytes <= 1024 * 1024:
    while result < bytes: result = result shl 1
  else:
    result = (bytes + 1024 * 1024 - 1) and not (1024 * 1024 - 1'u64)

proc destroyBuffer(h: VulkanBufferHandle) =
  if h.buf != nil: lib.vkDestroyBuffer(gDevice, h.buf, nil)
  if h.mem != nil: lib.vkFreeMemory(gDevice, h.mem, nil)

var gPoolKinds = initTable[VkBuffer, PoolKind]()

proc acquirePooled(kind: PoolKind; bytes: uint64): VulkanBufferHandle =
  let alloc = poolBucket(bytes)
  let key = (kind, alloc)
  if gBufferPool.hasKey(key) and gBufferPool[key].len > 0:
    result = gBufferPool[key].pop()
    gPoolBytes -= alloc
  else:
    case kind
    of pkDevice:
      var created: tuple[buf: VkBuffer, mem: VkDeviceMemory]
      try:
        created = createHostVisibleBuffer(alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      except CatchableError:
        # Out of VRAM: give the idle pooled buffers back and retry once.
        for bufs in gBufferPool.values:
          for h in bufs:
            gPoolKinds.del(h.buf)
            destroyBuffer(h)
        gBufferPool.clear()
        gPoolBytes = 0
        created = createHostVisibleBuffer(alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      result = VulkanBufferHandle(buf: created.buf, mem: created.mem, deviceOnly: true)
    of pkUpload, pkReadback:
      let coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      var created: tuple[buf: VkBuffer, mem: VkDeviceMemory]
      if kind == pkReadback:
        # CPU reads of uncached memory are slow; prefer HOST_CACHED.
        try:
          created = createHostVisibleBuffer(alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            coherent or VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
        except CatchableError:
          created = createHostVisibleBuffer(alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, coherent)
      else:
        created = createHostVisibleBuffer(alloc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, coherent)
      result = VulkanBufferHandle(buf: created.buf, mem: created.mem)
      gpuCheck(lib.vkMapMemory(gDevice, created.mem, 0, alloc, 0, addr result.mapped) == VK_SUCCESS,
        "vkMapMemory (staging) that bai")
    result.pooled = true
    result.allocBytes = alloc
    gPoolKinds[result.buf] = kind
  result.sizeBytes = bytes

proc releaseBuffer(h: VulkanBufferHandle) =
  ## Return a pooled buffer for reuse, or destroy it. Callers guarantee no
  ## queued GPU work still refers to it.
  if h.buf == nil and h.mem == nil: return
  if h.pooled and gPoolKinds.hasKey(h.buf) and gPoolBytes + h.allocBytes <= gPoolLimitBytes:
    gBufferPool.mgetOrPut((gPoolKinds[h.buf], h.allocBytes), @[]).add h
    gPoolBytes += h.allocBytes
  else:
    if h.pooled: gPoolKinds.del(h.buf)
    destroyBuffer(h)

proc immediateCopy(src, dst: VkBuffer; srcOff, dstOff, bytes: uint64) =
  ## One-shot transfer with its own command buffer and fence, independent of
  ## an open session (reads after sessionEnd, uploads outside a session).
  var cb: VkCommandBuffer
  var allocInfo = VkCommandBufferAllocateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext: nil,
    commandPool: gCommandPool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount: 1)
  gpuCheck(lib.vkAllocateCommandBuffers(gDevice, addr allocInfo, addr cb) == VK_SUCCESS,
    "vkAllocateCommandBuffers (copy) that bai")
  var beginInfo = VkCommandBufferBeginInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: nil,
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo: nil)
  var fence: VkFence
  try:
    gpuCheck(lib.vkBeginCommandBuffer(cb, addr beginInfo) == VK_SUCCESS, "vkBeginCommandBuffer (copy) that bai")
    var pre = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: nil,
      srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT or VK_ACCESS_TRANSFER_WRITE_BIT,
      dstAccessMask: VK_ACCESS_TRANSFER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT)
    lib.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT or VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, addr pre, 0, nil, 0, nil)
    var region = VkBufferCopy(srcOffset: srcOff, dstOffset: dstOff, size: bytes)
    lib.vkCmdCopyBuffer(cb, src, dst, 1, addr region)
    var post = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: nil,
      srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
      dstAccessMask: VK_ACCESS_HOST_READ_BIT or VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT)
    lib.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT or VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, addr post, 0, nil, 0, nil)
    gpuCheck(lib.vkEndCommandBuffer(cb) == VK_SUCCESS, "vkEndCommandBuffer (copy) that bai")
    var fenceInfo = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: nil, flags: 0)
    gpuCheck(lib.vkCreateFence(gDevice, addr fenceInfo, nil, addr fence) == VK_SUCCESS, "vkCreateFence (copy) that bai")
    var cbVar = cb
    var submitInfo = VkSubmitInfo(sType: VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext: nil,
      waitSemaphoreCount: 0, pWaitSemaphores: nil, pWaitDstStageMask: nil,
      commandBufferCount: 1, pCommandBuffers: addr cbVar,
      signalSemaphoreCount: 0, pSignalSemaphores: nil)
    gpuCheck(lib.vkQueueSubmit(gQueue, 1, addr submitInfo, fence) == VK_SUCCESS, "vkQueueSubmit (copy) that bai")
    gpuCheck(lib.vkWaitForFences(gDevice, 1, addr fence, 1, high(uint64)) == VK_SUCCESS,
      "vkWaitForFences (copy) that bai")
  finally:
    if fence != nil: lib.vkDestroyFence(gDevice, fence, nil)
    lib.vkFreeCommandBuffers(gDevice, gCommandPool, 1, addr cb)

proc recordTransferBarrierIfNeeded(src, dst: VkBuffer) =
  ## Copy after copy needs a barrier only for RAW/WAW/WAR between the two.
  if src in gXferWritten or dst in gXferWritten or dst in gXferRead:
    var b = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: nil,
      srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
      dstAccessMask: VK_ACCESS_TRANSFER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT)
    lib.vkCmdPipelineBarrier(gSessionCmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, addr b, 0, nil, 0, nil)
    if gVkStatsEnabled: inc gVkBarriers
    gXferWritten.clear()
    gXferRead.clear()

proc recordSessionCopy(src, dst: VkBuffer; srcOff, dstOff, bytes: uint64) =
  ## Earlier dispatches end with a compute->compute|transfer barrier; the
  ## next dispatch emits the transfer->compute barrier (gXferPending).
  recordTransferBarrierIfNeeded(src, dst)
  var region = VkBufferCopy(srcOffset: srcOff, dstOffset: dstOff, size: bytes)
  lib.vkCmdCopyBuffer(gSessionCmdBuf, src, dst, 1, addr region)
  if gVkStatsEnabled: inc gVkCopies
  gXferWritten.incl dst
  gXferRead.incl src
  gXferPending = true

proc stagedUpload(dst: VulkanBufferHandle; data: pointer; bytes: uint64) =
  ## Host data into a device-only buffer: inside a session the copy is
  ## recorded in order with the session's dispatches.
  if bytes == 0: return
  var stage = acquirePooled(pkUpload, bytes)
  copyMem(stage.mapped, data, int(bytes))
  if gSessionActive:
    recordSessionCopy(stage.buf, dst.buf, 0, 0, bytes)
    gSessionStaging.add stage
  else:
    try:
      immediateCopy(stage.buf, dst.buf, 0, 0, bytes)
    finally:
      releaseBuffer(stage)

proc stagedRead(src: VulkanBufferHandle; data: pointer; bytes: uint64) =
  if bytes == 0: return
  var rb = acquirePooled(pkReadback, bytes)
  try:
    immediateCopy(src.buf, rb.buf, 0, 0, bytes)
    copyMem(data, rb.mapped, int(bytes))
  finally:
    releaseBuffer(rb)

proc getOrCreatePipeline(shaderName: string, numBindings: int, pushConstSize: uint32,
                         localSizeX = 0, spirvPathOverride = "", sharedFloats = 0): PipelineEntry =
  ## Pipeline cache keyed by shader name + numBindings + localSizeX.
  ## The cacheKey used to be only shaderName+localSizeX → the same kernel with a different
  ## binding count (microVm 5 vs multiBuf 3..5) returned the old entry →
  ## vkUpdateDescriptorSets wrote past the layout → UB / validation error.
  let cacheKey = shaderName & "_nb" & $numBindings &
    (if localSizeX > 0: "_local" & $localSizeX else: "") & "_sm" & $sharedFloats
  if gPipelineCache.hasKey(cacheKey):
    return gPipelineCache[cacheKey]

  let spirvPath = if spirvPathOverride.len > 0: spirvPathOverride
                  else: currentSourcePath().parentDir() / "kernels" / (shaderName & ".spv")
  gpuCheck(fileExists(spirvPath),
    "Missing " & spirvPath & " - compile it first: glslc kernels/" & shaderName & ".comp -o kernels/" & shaderName & ".spv")
  let spirvBytes = readFile(spirvPath)
  gpuCheck(spirvBytes.len mod 4 == 0, "Invalid SPIR-V file: " & spirvPath)

  var shaderInfo = VkShaderModuleCreateInfo(
    sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext: nil, flags: 0,
    codeSize: uint(spirvBytes.len), pCode: cast[ptr uint32](unsafeAddr spirvBytes[0])
  )
  var shaderModule: VkShaderModule
  gpuCheck(lib.vkCreateShaderModule(gDevice, addr shaderInfo, nil, addr shaderModule) == VK_SUCCESS,
    "vkCreateShaderModule failed for " & shaderName)

  var bindings = newSeq[VkDescriptorSetLayoutBinding](numBindings)
  for i in 0 ..< numBindings:
    bindings[i] = VkDescriptorSetLayoutBinding(
      binding: uint32(i), descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: 1, stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers: nil
    )
  var setLayoutInfo = VkDescriptorSetLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    bindingCount: uint32(numBindings), pBindings: addr bindings[0]
  )
  var setLayout: VkDescriptorSetLayout
  gpuCheck(lib.vkCreateDescriptorSetLayout(gDevice, addr setLayoutInfo, nil, addr setLayout) == VK_SUCCESS,
    "vkCreateDescriptorSetLayout failed for " & shaderName)

  var pcRange = VkPushConstantRange(stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, offset: 0, size: pushConstSize)
  var pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    setLayoutCount: 1, pSetLayouts: addr setLayout,
    pushConstantRangeCount: (if pushConstSize > 0: 1'u32 else: 0'u32),
    pPushConstantRanges: (if pushConstSize > 0: addr pcRange else: nil)
  )
  var pipelineLayout: VkPipelineLayout
  gpuCheck(lib.vkCreatePipelineLayout(gDevice, addr pipelineLayoutInfo, nil, addr pipelineLayout) == VK_SUCCESS,
    "vkCreatePipelineLayout failed for " & shaderName)

  var specEntries = [
    VkSpecializationMapEntry(constantID: 0, offset: 0, size: uint(sizeof(uint32))),
    VkSpecializationMapEntry(constantID: 1, offset: 4, size: uint(sizeof(uint32)))]
  var specValues = [uint32(max(localSizeX, 1)), uint32(max(sharedFloats, 1))]
  var specInfo = VkSpecializationInfo(mapEntryCount: (if sharedFloats > 0: 2'u32 else: 1'u32),
    pMapEntries: addr specEntries[0], dataSize: (if sharedFloats > 0: 8'u else: 4'u),
    pData: addr specValues[0])
  var stageInfo = VkPipelineShaderStageCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, pNext: nil, flags: 0,
    stage: VK_SHADER_STAGE_COMPUTE_BIT, module: shaderModule, pName: "main",
    pSpecializationInfo: (if localSizeX > 0: addr specInfo else: nil)
  )
  var pipelineInfo = VkComputePipelineCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, pNext: nil, flags: 0,
    stage: stageInfo, layout: pipelineLayout, basePipelineHandle: nil, basePipelineIndex: -1
  )
  var pipeline: VkPipeline
  ensureVkPipelineCache()
  gpuCheck(lib.vkCreateComputePipelines(gDevice, gVkPipelineCacheObject, 1, addr pipelineInfo, nil, addr pipeline) == VK_SUCCESS,
    "vkCreateComputePipelines failed for " & shaderName)

  result = PipelineEntry(pipeline: pipeline, layout: pipelineLayout, setLayout: setLayout,
                          numBindings: numBindings, pushConstSize: pushConstSize)
  gPipelineCache[cacheKey] = result

const DSetPoolSets = 256

var gDSetPool: VkDescriptorPool
var gDSetPoolLeft = 0

proc takeDescriptorPool(): VkDescriptorPool =
  ## One session-scoped pool instead of one pool per dispatch: on a fast GPU
  ## the ~1000 prefill dispatches are dominated by that driver allocation.
  ## The pool is retired into gSessionCleanupPools when it runs out, and
  ## destroyed only at sessionEnd() - after the fence wait.
  if gDSetPoolLeft <= 0:
    if gDSetPool != nil: gSessionCleanupPools.add(gDSetPool)
    var poolSize = VkDescriptorPoolSize(descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      descriptorCount: uint32(DSetPoolSets * 8))
    var poolInfo = VkDescriptorPoolCreateInfo(
      sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, pNext: nil, flags: 0,
      maxSets: uint32(DSetPoolSets), poolSizeCount: 1, pPoolSizes: addr poolSize)
    gpuCheck(lib.vkCreateDescriptorPool(gDevice, addr poolInfo, nil, addr gDSetPool) == VK_SUCCESS,
      "vkCreateDescriptorPool (session) that bai")
    gDSetPoolLeft = DSetPoolSets
  dec gDSetPoolLeft
  result = gDSetPool

proc makeDescriptorSet(entry: PipelineEntry, buffers: openArray[VulkanBufferHandle]; offsets: seq[uint64] = @[]): tuple[pool: VkDescriptorPool, dset: VkDescriptorSet] =
  ## One descriptor set pointing at the given buffers. The set comes from the
  ## session's pool (takeDescriptorPool) - the pool is destroyed only in sessionEnd()
  ## after the fence, because a set may not be freed before its dispatch has finished
  ## (same reason sessionRead() must not be called in the middle of a session).
  let pool = takeDescriptorPool()

  var setLayoutVar = entry.setLayout
  var allocInfo = VkDescriptorSetAllocateInfo(
    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pNext: nil,
    descriptorPool: pool, descriptorSetCount: 1, pSetLayouts: addr setLayoutVar
  )
  var dset: VkDescriptorSet
  gpuCheck(lib.vkAllocateDescriptorSets(gDevice, addr allocInfo, addr dset) == VK_SUCCESS, "vkAllocateDescriptorSets (session) that bai")

  var bufInfos = newSeq[VkDescriptorBufferInfo](buffers.len)
  var writes = newSeq[VkWriteDescriptorSet](buffers.len)
  for i in 0 ..< buffers.len:
    let offB = if offsets.len > i: offsets[i] else: 0'u64
    let sz = buffers[i].sizeBytes
    # An offset past the buffer size used to silently use range=4 → silently read garbage.
    # Fails loudly with BYBY_VULKAN_STRICT=1; by default it still clamps but logs once.
    if offB >= sz:
      if getEnv("BYBY_VULKAN_STRICT", "0") != "0":
        raise newException(CatchableError,
          "makeDescriptorSet: buffer[" & $i & "] offset=" & $offB &
          " >= sizeBytes=" & $sz & " (would read garbage)")
      elif getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") == "1":
        stderr.writeLine "[vulkan] WARN buffer[" & $i & "] offset=" & $offB &
          " >= size=" & $sz & " → clamp range=4"
    let rng = if offB >= sz: 4'u64 else: sz - offB
    bufInfos[i] = VkDescriptorBufferInfo(buffer: buffers[i].buf, offset: offB, range: rng)
  for i in 0 ..< buffers.len:
    writes[i] = VkWriteDescriptorSet(
      sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext: nil,
      dstSet: dset, dstBinding: uint32(i), dstArrayElement: 0,
      descriptorCount: 1, descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      pImageInfo: nil, pBufferInfo: addr bufInfos[i], pTexelBufferView: nil
    )
  lib.vkUpdateDescriptorSets(gDevice, uint32(buffers.len), addr writes[0], 0, nil)
  if gVkStatsEnabled: inc gVkDescriptorSets
  result = (pool, dset)

proc vulkanSessionBegin*(): bool =
  ensureInit()
  if gSessionActive: return false
  var cmdAllocInfo = VkCommandBufferAllocateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext: nil,
    commandPool: gCommandPool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount: 1
  )
  if lib.vkAllocateCommandBuffers(gDevice, addr cmdAllocInfo, addr gSessionCmdBuf) != VK_SUCCESS:
    return false
  var beginInfo = VkCommandBufferBeginInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: nil,
    flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo: nil
  )
  if lib.vkBeginCommandBuffer(gSessionCmdBuf, addr beginInfo) != VK_SUCCESS:
    return false
  gVkSessionTimingActive = false
  if getEnv("BYBY_VULKAN_TIMESTAMPS", "0") == "1" and
      gQueueTimestampValidBits > 0 and
      gDeviceCaps.properties.limits.timestampComputeAndGraphics != 0 and
      gDeviceCaps.properties.limits.timestampPeriod > 0:
    var queryInfo = VkQueryPoolCreateInfo(sType: VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      queryType: VK_QUERY_TYPE_TIMESTAMP, queryCount: 2)
    if lib.vkCreateQueryPool(gDevice, addr queryInfo, nil, addr gVkSessionQueryPool) == VK_SUCCESS:
      lib.vkCmdResetQueryPool(gSessionCmdBuf, gVkSessionQueryPool, 0, 2)
      lib.vkCmdWriteTimestamp(gSessionCmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        gVkSessionQueryPool, 0)
      gVkSessionTimingActive = true
  gSessionCleanupPools.setLen(0)
  gSessionInternalBufs.setLen(0)
  gXferWritten.clear()
  gXferRead.clear()
  gXferPending = false
  # A failed sessionEnd() can leave the live descriptor pool behind; nothing
  # was submitted in that case, so drop it before the handle is reused.
  if gDSetPool != nil:
    lib.vkDestroyDescriptorPool(gDevice, gDSetPool, nil)
    gDSetPool = nil
    gDSetPoolLeft = 0
  gSessionActive = true
  result = true

proc vulkanSessionEnd*(): bool =
  if not gSessionActive: return false
  gSessionActive = false
  gVkLastError = ""
  if gVkSessionTimingActive:
    lib.vkCmdWriteTimestamp(gSessionCmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      gVkSessionQueryPool, 1)
  var endCmdStart: MonoTime
  if gVkStatsEnabled: endCmdStart = getMonoTime()
  if lib.vkEndCommandBuffer(gSessionCmdBuf) != VK_SUCCESS:
    gVkLastError = "vkEndCommandBuffer that bai"
    if gVkSessionQueryPool != nil:
      lib.vkDestroyQueryPool(gDevice, gVkSessionQueryPool, nil)
      gVkSessionQueryPool = nil
      gVkSessionTimingActive = false
    return false
  if gVkStatsEnabled:
    gVkEndCommandSeconds += float((getMonoTime() - endCmdStart).inNanoseconds) / 1e9
    inc gVkSessionEnds
  var fenceCreateStart: MonoTime
  if gVkStatsEnabled: fenceCreateStart = getMonoTime()
  var fenceInfo = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: nil, flags: 0)
  var fence: VkFence
  if lib.vkCreateFence(gDevice, addr fenceInfo, nil, addr fence) != VK_SUCCESS:
    if gVkStatsEnabled:
      gVkFenceCreateSeconds += float((getMonoTime() - fenceCreateStart).inNanoseconds) / 1e9
    if gVkSessionQueryPool != nil:
      lib.vkDestroyQueryPool(gDevice, gVkSessionQueryPool, nil)
      gVkSessionQueryPool = nil
      gVkSessionTimingActive = false
    return false
  if gVkStatsEnabled:
    gVkFenceCreateSeconds += float((getMonoTime() - fenceCreateStart).inNanoseconds) / 1e9
  var cmdBufVar = gSessionCmdBuf
  var submitInfo = VkSubmitInfo(
    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext: nil,
    waitSemaphoreCount: 0, pWaitSemaphores: nil, pWaitDstStageMask: nil,
    commandBufferCount: 1, pCommandBuffers: addr cmdBufVar,
    signalSemaphoreCount: 0, pSignalSemaphores: nil
  )
  var submitStart: MonoTime
  if gVkStatsEnabled: submitStart = getMonoTime()
  let submitOk = lib.vkQueueSubmit(gQueue, 1, addr submitInfo, fence) == VK_SUCCESS
  if gVkStatsEnabled:
    gVkQueueSubmitSeconds += float((getMonoTime() - submitStart).inNanoseconds) / 1e9
    if submitOk: inc gVkQueueSubmits
  var waitStart: MonoTime
  if gVkStatsEnabled and submitOk: waitStart = getMonoTime()
  let waitOk = submitOk and (lib.vkWaitForFences(gDevice, 1, addr fence, 1, high(uint64)) == VK_SUCCESS)
  if gVkStatsEnabled and submitOk:
    gVkFenceWaitSeconds += float((getMonoTime() - waitStart).inNanoseconds) / 1e9
  if not submitOk: gVkLastError = "vkQueueSubmit that bai"
  elif not waitOk: gVkLastError = "vkWaitForFences that bai"
  if waitOk and gVkSessionTimingActive:
    var stamps: array[2, uint64]
    let queryOk = lib.vkGetQueryPoolResults(gDevice, gVkSessionQueryPool, 0, 2,
      uint(sizeof(stamps)), addr stamps[0], uint64(sizeof(uint64)), VK_QUERY_RESULT_64_BIT)
    if queryOk == VK_SUCCESS:
      var ticks = stamps[1] - stamps[0]
      if gQueueTimestampValidBits < 64:
        ticks = ticks and ((1'u64 shl int(gQueueTimestampValidBits)) - 1'u64)
      gVkGpuSecondsTotal += float(ticks) * float(gDeviceCaps.properties.limits.timestampPeriod) / 1e9
  if gVkSessionQueryPool != nil:
    lib.vkDestroyQueryPool(gDevice, gVkSessionQueryPool, nil)
    gVkSessionQueryPool = nil
  gVkSessionTimingActive = false
  var cleanupStart: MonoTime
  if gVkStatsEnabled: cleanupStart = getMonoTime()
  lib.vkDestroyFence(gDevice, fence, nil)
  lib.vkFreeCommandBuffers(gDevice, gCommandPool, 1, addr cmdBufVar)
  if gDSetPool != nil:
    lib.vkDestroyDescriptorPool(gDevice, gDSetPool, nil)
    gDSetPool = nil
    gDSetPoolLeft = 0
  for p in gSessionCleanupPools:
    lib.vkDestroyDescriptorPool(gDevice, p, nil)
  gSessionCleanupPools.setLen(0)
  # Free multiBuf internal dummies only (not caller-owned AllocScratch).
  for h in gSessionInternalBufs: releaseBuffer(h)
  gSessionInternalBufs.setLen(0)
  # Destroy the buffers freed with sessionFree() during the session (safe, the fence has been waited on).
  for h in gSessionDeferredFree: releaseBuffer(h)
  gSessionDeferredFree.setLen(0)
  for h in gSessionStaging: releaseBuffer(h)
  gSessionStaging.setLen(0)
  if gVkStatsEnabled:
    gVkCleanupSeconds += float((getMonoTime() - cleanupStart).inNanoseconds) / 1e9
  result = waitOk

proc vulkanGetLastError*(): string = gVkLastError

proc vulkanSessionCommitContinue*(): bool =
  ## Equivalent of metalSessionCommitContinue: closes the current command buffer (waits for the GPU) and opens
  ## a new session so the caller can keep encoding. This driver has no "commit without waiting" on Vulkan.
  if not gSessionActive: return false
  if not vulkanSessionEnd(): return false
  result = vulkanSessionBegin()

proc vulkanRecoverQueue*(): bool =
  ## Vulkan has no "poisoned queue" like MTLCommandQueue. If a session is stuck
  ## (gSessionActive = true after an error), drop it so the next session can open. Returns true = ready.
  if gSessionActive:
    gSessionActive = false
    discard lib.vkEndCommandBuffer(gSessionCmdBuf)
    if gVkSessionQueryPool != nil:
      lib.vkDestroyQueryPool(gDevice, gVkSessionQueryPool, nil)
      gVkSessionQueryPool = nil
      gVkSessionTimingActive = false
    var cb = gSessionCmdBuf
    lib.vkFreeCommandBuffers(gDevice, gCommandPool, 1, addr cb)
    if gDSetPool != nil:
      lib.vkDestroyDescriptorPool(gDevice, gDSetPool, nil)
      gDSetPool = nil
      gDSetPoolLeft = 0
    for p in gSessionCleanupPools:
      lib.vkDestroyDescriptorPool(gDevice, p, nil)
    gSessionCleanupPools.setLen(0)
    for h in gSessionInternalBufs: releaseBuffer(h)
    gSessionInternalBufs.setLen(0)
    for h in gSessionDeferredFree: releaseBuffer(h)
    gSessionDeferredFree.setLen(0)
    for h in gSessionStaging: releaseBuffer(h)
    gSessionStaging.setLen(0)
  result = true

proc vulkanSessionUpload*(data: seq[float32]): VulkanBufferHandle =
  defer:
    if unlikely(firProbeOn) and data.len > 0:
      probeUploaded(cast[pointer](result.buf), unsafeAddr data[0], data.len*4)
  ensureInit()
  let bytes = uint64(data.len * sizeof(float32))
  if gScratchDeviceLocal:
    result = acquirePooled(pkDevice, bytes)
    if data.len > 0: stagedUpload(result, unsafeAddr data[0], bytes)
    return
  let (buf, mem) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  if data.len > 0:
    var dataVar = data
    uploadToBuffer(mem, addr dataVar[0], bytes)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: bytes)

proc vulkanSessionAllocScratch*(n: int): VulkanBufferHandle =
  ## Caller owns the handle: free with vulkanSessionFree AFTER sessionEnd
  ## (or registerInternalScratch if it is a throwaway dummy for this session).
  ensureInit()
  if n < 0:
    raise newException(CatchableError, "vulkanSessionAllocScratch: n < 0")
  let bytes = uint64(max(n, 0) * sizeof(float32))
  if gScratchDeviceLocal:
    return acquirePooled(pkDevice, bytes)
  let (buf, mem) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: bytes)

proc registerInternalScratch*(h: VulkanBufferHandle) =
  ## multiBuf dummy only — destroyed in vulkanSessionEnd. Do not register
  ## buffers the caller still reads after End.
  if gSessionActive:
    gSessionInternalBufs.add(h)

proc vulkanSessionRead*(h: VulkanBufferHandle, n: int): seq[float32] =
  ## ONLY call AFTER vulkanSessionEnd() has completed (fence waited) - reading
  ## in the middle of an uncommitted session gives garbage/stale data (lesson from Metal).
  result = newSeq[float32](n)
  if n > 0:
    if h.deviceOnly: stagedRead(h, addr result[0], uint64(n * sizeof(float32)))
    else: readFromBuffer(h.mem, addr result[0], uint64(n * sizeof(float32)))

proc vulkanSessionFree*(h: var VulkanBufferHandle) =
  if gSessionActive and (h.buf != nil or h.mem != nil):
    gSessionDeferredFree.add(h)   # may still be referenced by the command buffer being recorded -> deferred destroy
    h.buf = nil
    h.mem = nil
    return
  releaseBuffer(h)
  h.buf = nil
  h.mem = nil

proc vulkanSessionCopyInto*(src: VulkanBufferHandle, dst: var VulkanBufferHandle,
                             srcOffsetElems, dstOffsetElems, n: int): bool =
  if not gSessionActive: return false
  # Barriers are recorded lazily: a KV append is 4*HKV copies per layer, and
  # a barrier pair around each one stalled the GPU thousands of times a token.
  recordSessionCopy(src.buf, dst.buf, uint64(srcOffsetElems * sizeof(float32)),
    uint64(dstOffsetElems * sizeof(float32)), uint64(n * sizeof(float32)))
  result = true

proc uploadResidentData(data: pointer; nBytes: uint64): VulkanBufferHandle =
  ensureInit()
  let localMode = getEnv("BYBY_VULKAN_DEVICE_LOCAL", "auto")
  let deviceLocal = useDeviceLocalWeights(gDeviceCaps.properties.deviceType, localMode)
  let allocBytes = max(4'u64, (nBytes + 3) and not 3'u64)
  let flags = if deviceLocal: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
              else: VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  let (buf, mem) = createHostVisibleBuffer(allocBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, flags)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: nBytes, deviceOnly: deviceLocal)
  if nBytes == 0: return
  if not deviceLocal:
    uploadToBuffer(mem, data, nBytes)
    return
  let (stageBuf, stageMem) = createHostVisibleBuffer(allocBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
  var staging = VulkanBufferHandle(buf: stageBuf, mem: stageMem, sizeBytes: allocBytes)
  uploadToBuffer(stageMem, data, nBytes)
  let ownsSession = not gSessionActive
  if ownsSession and not vulkanSessionBegin():
    vulkanSessionFree(staging); vulkanSessionFree(result)
    raise newException(CatchableError, "resident Vulkan staging session failed")
  if not vulkanSessionCopyInto(staging, result, 0, 0, int(allocBytes div 4)):
    vulkanSessionFree(staging); vulkanSessionFree(result)
    if ownsSession: discard vulkanSessionEnd()
    raise newException(CatchableError, "resident Vulkan staging copy failed")
  # The queued transfer retains its staging allocation until the fence.
  vulkanSessionFree(staging)
  if ownsSession and not vulkanSessionEnd():
    vulkanSessionFree(result)
    raise newException(CatchableError, "resident Vulkan upload failed: " & gVkLastError)

proc vulkanImportHostBuffer*(data: pointer; bytes: int): VulkanBufferHandle =
  defer:
    if unlikely(firProbeOn): probeUploaded(cast[pointer](result.buf), data, bytes)
  ## Storage buffer over caller memory (a model mmap) without copying, on an
  ## iGPU with VK_EXT_external_memory_host. `data` must be aligned to the
  ## driver's import alignment and readable up to the next multiple of it; the
  ## caller keeps it mapped until vulkanSessionFree. buf == nil means the
  ## driver declined and the caller should copy instead.
  ensureInit()
  if not gHostImportEnabled or data == nil or bytes <= 0: return
  let align = gDeviceCaps.hostImportAlignment
  if cast[uint64](data) mod align != 0: return
  let size = (uint64(bytes) + align - 1) div align * align
  type
    ExternalBufferInfo = object
      sType: uint32
      pNext: pointer
      handleTypes: uint32
    HostPointerProps = object
      sType: uint32
      pNext: pointer
      memoryTypeBits: uint32
    ImportHostPointerInfo = object
      sType: uint32
      pNext: pointer
      handleType: uint32
      pHostPointer: pointer
  # HOST_ALLOCATION first; a file mapping may need MAPPED_FOREIGN_MEMORY.
  for handleType in [0x80'u32, 0x100'u32]:
    var hostProps = HostPointerProps(sType: 1000178001)
    if vkGetMemoryHostPointerPropertiesEXT(gDevice, handleType, data,
        addr hostProps) != VK_SUCCESS or hostProps.memoryTypeBits == 0: continue
    var external = ExternalBufferInfo(sType: 1000072000, handleTypes: handleType)
    var bufInfo = VkBufferCreateInfo(sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      pNext: addr external, size: size,
      usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      sharingMode: VK_SHARING_MODE_EXCLUSIVE)
    var buf: VkBuffer
    if lib.vkCreateBuffer(gDevice, addr bufInfo, nil, addr buf) != VK_SUCCESS: continue
    var req: VkMemoryRequirements
    lib.vkGetBufferMemoryRequirements(gDevice, buf, addr req)
    let typeBits = req.memoryTypeBits and hostProps.memoryTypeBits
    var memProps: VkPhysicalDeviceMemoryProperties
    lib.vkGetPhysicalDeviceMemoryProperties(gPhysicalDevice, addr memProps)
    var typeIndex = -1
    for i in 0 ..< int(memProps.memoryTypeCount):
      if (typeBits and (1'u32 shl uint32(i))) != 0:
        typeIndex = i
        break
    if typeIndex < 0 or req.size > size:
      lib.vkDestroyBuffer(gDevice, buf, nil)
      continue
    var importInfo = ImportHostPointerInfo(sType: 1000178000, handleType: handleType,
      pHostPointer: data)
    var allocInfo = VkMemoryAllocateInfo(sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      pNext: addr importInfo, allocationSize: size, memoryTypeIndex: uint32(typeIndex))
    var mem: VkDeviceMemory
    if lib.vkAllocateMemory(gDevice, addr allocInfo, nil, addr mem) != VK_SUCCESS:
      lib.vkDestroyBuffer(gDevice, buf, nil)
      continue
    if lib.vkBindBufferMemory(gDevice, buf, mem, 0) != VK_SUCCESS:
      lib.vkDestroyBuffer(gDevice, buf, nil)
      lib.vkFreeMemory(gDevice, mem, nil)
      continue
    return VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: uint64(bytes))

proc vulkanImportHostRange*(data: pointer; bytes: int): tuple[h: VulkanBufferHandle, offset: int] =
  ## vulkanImportHostBuffer for data at any address inside a mapping (a tensor
  ## in an unaligned .nimq): the buffer starts at the enclosing import-aligned
  ## address and `offset` is the byte position of `data` inside it. Kernels
  ## take that offset as an argument.
  ensureInit()
  if not gHostImportEnabled or data == nil or bytes <= 0: return
  let align = gDeviceCaps.hostImportAlignment
  result.offset = int(cast[uint64](data) mod align)
  result.h = vulkanImportHostBuffer(cast[pointer](cast[uint64](data) - uint64(result.offset)),
    result.offset + bytes)
  if result.h.buf == nil: result.offset = 0

proc vulkanSessionUploadPrivate*(data: openArray[float32]): VulkanBufferHandle =
  result = uploadResidentData((if data.len > 0: cast[pointer](unsafeAddr data[0]) else: nil),
    uint64(data.len * sizeof(float32)))
  if unlikely(firProbeOn) and data.len > 0:
    probeUploaded(cast[pointer](result.buf), unsafeAddr data[0], data.len*4)

proc vulkanSessionUploadBytesPrivate*(data: openArray[uint8]): VulkanBufferHandle =
  result = uploadResidentData((if data.len > 0: cast[pointer](unsafeAddr data[0]) else: nil), uint64(data.len))
  if unlikely(firProbeOn) and data.len > 0:
    probeUploaded(cast[pointer](result.buf), unsafeAddr data[0], data.len)

proc dispatchGeneric(shaderName: string, buffers: openArray[VulkanBufferHandle],
                      pushConstBytes: pointer, pushConstSize: uint32,
                      groupX, groupY, groupZ: uint32, localSizeX = 0,
                      spirvPathOverride = "",
                      bufferOffsets: seq[uint64] = @[]; sharedFloats = 0): bool =
  if not gSessionActive: return false
  let entry = getOrCreatePipeline(shaderName, buffers.len, pushConstSize,
    localSizeX, spirvPathOverride, sharedFloats)
  let (pool, dset) = makeDescriptorSet(entry, buffers, bufferOffsets)
  discard pool  # owned by the session pool, retired at sessionEnd
  if gXferPending:
    # Copies recorded since the last dispatch (uploads, KV appends).
    var xfer = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: nil,
      srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
      dstAccessMask: VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT)
    lib.vkCmdPipelineBarrier(gSessionCmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, addr xfer, 0, nil, 0, nil)
    if gVkStatsEnabled: inc gVkBarriers
    gXferPending = false
    gXferWritten.clear()
    gXferRead.clear()
  lib.vkCmdBindPipeline(gSessionCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, entry.pipeline)
  var dsetVar = dset
  lib.vkCmdBindDescriptorSets(gSessionCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, entry.layout, 0, 1, addr dsetVar, 0, nil)
  if pushConstSize > 0:
    lib.vkCmdPushConstants(gSessionCmdBuf, entry.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstSize, pushConstBytes)
  lib.vkCmdDispatch(gSessionCmdBuf, groupX, groupY, groupZ)
  if gVkStatsEnabled: inc gVkDispatches
  # Metal's command encoder orders compute dispatches with a buffer barrier;
  # Vulkan requires an explicit memory dependency for write->read visibility.
  # Also covers a copy recorded next (KV append, readback source).
  var barrier = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    pNext: nil, srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
    dstAccessMask: VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT or
                   VK_ACCESS_TRANSFER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT)
  lib.vkCmdPipelineBarrier(gSessionCmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT or VK_PIPELINE_STAGE_TRANSFER_BIT,
    0, 1, addr barrier, 0, nil, 0, nil)
  if gVkStatsEnabled: inc gVkBarriers
  result = true

proc vulkanSessionMicroVmEnc*(code: VulkanBufferHandle, mem: VulkanBufferHandle,
                               nThreads, tgFloats, tgSize: int): bool =
  ## Bytecode interpreter counterpart to Metal's micro_vm_kernel encoder.
  ## The Vulkan shader is specialized to the exact workgroup width because
  ## threadgroup-memory reductions/scans depend on local invocation IDs.
  if not gSessionActive or code.buf == nil or nThreads <= 0 or tgSize <= 0:
    return false
  if tgSize > 1024 or tgFloats > 1024:
    return false
  let groups = uint32((nThreads + tgSize - 1) div tgSize)
  let nWords = int(code.sizeBytes div uint64(sizeof(uint32)))
  var pc = [int32(nWords), int32(max(1, tgFloats))]
  # mem is also a valid backing store for optional byte/LUT/const/input views;
  # current block programs address these through the shared scratch layout.
  result = dispatchGeneric("micro_vm", [code, mem, mem, mem, mem, mem, mem],
    addr pc[0], 8, groups, 1, 1, tgSize)

var gJitSpvPaths = initTable[string, tuple[source, path: string]]()
var gJitPathHits, gJitPathMisses, gJitDummyAllocations: int

proc vulkanJitCacheStats*(): tuple[pathHits, pathMisses, dummyAllocations: int] =
  (gJitPathHits, gJitPathMisses, gJitDummyAllocations)

proc vulkanJitCacheDir*(): string =
  ## BYBY_VULKAN_JIT_DIR points the SPIR-V cache at persistent storage (Colab:
  ## a mounted Drive, CI: a workspace cache). The default lives in the temp
  ## dir, which a fresh container always starts empty - that is what makes the
  ## first prefill spend ~10-20 s compiling shaders on a fast GPU.
  let override = getEnv("BYBY_VULKAN_JIT_DIR")
  if override.len > 0: return override
  getTempDir() / "nimformers-vulkan-jit"

proc vulkanJitShaderPaths(kernelName, source: string): tuple[srcPath, spvPath: string] =
  let sourceKey = toHex(cast[uint64](hash(source)), 16) & "_" & $source.len
  let dir = vulkanJitCacheDir()
  createDir(dir)
  let stem = kernelName & "_" & sourceKey
  (dir / (stem & ".comp"), dir / (stem & ".spv"))

proc vulkanJitPrewarm*(jobs: openArray[tuple[name, source: string]]) =
  ## Compile every shader the next phase needs concurrently. Each glslc call is
  ## a separate process, so N shaders cost about one compile instead of the
  ## sum. On a fresh container (no SPIR-V disk cache) this is the difference
  ## between ~15 s and ~2 s of first-prefill time, and it is the only prefill
  ## cost that does not shrink on a faster GPU.
  if getEnv("BYBY_VULKAN_DISPATCH_CACHE", "1") == "0": return
  if jobs.len == 0: return
  let compiler = findExe("glslc")
  if compiler.len == 0:
    raise newException(CatchableError,
      "Vulkan JIT needs glslc on PATH (install shaderc/Vulkan SDK)")
  let budget = max(2, min(8, countProcessors()))
  type PendingJob = tuple[name, source, srcPath, spvPath: string, running: Process]
  var pending: seq[PendingJob]
  proc drain() =
    for job in pending.mitems:
      let code = job.running.waitForExit()
      job.running.close()
      if code == 0:
        gJitSpvPaths[job.name] = (job.source, job.spvPath)
      elif getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") != "0":
        stderr.writeLine "[vulkan_jit] prewarm failed kernel=" & job.name &
          " exit=" & $code & " (retried on first dispatch)"
    pending.setLen(0)
  var spawned = 0
  for job in jobs:
    if job.name.len == 0 or job.source.len == 0: continue
    let (srcPath, spvPath) = vulkanJitShaderPaths(job.name, job.source)
    if fileExists(spvPath):
      gJitSpvPaths[job.name] = (job.source, spvPath)
      continue
    if job.name in gJitSpvPaths and gJitSpvPaths[job.name].source == job.source:
      continue
    writeFile(srcPath, job.source)
    var args = @["-fshader-stage=compute"]
    if job.source.contains("GL_KHR_cooperative_matrix"):
      args.add "--target-env=vulkan1.3"
    args.add srcPath
    args.add "-o"
    args.add spvPath
    try:
      # poParentStreams keeps the child off our pipes: a failed compile can
      # print freely without blocking a background prewarm.
      pending.add((job.name, job.source, srcPath, spvPath,
        startProcess(compiler, args = args, options = {poUsePath, poParentStreams})))
    except OSError:
      continue
    inc spawned
    if pending.len >= budget: drain()
  drain()
  if getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") != "0" and spawned > 0:
    stderr.writeLine "[vulkan_jit] prewarm compiled " & $spawned &
      " shader(s), concurrency " & $budget

proc compileVulkanJitShader(kernelName, source: string): string =
  ## Runtime GLSL -> SPIR-V compile, cached by complete source hash. glslc is
  ## the Vulkan analogue of Metal's newLibraryWithSource JIT path.
  let useCache = getEnv("BYBY_VULKAN_DISPATCH_CACHE", "1") != "0"
  if useCache and kernelName in gJitSpvPaths:
    let cached = gJitSpvPaths[kernelName]
    if cached.source == source:
      inc gJitPathHits
      return cached.path
  inc gJitPathMisses
  let compiler = findExe("glslc")
  if compiler.len == 0:
    raise newException(CatchableError,
      "Vulkan JIT needs glslc on PATH (install shaderc/Vulkan SDK)")
  let (srcPath, spvPath) = vulkanJitShaderPaths(kernelName, source)
  if fileExists(spvPath):
    # cache hit - silent
    discard
  else:
    if getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") != "0":
      stderr.writeLine "[vulkan_jit] compile kernel=" & kernelName &
        " srcBytes=" & $source.len & " -> " & spvPath
    writeFile(srcPath, source)
    if getEnv("BYBY_JIT_DUMP_SOURCE", "0") == "1":
      stderr.writeLine "════ [vulkan_jit] GLSL source kernel=" & kernelName & " ════"
      stderr.writeLine source
      stderr.writeLine "════ [vulkan_jit] het source " & kernelName & " ════"
    let target = if source.contains("GL_KHR_cooperative_matrix"): " --target-env=vulkan1.3"
                 elif source.contains("GL_KHR_shader_subgroup"): " --target-env=vulkan1.1"
                 else: ""
    let cmd = quoteShell(compiler) & target & " -fshader-stage=compute " &
      quoteShell(srcPath) & " -o " & quoteShell(spvPath)
    let compiled = execCmdEx(cmd)
    if compiled.exitCode != 0:
      stderr.writeLine "[vulkan_jit] FAIL kernel=" & kernelName &
        " glslc exit=" & $compiled.exitCode
      raise newException(CatchableError,
        "glslc Vulkan JIT compile failed for " & kernelName & ":\n" & compiled.output)
    # first compile only - silent on cache hits
  result = spvPath
  if useCache: gJitSpvPaths[kernelName] = (source, spvPath)

proc vulkanSessionJitMicroVmEnc*(kernelName, source: string,
                                  mem: VulkanBufferHandle,
                                  nThreads, tgSize: int; tgFloats = 1024): bool =
  ## One statically-lowered SPIR-V dispatch per bytecode VM program. All
  ## logical buffers currently alias the caller's unified VM scratch buffer.
  if not gSessionActive or mem.buf == nil or nThreads <= 0 or tgSize <= 0 or tgSize > gDeviceCaps.maxThreads or
      tgFloats * 4 > gDeviceCaps.maxSharedBytes:
    return false
  let spvPath = compileVulkanJitShader(kernelName, source)
  let groups = uint32((nThreads + tgSize - 1) div tgSize)
  result = dispatchGeneric(kernelName, [mem, mem, mem, mem, mem], nil, 0,
    groups, 1, 1, tgSize, spvPath, sharedFloats = max(1, tgFloats))
  if not result:
    stderr.writeLine "[vulkan_jit] microVm FAIL kernel=" & kernelName

proc vulkanSessionVecOpEnc*(a, b: VulkanBufferHandle, c: var VulkanBufferHandle, op, n: int): bool =
  var pc = [int32(op), int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("vecop", [a, b, c], addr pc[0], 8, groups, 1, 1)

proc coopMatmulShader(weightFormat: int): tuple[name, source: string] =
  const templateSrc = staticRead("kernels/matmul_coop.glsl")
  ("coop_matmul_format" & $weightFormat, "#version 450\n#define WEIGHT_FORMAT " &
    $weightFormat & "\n" & templateSrc.replace("#version 450\n", ""))

const Fp4KmFp8Shader = (name: "coop_matmul_fp4km_fp8_e4m3",
  source: staticRead("kernels/matmul_fp4km_fp8.glsl"))

proc tiledMatmulShader(weightFormat: int): tuple[name, source: string] =
  ## matmul_fp4km_tiled.glsl: format 1 = packed FP4-KM, 0 = FP32 rows,
  ## 2 = asymmetric Q4.
  const templateSrc = staticRead("kernels/matmul_fp4km_tiled.glsl")
  let name = case weightFormat
    of 1: "coop_matmul_fp4km_tiled"
    of 2: "coop_matmul_q4_tiled"
    else: "coop_matmul_f32_tiled"
  (name, "#version 450\n#define WEIGHT_FORMAT " & $weightFormat &
    "\n#define WG_THREADS " & $(4 * gDeviceCaps.subgroupSize) & "\n" &
    templateSrc.replace("#version 450\n", ""))

proc tiledCoopSupported(): bool =
  ## The tiled kernel runs four subgroups per workgroup: 32-wide (NVIDIA,
  ## Intel, AMD wave32) or 64-wide (AMD wave64).
  gCoopEnabled and gDeviceCaps.cooperativeMatrix and
    gDeviceCaps.subgroupSize in [32, 64] and
    gDeviceCaps.maxThreads >= 4 * gDeviceCaps.subgroupSize and
    gDeviceCaps.maxSharedBytes >= 24 * 1024

const SplitKReduceShader = (name: "coop_splitk_reduce", source: """#version 450
// Sums the split-K partial results of matmul_fp4km_tiled: y[i] = sum_z part[z][i].
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Part { float part[]; };
layout(std430, binding = 1) writeonly buffer Out { float y[]; };
layout(push_constant) uniform Params { uint count, slices; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    float s = 0.0;
    for (uint z = 0u; z < p.slices; ++z) s += part[z * p.count + i];
    y[i] = s;
}
""")

var gSplitKScratch: VulkanBufferHandle   ## partial sums, reused (barriers order it)
var gSplitKLogged = initTable[string, bool]()

proc tiledSplitK(M, K, N: int): int =
  ## K slices for the tiled kernel. Its grid is ceil(N/64) x ceil(M/64)
  ## workgroups; a short prompt gives too few to keep every SM/CU (and the
  ## matrix units inside it) busy, so split K until the grid reaches about
  ## four workgroups per SM/CU. Each slice keeps at least 256 values of K.
  ## BYBY_VULKAN_SPLIT_K=<n> fixes the slice count (1 = off).
  let mode = getEnv("BYBY_VULKAN_SPLIT_K", "auto")
  if mode != "auto":
    try: result = parseInt(mode)
    except ValueError: result = 1
  else:
    let units = if gDeviceCaps.computeUnits > 0: gDeviceCaps.computeUnits else: 16
    let grid = ((N + 63) div 64) * ((M + 63) div 64)
    result = 1
    while grid * result < 4 * units and K div (64 * result * 2) >= 4 and result < 16:
      result *= 2
  result = max(1, min(result, min(16, K div 64)))


# Off until the caller has checked them against the generic cooperative kernel
# on this device (vulkanSet*GemmTiled); MoltenVK cannot run either locally.
var gFp4TiledEnabled = false
var gFp4TiledLogged = false
var gF32TiledEnabled = false
var gF32TiledLogged = false

proc vulkanSetFp4GemmTiled*(enabled: bool) = gFp4TiledEnabled = enabled
proc vulkanSetF32GemmTiled*(enabled: bool) = gF32TiledEnabled = enabled

proc vulkanCooperativeGemmEnabled*(): bool =
  ## True when prefill GEMMs will take a cooperative-matrix route, so packed
  ## scalar fallback kernels need not be compiled ahead of time.
  ensureInit()
  gFp8CoopEnabled or (gCoopEnabled and gDeviceCaps.cooperativeMatrix)

proc vulkanCoopPrewarmJobs*(): seq[tuple[name, source: string]] =
  ## Cooperative GEMM shaders for vulkanJitPrewarm. They are otherwise compiled
  ## serially by glslc on the first prefill dispatch that needs them.
  ensureInit()
  if gFp8CoopEnabled: result.add Fp4KmFp8Shader
  if gCoopEnabled and gDeviceCaps.cooperativeMatrix:
    result.add coopMatmulShader(0)
    result.add coopMatmulShader(1)
    if tiledCoopSupported():
      result.add tiledMatmulShader(1)
      result.add tiledMatmulShader(0)
      result.add tiledMatmulShader(2)
      result.add SplitKReduceShader
    result.add coopMatmulShader(2)

proc tryCooperativeMatmul(a, w, c, lut, zeros: VulkanBufferHandle;
                         M, K, N, weightFormat: int; groupSize = 0;
                         groupsPerRow = 0; wOffset = 0): bool =
  ## Inference GEMM only. GEMV keeps its bandwidth-oriented scalar kernel.
  ## wOffset: byte position of asymmetric Q4 weights (format 2) in w.
  if not gCoopEnabled or not gDeviceCaps.cooperativeMatrix or not gSessionActive or
      M < coopMinRows() or K < 64 or N < 16: return false
  if wOffset < 0 or (wOffset != 0 and weightFormat != 2): return false
  if a.buf == nil or w.buf == nil or c.buf == nil: return false
  if a.sizeBytes < uint64(M) * uint64(K) * 4 or c.sizeBytes < uint64(M) * uint64(N) * 4:
    return false
  let weightBytes = case weightFormat
    of 1: N * (K div 2) + N * (K div 32)
    of 2: N * ((K + 1) div 2)
    else: N * K * 4
  if weightBytes mod 4 != 0 or w.sizeBytes < uint64(wOffset + weightBytes): return false
  if weightFormat == 1 and (K mod 32 != 0 or lut.sizeBytes < 272 * 4): return false
  if weightFormat == 2 and (groupsPerRow <= 0 or
      lut.sizeBytes < uint64(N * groupsPerRow * 4) or
      zeros.sizeBytes < uint64(N * groupsPerRow * 4)): return false
  let (kn, source) = coopMatmulShader(weightFormat)
  var spv: string
  try:
    spv = compileVulkanJitShader(kn, source)
  except CatchableError as e:
    gCoopEnabled = false
    stderr.writeLine "[vulkan] cooperative matrix compiler unavailable; scalar path: " & e.msg
    return false
  var pc = [uint32(M), uint32(K), uint32(N), uint32(groupSize), uint32(groupsPerRow),
    uint32(wOffset)]
  dispatchGeneric(kn, [a, w, c, lut, zeros], addr pc[0], 24,
    uint32((N + 15) div 16), uint32((M + 15) div 16), 1,
    gDeviceCaps.subgroupSize, spv)

proc tryFp4KmFp8Cooperative(a, w, c, lut: VulkanBufferHandle;
                            M, K, N: int): bool =
  ## Packed FP4-KM -> E4M3 is converted only for the current 16x16 MMA tile.
  ## No full-size dequantized weight buffer is created.
  if not gFp8CoopEnabled or not gSessionActive or M < coopMinRows() or K < 64 or N < 16:
    return false
  if a.buf == nil or w.buf == nil or c.buf == nil or lut.buf == nil: return false
  if K mod 32 != 0 or w.sizeBytes < uint64(N * (K div 2) + N * (K div 32)) or
      a.sizeBytes < uint64(M * K * 4) or c.sizeBytes < uint64(M * N * 4) or
      lut.sizeBytes < 272'u64 * 4: return false
  const kernelName = Fp4KmFp8Shader.name
  var spv: string
  try:
    spv = compileVulkanJitShader(kernelName, Fp4KmFp8Shader.source)
  except CatchableError as e:
    gFp8CoopEnabled = false
    stderr.writeLine "[vulkan] FP8 cooperative shader unavailable; use FP16/packed fallback: " & e.msg
    return false
  var pc = [uint32(M), uint32(K), uint32(N), 0'u32, 0'u32]
  result = dispatchGeneric(kernelName, [a, w, c, lut, lut], addr pc[0], 20,
    uint32((N + 15) div 16), uint32((M + 15) div 16), 1,
    gDeviceCaps.subgroupSize, spv)

proc tryTiledCooperative(a, w, c, lut: VulkanBufferHandle; M, K, N, weightFormat: int;
                         zeros = VulkanBufferHandle(); groupSize = 0; groupsPerRow = 0;
                         wOffset = 0): bool =
  ## matmul_fp4km_tiled: weights converted once per 64 output rows instead of
  ## once per 16x16 tile. False when the device/shape does not fit its contract.
  ## Format 2 (asymmetric Q4): lut = scales, zeros, [N, groupsPerRow] each.
  if not tiledCoopSupported() or not gSessionActive or
      M < coopMinRows() or K mod 64 != 0 or N < 16: return false
  if a.buf == nil or w.buf == nil or c.buf == nil or lut.buf == nil: return false
  let weightBytes = case weightFormat
    of 1: N * (K div 2) + N * (K div 32)
    of 2: N * (K div 2)
    else: N * K * 4
  if wOffset < 0 or (wOffset != 0 and weightFormat != 2) or wOffset mod 16 != 0: return false
  if w.sizeBytes < uint64(wOffset + weightBytes) or a.sizeBytes < uint64(M * K * 4) or
      c.sizeBytes < uint64(M * N * 4) or
      (weightFormat == 1 and lut.sizeBytes < 272'u64 * 4): return false
  if weightFormat == 2 and (zeros.buf == nil or groupsPerRow <= 0 or groupSize mod 2 != 0 or
      lut.sizeBytes < uint64(N * groupsPerRow * 4) or
      zeros.sizeBytes < uint64(N * groupsPerRow * 4)): return false
  let (kn, source) = tiledMatmulShader(weightFormat)
  var spv: string
  try:
    spv = compileVulkanJitShader(kn, source)
  except CatchableError as e:
    case weightFormat
    of 1: gFp4TiledEnabled = false
    of 0: gF32TiledEnabled = false
    else: discard   # Q4: its autotune drops the plan
    stderr.writeLine "[vulkan] tiled cooperative shader unavailable: " & e.msg
    return false
  # Split K across workgroup z when M x N alone leaves SMs/CUs idle; the
  # slices write partial sums that one reduce pass adds into c.
  let split = tiledSplitK(M, K, N)
  let kPer = ((K div 64 + split - 1) div split) * 64
  let slices = (K + kPer - 1) div kPer
  var dst = c
  var reduceSpv: string
  if slices > 1:
    try:
      reduceSpv = compileVulkanJitShader(SplitKReduceShader.name, SplitKReduceShader.source)
    except CatchableError as e:
      stderr.writeLine "[vulkan] split-K reduce shader unavailable: " & e.msg
      return false
    let need = uint64(slices * M * N) * 4
    if gSplitKScratch.buf == nil or gSplitKScratch.sizeBytes < need:
      vulkanSessionFree(gSplitKScratch)   # deferred while this session still uses it
      gSplitKScratch = vulkanSessionAllocScratch(slices * M * N * 3 div 2)
    dst = gSplitKScratch
  let threads = 4 * gDeviceCaps.subgroupSize
  let gridX = uint32((N + 63) div 64)
  let gridY = uint32((M + 63) div 64)
  let logKey = $weightFormat & "_" & $K & "x" & $N & "_m" & $M
  if logKey notin gSplitKLogged:
    gSplitKLogged[logKey] = true
    if getEnv("BYBY_VULKAN_SPLIT_K_VERBOSE", "0") == "1" or gSplitKLogged.len <= 8:
      stderr.writeLine "[vulkan] tiled tensor-core GEMM " & $K & "x" & $N & " M=" & $M &
        ": " & $(gridX * gridY) & " workgroups x split-K " & $slices & " = " &
        $(int(gridX * gridY) * slices) & " on " &
        (if gDeviceCaps.computeUnits > 0: $gDeviceCaps.computeUnits else: "?") & " SM/CU"
  var pc = [uint32(M), uint32(K), uint32(N), uint32(groupSize), uint32(groupsPerRow),
    uint32(wOffset), uint32(if slices > 1: kPer else: 0)]
  result = if weightFormat == 2:
      dispatchGeneric(kn, [a, w, dst, lut, zeros], addr pc[0], 28,
        gridX, gridY, uint32(slices), threads, spv)
    else:
      dispatchGeneric(kn, [a, w, dst, lut], addr pc[0], 28,
        gridX, gridY, uint32(slices), threads, spv)
  if result and slices > 1:
    var rp = [uint32(M * N), uint32(slices)]
    result = dispatchGeneric(SplitKReduceShader.name, [gSplitKScratch, c], addr rp[0], 8,
      uint32((M * N + 255) div 256), 1, 1, 256, reduceSpv)

proc vulkanSessionFp4GemmTiledEnc*(a, w, lut: VulkanBufferHandle;
                                   c: var VulkanBufferHandle; M, K, N: int): bool =
  if c.buf == nil: c = vulkanSessionAllocScratch(M * N)
  tryTiledCooperative(a, w, c, lut, M, K, N, 1)

proc vulkanSessionMatmulNTTiledEnc*(a, w: VulkanBufferHandle;
                                    c: var VulkanBufferHandle; M, K, N: int): bool =
  ## FP32 weights [N,K], FP16 MMA with FP32 accumulation (inference only).
  if c.buf == nil: c = vulkanSessionAllocScratch(M * N)
  tryTiledCooperative(a, w, c, w, M, K, N, 0)

proc vulkanSessionFp4GemmEnc*(a, w, lut: VulkanBufferHandle;
                              c: var VulkanBufferHandle; M, K, N: int): bool =
  if c.buf == nil: c = vulkanSessionAllocScratch(M * N)
  if tryFp4KmFp8Cooperative(a, w, c, lut, M, K, N):
    if not gFp8Fp4Logged:
      stderr.writeLine "[vulkan] FP4-KM prefill route=E4M3 FP8 tile decode + FP32 accumulation"
      gFp8Fp4Logged = true
    return true
  if gFp4TiledEnabled and vulkanSessionFp4GemmTiledEnc(a, w, lut, c, M, K, N):
    if not gFp4TiledLogged:
      stderr.writeLine "[vulkan] FP4-KM prefill route=tiled FP16 decode (64x64/workgroup) + FP32 accumulation"
      gFp4TiledLogged = true
    return true
  result = tryCooperativeMatmul(a, w, c, lut, lut, M, K, N, 1)
  if result and not gFp16Fp4Logged:
    stderr.writeLine "[vulkan] FP4-KM prefill route=FP16 tile decode + FP32 accumulation " &
      "(native FP8 cooperative matrix unavailable/disabled)"
    gFp16Fp4Logged = true

proc vulkanSessionMatmulNTAcceleratedEnc*(a, w: VulkanBufferHandle;
    c: var VulkanBufferHandle; M, K, N: int): bool =
  ## Return false when unsupported, so the caller retains exact FP32 GEMM.
  if gF32TiledEnabled and tryTiledCooperative(a, w, c, w, M, K, N, 0):
    if not gF32TiledLogged:
      stderr.writeLine "[vulkan] FP32 prefill GEMM route=tiled FP16 (64x64/workgroup) + FP32 accumulation"
      gF32TiledLogged = true
    return true
  tryCooperativeMatmul(a, w, c, w, w, M, K, N, 0)

proc vulkanSessionRopeEnc*(x: var VulkanBufferHandle, cosT, sinT: VulkanBufferHandle, Hn, D, T: int): bool =
  var pc = [int32(Hn), int32(D), int32(T)]
  let halfD = uint32(D div 2)
  dispatchGeneric("rope", [x, cosT, sinT], addr pc[0], 12,
                  (halfD + 63) div 64, uint32(Hn), uint32(T))

proc vulkanSessionSoftmaxEnc*(x: VulkanBufferHandle, y: var VulkanBufferHandle, rows, cols: int): bool =
  var pc = [int32(cols)]
  dispatchGeneric("softmax", [x, y], addr pc[0], 4, uint32(rows), 1, 1)

proc vulkanSessionRmsnormEnc*(x, gamma: VulkanBufferHandle, y: var VulkanBufferHandle, rows, cols: int, eps: float32): bool =
  var pc = (cols: int32(cols), eps: eps)
  dispatchGeneric("rmsnorm", [x, gamma, y], addr pc, 8, uint32(rows), 1, 1)

proc vulkanSessionLayernormEnc*(x, gamma, beta: VulkanBufferHandle, y: var VulkanBufferHandle, rows, cols: int, eps: float32): bool =
  var pc = (cols: int32(cols), eps: eps)
  dispatchGeneric("layernorm", [x, gamma, beta, y], addr pc, 8, uint32(rows), 1, 1)

proc vulkanSessionMatmulEnc*(a, b: VulkanBufferHandle, c: var VulkanBufferHandle, M, K, N: int): bool =
  var pc = [int32(M), int32(K), int32(N)]
  # ADDED: m==1 = GEMV - the 16x16 tile below dispatches groupY=ceil(1/16)=1,
  # i.e. only 1/16 of the invocations per workgroup produce a correct result (the same bug fixed
  # in Metal matmul_kernel_t8, explained in metal_shim.m).
  # Uses the matmul_gemv.comp shader (subgroupAdd, no barrier) when M=1.
  if M == 1:
    dispatchGeneric("matmul_gemv", [a, b, c], addr pc[0], 12, uint32(N), 1, 1)
  else:
    const TILE = 16'u32
    dispatchGeneric("matmul", [a, b, c], addr pc[0], 12,
                    (uint32(N) + TILE - 1) div TILE, (uint32(M) + TILE - 1) div TILE, 1)

proc vulkanSessionMatmulNTEnc*(a, b: VulkanBufferHandle, c: var VulkanBufferHandle, M, K, N: int;
                               bOffsetElems = 0): bool =
  ## C(M x N) = A(M x K) @ B(N x K)^T - uses the separate matmul_nt.comp shader
  ## (reads B in its natural [N,K] layout, no CPU transpose first).
  ## bOffsetElems: B starts that many floats into its buffer.
  var pc = [int32(M), int32(K), int32(N), int32(bOffsetElems)]
  # ADDED: m==1 GEMV, see the explanation in vulkanSessionMatmulEnc above.
  if M == 1:
    dispatchGeneric("matmul_nt_gemv", [a, b, c], addr pc[0], 16, uint32(N), 1, 1)
  else:
    const TILE = 16'u32
    dispatchGeneric("matmul_nt", [a, b, c], addr pc[0], 16,
                    (uint32(N) + TILE - 1) div TILE, (uint32(M) + TILE - 1) div TILE, 1)

include q4_autotune

proc vulkanSessionAttentionFusedEnc*(q, k, v: VulkanBufferHandle, o, sMatrix: var VulkanBufferHandle,
                                      B, H, S, D: int, scale: float32): bool =
  var pc = (B: int32(B), H: int32(H), S: int32(S), D: int32(D), scale: scale)
  dispatchGeneric("attention_fused", [q, k, v, o, sMatrix], addr pc, 20,
                  (uint32(B * H) + 31) div 32, (uint32(S) + 7) div 8, 1)

proc vulkanSessionAttentionFusedBackwardEnc*(q, k, v, sMatrix, dy: VulkanBufferHandle,
                                              dq, dk, dv: var VulkanBufferHandle,
                                              B, H, S, D: int, scale: float32): bool =
  ## Mirror of attention_fused_backward_kernel (Metal) through kernels/
  ## attention_fused_backward.comp - see the .comp file for details. Dispatches
  ## 1 thread/(bh,ti): groupCount=(B*H,S,1) with local_size=(1,1,1) in the
  ## shader, SAME as Metal's dispatchThreads (no group optimisation).
  var pc = (B: int32(B), H: int32(H), S: int32(S), D: int32(D), scale: scale)
  dispatchGeneric("attention_fused_backward", [q, k, v, sMatrix, dy, dq, dk, dv], addr pc, 20,
                  uint32(B * H), uint32(S), 1)

proc vulkanSessionAttentionDecodeGqaEnc*(q, k, v: VulkanBufferHandle, o, scores: var VulkanBufferHandle,
                                          H, S, D: int, scale: float32): bool =
  ## Mirror of attention_decode_gqa_kernel.cu (CUDA)/Metal - 1 workgroup (128
  ## threads)/query head. WARNING: mirrored from the CUDA version that HAD NOT RUN ON
  ## REAL HARDWARE when written - verify with BYBY_VERIFY_ATTN_DECODE=1 (see
  ## nim_inference.nim) before trusting its results on a real machine.
  var pc = (H: int32(H), S: int32(S), D: int32(D), scale: scale)
  dispatchGeneric("attention_decode_gqa", [q, k, v, o, scores], addr pc, 16,
                  uint32(H), 1, 1)

const attnAllHeadsGlsl = staticRead(currentSourcePath().parentDir() / "kernels" / "attention_decode_all_heads.comp")
  ## The GLSL is embedded when Nim compiles -> without a .spv file it is compiled
  ## with glslc at run time (cached in the temp directory), no manual build needed.
var gAttnAllHeadsSpv = ""        ## path of the chosen .spv ("" = not chosen yet)
var gAttnAllHeadsBroken = false  ## true = no .spv and glslc failed -> the caller uses the old path

proc vulkanSessionAttentionDecodeAllHeadsEnc*(q, kAll, vAll: VulkanBufferHandle,
                                               o: var VulkanBufferHandle,
                                               H, HKV, D, S, maxSeqLen: int): bool =
  ## FUSED attention decode: ALL H heads in EXACTLY one dispatch (1 workgroup/head).
  ## Mirror of attention_decode_all_heads_kernel (Metal). Returns false (does NOT raise) when it
  ## cannot run -> the caller falls back to the per-head-group loop.
  if not gSessionActive or gAttnAllHeadsBroken: return false
  if q.buf == nil or kAll.buf == nil or vAll.buf == nil or o.buf == nil: return false
  if H <= 0 or HKV <= 0 or D <= 0 or D > 256 or S <= 0 or S > maxSeqLen: return false
  if gAttnAllHeadsSpv.len == 0:
    let prebuilt = currentSourcePath().parentDir() / "kernels" / "attention_decode_all_heads.spv"
    if fileExists(prebuilt):
      gAttnAllHeadsSpv = prebuilt
    else:
      try:
        gAttnAllHeadsSpv = compileVulkanJitShader("attention_decode_all_heads", attnAllHeadsGlsl)
      except CatchableError as e:
        stderr.writeLine "[vulkan] attention_decode_all_heads: no .spv and glslc failed (" &
          e.msg.splitLines()[0] & ") -> using the old per-head path"
        gAttnAllHeadsBroken = true
        return false
  var pc = [int32(H), int32(HKV), int32(D), int32(S), int32(maxSeqLen)]
  result = dispatchGeneric("attention_decode_all_heads", [q, kAll, vAll, o], addr pc[0], 20,
                           uint32(H), 1, 1, 0, gAttnAllHeadsSpv)

# ---- Prefill flash attention -------------------------------------------------
# attention_prefill_flash.glsl (scalar FP32) and attention_prefill_flash_coop.glsl
# (cooperative matrix): causal GQA attention of T new rows over the resident KV
# cache in one dispatch, with an online softmax, instead of reading Q/K/V back to
# the CPU, expanding K/V per head and running attention_fused.

const KvScatterShader = (name: "kv_cache_scatter", source: """#version 450
// src [T, HKV, D] (the K or V projection) -> dst [HKV, maxSeqLen, D] at pos.
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) readonly buffer Src { float src[]; };
layout(std430, binding = 1) buffer Dst { float dst[]; };
layout(push_constant) uniform Params { uint T, HKV, D, pos, maxSeqLen; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint total = p.T * p.HKV * p.D;
    if (i >= total) return;
    uint d = i % p.D;
    uint th = i / p.D;
    uint h = th % p.HKV;
    uint t = th / p.HKV;
    dst[(h * p.maxSeqLen + p.pos + t) * p.D + d] = src[i];
}
""")

const RopeKvShader = (name: "rope_kv", source: """#version 450
layout(local_size_x_id = 0) in;
layout(std430, binding = 0) buffer QBuf { float q[]; };
layout(std430, binding = 1) readonly buffer KBuf { float k[]; };
layout(std430, binding = 2) readonly buffer VBuf { float v[]; };
layout(std430, binding = 3) readonly buffer CosBuf { float cosTab[]; };
layout(std430, binding = 4) readonly buffer SinBuf { float sinTab[]; };
layout(std430, binding = 5) buffer KAllBuf { float kAll[]; };
layout(std430, binding = 6) buffer VAllBuf { float vAll[]; };
layout(push_constant) uniform Params {
    uint H, HKV, D, maxSeqLen, pos;
    float scale;
} p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    uint halfD = p.D / 2;
    uint totalQ = p.H * halfD;
    uint totalK = p.HKV * halfD;
    uint totalV = p.HKV * p.D;
    uint cacheBase = p.pos * p.D;
    if (i < totalQ) {
        uint h = i / halfD, d = i % halfD;
        uint base = h * p.D;
        float x1 = q[base + d], x2 = q[base + d + halfD];
        float c = cosTab[cacheBase / p.D * halfD + d];
        float s = sinTab[cacheBase / p.D * halfD + d];
        q[base + d] = (x1 * c - x2 * s) * p.scale;
        q[base + d + halfD] = (x2 * c + x1 * s) * p.scale;
        return;
    }
    i -= totalQ;
    if (i < totalK) {
        uint h = i / halfD, d = i % halfD;
        uint srcBase = h * p.D;
        uint dstBase = h * p.maxSeqLen * p.D + cacheBase;
        float x1 = k[srcBase + d], x2 = k[srcBase + d + halfD];
        float c = cosTab[cacheBase / p.D * halfD + d];
        float s = sinTab[cacheBase / p.D * halfD + d];
        kAll[dstBase + d] = x1 * c - x2 * s;
        kAll[dstBase + d + halfD] = x2 * c + x1 * s;
        return;
    }
    i -= totalK;
    if (i < totalV) {
        uint h = i / p.D, d = i % p.D;
        vAll[h * p.maxSeqLen * p.D + cacheBase + d] = v[i];
    }
}
""")

const flashScalarGlsl = staticRead("kernels/attention_prefill_flash.glsl")
const flashFp16Glsl = staticRead("kernels/attention_prefill_flash_fp16.glsl")
const flashCoopGlsl = staticRead("kernels/attention_prefill_flash_coop.glsl")
const FlashScalarMaxD = 128
const FlashScalarSharedBytes = (16 * 128 + 16 * 129 + 16 * 128 + 16 * 17) * 4

var gFlashKind = initTable[int, string]()   ## head dim -> "coop" / "scalar" / ""

proc flashCoopShader(D: int): tuple[name, source: string] =
  ("attention_prefill_flash_coop_d" & $D,
   "#version 450\n#define HEAD_DIM " & $D & "\n" & flashCoopGlsl.replace("#version 450\n", ""))

proc flashCoopSupported(D: int): bool =
  gCoopEnabled and gDeviceCaps.cooperativeMatrix and D mod 16 == 0 and D <= 128 and
    gDeviceCaps.subgroupSize >= 16 and gDeviceCaps.maxSharedBytes >= 16 * 1024

proc flashScalarSupported(D: int): bool =
  D > 0 and D <= FlashScalarMaxD and gDeviceCaps.maxThreads >= 128 and
    gDeviceCaps.maxSharedBytes >= FlashScalarSharedBytes

proc vulkanSessionKvScatterEnc*(src, dst: VulkanBufferHandle; T, HKV, D, pos, maxSeqLen: int): bool =
  ## dst[h, pos + t, :] = src[t, h, :] for the T new positions (KV cache append).
  if not gSessionActive or src.buf == nil or dst.buf == nil: return false
  if T <= 0 or pos < 0 or pos + T > maxSeqLen: return false
  if dst.sizeBytes < uint64(HKV * maxSeqLen * D) * 4: return false
  var spv: string
  try:
    spv = compileVulkanJitShader(KvScatterShader.name, KvScatterShader.source)
  except CatchableError as e:
    stderr.writeLine "[vulkan] kv_cache_scatter unavailable: " & e.msg.splitLines()[0]
    return false
  var pc = [uint32(T), uint32(HKV), uint32(D), uint32(pos), uint32(maxSeqLen)]
  let total = T * HKV * D
  dispatchGeneric(KvScatterShader.name, [src, dst], addr pc[0], 20,
    uint32((total + 255) div 256), 1, 1, 256, spv)

proc vulkanSessionKvAppendEnc*(k, v: VulkanBufferHandle;
    kAll, vAll: var VulkanBufferHandle; HKV, D, maxSeqLen, pos: int): bool =
  ## Append both FP32 KV tensors into the combined resident caches. This
  ## replaces four per-head vkCmdCopyBuffer operations per KV head (K/V into
  ## both per-head and combined caches) with two contiguous GPU scatters.
  if not gSessionActive or k.buf == nil or v.buf == nil or
      kAll.buf == nil or vAll.buf == nil or HKV <= 0 or D <= 0:
    return false
  let need = uint64(HKV * maxSeqLen * D) * 4
  if kAll.sizeBytes < need or vAll.sizeBytes < need: return false
  let kOk = vulkanSessionKvScatterEnc(k, kAll, 1, HKV, D, pos, maxSeqLen)
  let vOk = vulkanSessionKvScatterEnc(v, vAll, 1, HKV, D, pos, maxSeqLen)
  kOk and vOk

proc vulkanSessionRopeKvEnc*(q, k, v, cosT, sinT: VulkanBufferHandle;
    kAll, vAll: var VulkanBufferHandle; H, HKV, D, maxSeqLen, pos: int;
    scale: float32): bool =
  ## Fuse decode RoPE(Q/K), Q scaling, and K/V append into one dispatch.
  ## This mirrors Metal's rope_kv kernel and avoids host-built per-token
  ## cosine/sine slices plus five intermediate dispatches on the scalar path.
  if not gSessionActive or q.buf == nil or k.buf == nil or v.buf == nil or
      cosT.buf == nil or sinT.buf == nil or kAll.buf == nil or vAll.buf == nil:
    return false
  if H <= 0 or HKV <= 0 or D <= 0 or D mod 2 != 0 or pos < 0 or pos >= maxSeqLen:
    return false
  if q.sizeBytes < uint64(H * D) * 4 or k.sizeBytes < uint64(HKV * D) * 4 or
      v.sizeBytes < uint64(HKV * D) * 4 or cosT.sizeBytes < uint64(maxSeqLen * D div 2) * 4 or
      sinT.sizeBytes < uint64(maxSeqLen * D div 2) * 4:
    return false
  let cacheNeed = uint64(HKV * maxSeqLen * D) * 4
  if kAll.sizeBytes < cacheNeed or vAll.sizeBytes < cacheNeed: return false
  if getEnv("BYBY_FUSE_ROPE", "1") == "0": return false
  var spv: string
  try:
    spv = compileVulkanJitShader(RopeKvShader.name, RopeKvShader.source)
  except CatchableError as e:
    stderr.writeLine "[vulkan] rope_kv unavailable: " & e.msg.splitLines()[0]
    return false
  var pc = (H: uint32(H), HKV: uint32(HKV), D: uint32(D),
            maxSeqLen: uint32(maxSeqLen), pos: uint32(pos), scale: scale)
  let total = H * (D div 2) + HKV * (D div 2) + HKV * D
  dispatchGeneric(RopeKvShader.name, [q, k, v, cosT, sinT, kAll, vAll],
    addr pc, 24, uint32((total + 255) div 256), 1, 1, 256, spv)

proc encodeFlash(kind: string; q, kAll, vAll, o: VulkanBufferHandle;
                 T, H, HKV, D, pos, maxSeqLen: int; scale: float32): bool =
  var pc: array[7, uint32]
  pc[0] = uint32(T); pc[1] = uint32(H); pc[2] = uint32(HKV); pc[3] = uint32(D)
  pc[4] = uint32(pos); pc[5] = uint32(maxSeqLen)
  let scaleLog2 = scale * 1.4426950408889634'f32
  copyMem(addr pc[6], unsafeAddr scaleLog2, 4)
  let groupsX = uint32((T + 15) div 16)
  if kind == "coop":
    let (name, source) = flashCoopShader(D)
    let spv = compileVulkanJitShader(name, source)
    dispatchGeneric(name, [q, kAll, vAll, o], addr pc[0], 28, groupsX, uint32(H), 1,
      gDeviceCaps.subgroupSize, spv)
  elif kind == "fp16":
    let spv = compileVulkanJitShader("attention_prefill_flash_fp16", flashFp16Glsl)
    dispatchGeneric("attention_prefill_flash_fp16", [q, kAll, vAll, o], addr pc[0], 28,
      groupsX, uint32(H), 1, 0, spv)
  else:
    let spv = compileVulkanJitShader("attention_prefill_flash", flashScalarGlsl)
    dispatchGeneric("attention_prefill_flash", [q, kAll, vAll, o], addr pc[0], 28,
      groupsX, uint32(H), 1, 0, spv)

proc flashReference(q, k, v: seq[float32]; T, H, HKV, D, pos, maxSeqLen: int;
                    scale: float32): seq[float32] =
  result = newSeq[float32](T * H * D)
  let rep = max(1, H div HKV)
  var sc = newSeq[float64](pos + T)
  for t in 0 ..< T:
    for h in 0 ..< H:
      let hkv = min(h div rep, HKV - 1)
      let n = pos + t + 1
      var mx = -1e300
      for key in 0 ..< n:
        var s = 0.0
        for d in 0 ..< D:
          s += float64(q[(t * H + h) * D + d]) * float64(k[(hkv * maxSeqLen + key) * D + d])
        sc[key] = s * float64(scale)
        mx = max(mx, sc[key])
      var total = 0.0
      for key in 0 ..< n:
        sc[key] = exp(sc[key] - mx)
        total += sc[key]
      for d in 0 ..< D:
        var acc = 0.0
        for key in 0 ..< n: acc += sc[key] * float64(v[(hkv * maxSeqLen + key) * D + d])
        result[(t * H + h) * D + d] = float32(acc / total)

proc flashParity(kind: string; D: int): tuple[ok: bool, err: float] =
  ## Runs the kernel on random data (GQA, a continued prompt, NaN past the
  ## filled cache to prove the mask) against an FP64 reference.
  const T = 37
  const H = 4
  const HKV = 2
  const pos = 11
  const maxSeqLen = 64
  var seed = 0x9E3779B9'u32
  proc rnd(): float32 =
    seed = seed xor (seed shl 13); seed = seed xor (seed shr 17); seed = seed xor (seed shl 5)
    float32(seed and 0xFFFFFF) / float32(0x800000) - 1.0'f32
  var q = newSeq[float32](T * H * D)
  for i in 0 ..< q.len: q[i] = rnd() * 3.0'f32   # peaky rows exercise the rescale
  var k = newSeq[float32](HKV * maxSeqLen * D)
  var v = newSeq[float32](HKV * maxSeqLen * D)
  for h in 0 ..< HKV:
    for key in 0 ..< maxSeqLen:
      for d in 0 ..< D:
        let i = (h * maxSeqLen + key) * D + d
        if key < pos + T:
          k[i] = rnd(); v[i] = rnd()
        else:
          k[i] = NaN; v[i] = NaN
  let scale = 1.0'f32 / sqrt(float32(D))
  let expected = flashReference(q, k, v, T, H, HKV, D, pos, maxSeqLen, scale)
  if not vulkanSessionBegin(): return (false, Inf)
  var hq = vulkanSessionUpload(q)
  var hk = vulkanSessionUpload(k)
  var hv = vulkanSessionUpload(v)
  var ho = vulkanSessionAllocScratch(T * H * D)
  var dispatched = false
  try:
    dispatched = encodeFlash(kind, hq, hk, hv, ho, T, H, HKV, D, pos, maxSeqLen, scale)
  except CatchableError as e:
    stderr.writeLine "[vulkan] flash attention (" & kind & ") unavailable: " & e.msg.splitLines()[0]
  let ended = vulkanSessionEnd()
  if dispatched and ended:
    let got = vulkanSessionRead(ho, T * H * D)
    result = (true, 0.0)
    for i in 0 ..< got.len:
      let e = abs(float(got[i]) - float(expected[i]))
      if e != e or e > result.err: result.err = (if e != e: Inf else: e)
    let tol = if kind == "coop": 2e-2 elif kind == "fp16": 3e-3 else: 1e-4
    result.ok = result.err <= tol
  else:
    result = (false, Inf)
  vulkanSessionFree(hq); vulkanSessionFree(hk); vulkanSessionFree(hv); vulkanSessionFree(ho)

proc vulkanFlashAttentionPrepare*(D: int): string =
  ## Picks and checks the prefill flash-attention kernel for head dim D, outside
  ## any session: "coop" (tensor cores), "scalar", or "" (none passed parity;
  ## the caller keeps its old attention path). The order is tensor core, scalar
  ## FP16, then scalar FP32; BYBY_FLASH_ATTENTION=scalar forces the FP32 path.
  ensureInit()
  if D in gFlashKind: return gFlashKind[D]
  let mode = getEnv("BYBY_FLASH_ATTENTION", "auto")
  result = ""
  if mode == "fp16" and not gDeviceCaps.float16:
    stderr.writeLine "[vulkan] FP16 flash attention unavailable: Vulkan shaderFloat16 feature is not exposed; using FP32"
  if mode != "0" and not gSessionActive:
    var candidates: seq[string]
    if mode != "scalar" and mode != "fp16" and flashCoopSupported(D): candidates.add "coop"
    if mode != "scalar" and gDeviceCaps.float16 and D > 0 and D <= FlashScalarMaxD and
        gDeviceCaps.maxThreads >= 128 and gDeviceCaps.maxSharedBytes >= FlashScalarSharedBytes div 2:
      candidates.add "fp16"
    if flashScalarSupported(D): candidates.add "scalar"
    for kind in candidates:
      let (ok, err) = flashParity(kind, D)
      if ok:
        result = kind
        break
      stderr.writeLine "[vulkan] flash attention (" & kind & ", D=" & $D &
        ") failed parity (max err " & $err & "), skipping it"
  gFlashKind[D] = result
  if result.len > 0:
    stderr.writeLine "[vulkan] prefill flash attention D=" & $D & ": " &
      (if result == "coop": "tensor cores (KHR_cooperative_matrix FP16/FP32, lazy rescale)"
       elif result == "fp16": "scalar FP16 operands with FP32 softmax/accumulation"
       else: "scalar FP32")

proc vulkanSessionFlashAttentionPrefillEnc*(q, kAll, vAll: VulkanBufferHandle;
                                            o: var VulkanBufferHandle;
                                            T, H, HKV, D, pos, maxSeqLen: int;
                                            scale: float32): bool =
  ## o[T, H, D] = causal softmax(q K^T * scale) V over cache positions
  ## 0 .. pos + T - 1. vulkanFlashAttentionPrepare(D) must have picked a kernel.
  if not gSessionActive: return false
  let kind = gFlashKind.getOrDefault(D, "")
  if kind.len == 0: return false
  if q.buf == nil or kAll.buf == nil or vAll.buf == nil: return false
  if T <= 0 or H <= 0 or HKV <= 0 or H mod HKV != 0 or pos < 0 or pos + T > maxSeqLen: return false
  if q.sizeBytes < uint64(T * H * D) * 4 or kAll.sizeBytes < uint64(HKV * maxSeqLen * D) * 4 or
      vAll.sizeBytes < uint64(HKV * maxSeqLen * D) * 4: return false
  if o.buf == nil: o = vulkanSessionAllocScratch(T * H * D)
  try:
    result = encodeFlash(kind, q, kAll, vAll, o, T, H, HKV, D, pos, maxSeqLen, scale)
  except CatchableError as e:
    stderr.writeLine "[vulkan] flash attention dispatch failed: " & e.msg.splitLines()[0]
    result = false

# ---- Mamba2 SSD / FLA chunked prefill ----------------------------------------
# mamba2_ssd_fla.glsl: five pipelines (cumsum, FLA intra-chunk, chunk states,
# chunk-state combine, inter-chunk) replacing the sequential/Hillis-Steele SSM
# scan with tensor-core matmuls for long prompts. Everything addresses one
# arena buffer through push-constant offsets (the microVM prefill layout).

const ssdFlaGlsl = staticRead("kernels/mamba2_ssd_fla.glsl")

var gSsdKind = initTable[string, string]()   ## "N_D_Q" -> "coop" / "scalar" / ""

proc ssdShader(kind: string; mode, N, D, Q: int): tuple[name, source: string] =
  let scalar = kind == "scalar" and mode in [1, 2, 4]
  ("mamba2_ssd_fla_" & (if scalar: "s" else: "c") & "_m" & $mode & "_n" & $N &
     "_d" & $D & "_q" & $Q,
   "#version 450\n#define SSD_MODE " & $mode & "\n#define SSD_SCALAR " &
     (if scalar: "1" else: "0") & "\n#define SSD_N " & $N &
     "\n#define SSD_D " & $D & "\n#define SSD_Q " & $Q & "\n" &
     ssdFlaGlsl.replace("#version 450\n", ""))

proc ssdCoopSupported(N, D, Q: int): bool =
  gCoopEnabled and gDeviceCaps.cooperativeMatrix and
    gDeviceCaps.subgroupSize >= 16 and gDeviceCaps.maxThreads >= 256 and
    gDeviceCaps.maxSharedBytes >= 16 * 1024 and
    N mod 16 == 0 and N in 16 .. 256 and D mod 16 == 0 and D in 16 .. 128 and
    Q mod 16 == 0 and (Q and (Q - 1)) == 0 and Q in 16 .. 256

proc ssdScalarSupported(N, D, Q: int): bool =
  gDeviceCaps.maxThreads >= 256 and
    gDeviceCaps.maxSharedBytes >= (2 * 16 * (N + 1) + 16 * D + 288) * 4 and
    N mod 16 == 0 and N in 16 .. 256 and D mod 16 == 0 and D in 16 .. 128 and
    Q mod 16 == 0 and (Q and (Q - 1)) == 0 and Q in 16 .. 256

proc encodeSsdFla(kind: string; arena: VulkanBufferHandle;
                  T, H, G, N, D, Q, nC, C, cDim: int;
                  offConv, offDt, offCum, offALog, offDSkip,
                  offYPre, offStates, offSloc: int; cont: bool): bool =
  ## Encodes the five SSD dispatches; dispatchGeneric inserts the
  ## write->read barrier each phase needs.
  var pc: array[18, uint32]
  pc[0] = uint32(T); pc[1] = uint32(H); pc[2] = uint32(G); pc[3] = uint32(N)
  pc[4] = uint32(D); pc[5] = uint32(Q); pc[6] = uint32(nC); pc[7] = uint32(C)
  pc[8] = uint32(cDim)
  pc[9] = uint32(offConv); pc[10] = uint32(offDt); pc[11] = uint32(offCum)
  pc[12] = uint32(offALog); pc[13] = uint32(offDSkip); pc[14] = uint32(offYPre)
  pc[15] = uint32(offStates); pc[16] = uint32(offSloc)
  pc[17] = uint32(cont)
  # Cooperative modes run one subgroup per tile; scalar modes 256 lanes.
  let sg = if kind == "coop": int(gDeviceCaps.subgroupSize) else: 256
  let slabs = uint32(Q div 16)
  for mode in [0, 1, 2, 3, 4]:
    let (name, source) = ssdShader(kind, mode, N, D, Q)
    let spv = compileVulkanJitShader(name, source)
    let ok =
      case mode
      of 0: dispatchGeneric(name, [arena], addr pc[0], 72,
              uint32(nC), uint32(H), 1, Q, spv)
      of 1: dispatchGeneric(name, [arena], addr pc[0], 72,
              uint32(nC) * slabs, uint32(H), 1, sg, spv)
      of 2: dispatchGeneric(name, [arena], addr pc[0], 72,
              uint32(nC) * uint32(N div 16), uint32(H), 1, sg, spv)
      of 3: dispatchGeneric(name, [arena], addr pc[0], 72,
              uint32(H), 1, 1, 256, spv)
      else: dispatchGeneric(name, [arena], addr pc[0], 72,
              uint32(nC) * slabs, uint32(H), 1, sg, spv)
    if not ok: return false
  true

proc ssdReference(arena: var seq[float64];
                  T, H, G, N, D, Q, C, cDim: int;
                  offConv, offDt, offALog, offDSkip,
                  offYPre, offStates: int; cont: bool) =
  ## FP64 serial SSM recurrence writing yPre and the final state (states[T]).
  for h in 0 ..< H:
    let g = h div (H div G)
    let A = -exp(arena[offALog + h])
    let dsk = arena[offDSkip + h]
    var s = newSeq[float64](N * D)
    if cont:
      for n in 0 ..< N:
        for d in 0 ..< D:
          s[n * D + d] = arena[offStates + (h * D + d) * N + n]
    for t in 0 ..< T:
      let row = offConv + t * cDim
      let dt = arena[offDt + t * H + h]
      let decay = exp(dt * A)
      for n in 0 ..< N:
        let b = arena[row + C + g * N + n] * dt
        for d in 0 ..< D:
          s[n * D + d] = decay * s[n * D + d] + b * arena[row + h * D + d]
      for d in 0 ..< D:
        var acc = dsk * arena[row + h * D + d]
        for n in 0 ..< N:
          acc += arena[row + C + G * N + g * N + n] * s[n * D + d]
        arena[offYPre + t * C + h * D + d] = acc
    for n in 0 ..< N:
      for d in 0 ..< D:
        arena[offStates + (C + h * D + d) * N + n] = s[n * D + d]

proc ssdParity(kind: string; N, D, Q: int): tuple[ok: bool, err: float] =
  ## Runs all five pipelines end to end on random data (a partial last chunk
  ## and a carried initial state) against the FP64 recurrence.
  let T = 2 * Q - 3
  const H = 2
  const G = 2
  let C = H * D
  let cDim = C + 2 * G * N
  let nC = (T + Q - 1) div Q
  var o = 0
  template take(n: int): int =
    let r = o; o += n; r
  let offConv = take(T * cDim)
  let offDt = take(T * H)
  let offCum = take(T * H)
  let offALog = take(H)
  let offDSkip = take(H)
  let offYPre = take(T * C)
  let offStates = take(2 * C * N)
  let offSloc = take(nC * H * N * D)
  var seed = 0x51ED270B'u32
  proc rnd(): float32 =
    seed = seed xor (seed shl 13); seed = seed xor (seed shr 17); seed = seed xor (seed shl 5)
    float32(seed and 0xFFFFFF) / float32(0x800000) - 1.0'f32
  var arena = newSeq[float32](o)
  for t in 0 ..< T:
    for i in 0 ..< cDim: arena[offConv + t * cDim + i] = rnd()
    for h in 0 ..< H: arena[offDt + t * H + h] = abs(rnd()) * 0.6'f32 + 0.05'f32
  for h in 0 ..< H:
    arena[offALog + h] = rnd() * 0.5'f32
    arena[offDSkip + h] = rnd()
  for i in 0 ..< C * N:
    arena[offStates + i] = rnd()                      # carried initial state
  var refArena = newSeq[float64](o)
  for i in 0 ..< o: refArena[i] = float64(arena[i])
  ssdReference(refArena, T, H, G, N, D, Q, C, cDim,
    offConv, offDt, offALog, offDSkip, offYPre, offStates, true)
  if not vulkanSessionBegin(): return (false, Inf)
  var hArena = vulkanSessionUpload(arena)
  var dispatched = false
  try:
    dispatched = encodeSsdFla(kind, hArena, T, H, G, N, D, Q, nC, C, cDim,
      offConv, offDt, offCum, offALog, offDSkip, offYPre, offStates, offSloc, true)
  except CatchableError as e:
    stderr.writeLine "[vulkan] mamba2 SSD unavailable: " & e.msg.splitLines()[0]
  let ended = vulkanSessionEnd()
  if dispatched and ended:
    let got = vulkanSessionRead(hArena, o)
    result = (true, 0.0)
    for t in 0 ..< T:
      for i in 0 ..< C:
        let e = abs(float(got[offYPre + t * C + i]) - refArena[offYPre + t * C + i])
        if e != e or e > result.err: result.err = (if e != e: Inf else: e)
    for i in 0 ..< C * N:
      let e = abs(float(got[offStates + C * N + i]) -
                  refArena[offStates + C * N + i])
      if e != e or e > result.err: result.err = (if e != e: Inf else: e)
    result.ok = result.err <= (if kind == "coop": 5e-2 else: 1e-3)
  else:
    result = (false, Inf)
  vulkanSessionFree(hArena)

proc vulkanMamba2SsdPrepare*(N, D, Q: int): string =
  ## Picks and parity-checks the SSD pipelines for (d_state, head dim, chunk)
  ## outside any session: "coop" (tensor cores), "scalar" (FP32 FMA tiles) or
  ## "" (unavailable; the caller keeps its scan path). BYBY_SSD=0 disables it,
  ## BYBY_SSD=scalar skips the tensor-core kernels.
  ensureInit()
  let key = $N & "_" & $D & "_" & $Q
  if key in gSsdKind: return gSsdKind[key]
  result = ""
  let mode = getEnv("BYBY_SSD", "auto")
  if mode != "0" and not gSessionActive:
    var candidates: seq[string]
    if mode != "scalar" and ssdCoopSupported(N, D, Q): candidates.add "coop"
    if ssdScalarSupported(N, D, Q): candidates.add "scalar"
    for kind in candidates:
      let (ok, err) = ssdParity(kind, N, D, Q)
      if ok:
        result = kind
        stderr.writeLine "[vulkan] mamba2 SSD prefill N=" & $N & " D=" & $D &
          " Q=" & $Q & ": " &
          (if kind == "coop": "tensor cores (KHR_cooperative_matrix FP16/FP32, "
           else: "scalar FP32 (") & "max err " & $err & ")"
        break
      stderr.writeLine "[vulkan] mamba2 SSD (" & kind & ", N=" & $N & ", D=" & $D &
        ", Q=" & $Q & ") failed parity (max err " & $err & "), skipping it"
  gSsdKind[key] = result

proc vulkanSessionMamba2SsdPrefillEnc*(arena: VulkanBufferHandle;
    T, H, G, N, D, Q, C, cDim: int;
    offConv, offDt, offCum, offALog, offDSkip,
    offYPre, offStates, offSloc: int; cont: bool): bool =
  ## Encodes the whole chunked SSD pass (five dispatches) into the open
  ## session. vulkanMamba2SsdPrepare(N, D, Q) must have picked a kernel.
  if not gSessionActive or arena.buf == nil: return false
  let kind = gSsdKind.getOrDefault($N & "_" & $D & "_" & $Q, "")
  if kind.len == 0: return false
  let nC = (T + Q - 1) div Q
  if T <= 0 or nC > 256 or H mod G != 0 or C != H * D: return false
  try:
    result = encodeSsdFla(kind, arena, T, H, G, N, D, Q, nC, C, cDim,
      offConv, offDt, offCum, offALog, offDSkip, offYPre, offStates, offSloc, cont)
  except CatchableError as e:
    stderr.writeLine "[vulkan] mamba2 SSD dispatch failed: " & e.msg.splitLines()[0]
    result = false

proc vulkanAvailable*(): bool =
  ## Used by detectBackend() in gpubackend.nim - tries load+init and catches errors
  ## safely (does NOT raise) when the machine has no Vulkan/MoltenVK.
  try:
    ensureInit()
    result = true
  except CatchableError:
    result = false

proc vulkanSessionUploadBytes*(data: seq[uint8]): VulkanBufferHandle =
  defer:
    if unlikely(firProbeOn) and data.len > 0:
      probeUploaded(cast[pointer](result.buf), unsafeAddr data[0], data.len)
  ## Raw-bytes variant (vulkanSessionUpload takes only seq[float32]) - used
  ## for packed int4 weights (wq), mirrors openclSessionUploadBytes.
  ensureInit()
  let bytes = uint64(data.len)
  if gScratchDeviceLocal:
    result = acquirePooled(pkDevice, bytes)
    if data.len > 0: stagedUpload(result, unsafeAddr data[0], bytes)
    return
  let (buf, mem) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  if data.len > 0:
    var dataVar = data
    uploadToBuffer(mem, addr dataVar[0], bytes)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: bytes)

proc vulkanSessionActivationEnc*(opCode: int, x: VulkanBufferHandle, y: var VulkanBufferHandle, n: int, p: float32 = 0'f32): bool =
  ## The PC layout must match struct PC {int op; int n; float p;} in
  ## activation.comp (12 bytes) - only 8 bytes were pushed before (p missing), so
  ## opcode 7 (pow) on the session path never got the right exponent.
  var pc: tuple[op, n: int32, p: float32] = (int32(opCode), int32(n), p)
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("activation", [x, y], addr pc, 12, groups, 1, 1)
# =======================================================================
# HIGH-LEVEL ONE-SHOT WRAPPERS - mirror the interface of metal_backend.nim and
# gpubackend.nim to give Vulkan a uniform API. Functions with a
# .comp shader use the GPU through the session encoder; those without their own shader
# still use the CPU fallback according to the caller's policy.

proc vulkanSessionGatherEnc*(src, idx: VulkanBufferHandle, dst: var VulkanBufferHandle, n: int): bool =
  var pc = [int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("gather", [src, idx, dst], addr pc[0], 4, groups, 1, 1)

proc vulkanSessionArgmaxIndexEnc*(x: VulkanBufferHandle, outVal, outIdx: var VulkanBufferHandle,
                                   outer, dimSize, inner: int): bool =
  let pairs = outer * inner
  var pc = [int32(outer), int32(dimSize), int32(inner)]
  dispatchGeneric("argmax", [x, outVal, outIdx], addr pc[0], 12, uint32(pairs), 1, 1)

proc vulkanSessionCompareEnc*(a, b: VulkanBufferHandle, mask: var VulkanBufferHandle, op, n: int): bool =
  var pc = [int32(op), int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("compare", [a, b, mask], addr pc[0], 8, groups, 1, 1)

proc vulkanSessionDropoutMaskEnc*(mask: var VulkanBufferHandle, seed: uint32, p: float32, n: int): bool =
  let scale = if p < 1'f32: 1'f32 / (1'f32 - p) else: 0'f32
  var pc = (seed: seed, p: p, scale: scale, n: int32(n))
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("dropout_rng", [mask], addr pc, 16, groups, 1, 1)

proc vulkanSessionPermuteEnc*(src: VulkanBufferHandle, dst: var VulkanBufferHandle,
                               newStrides, oldStrides, dimsMap: seq[int32], n: int): bool =
  let ndim = newStrides.len
  doAssert ndim <= 8, "vulkanSessionPermuteEnc: rank > 8 not supported (fixed 8-entry push-constant array)"
  var pc: tuple[newStrides: array[8, int32], oldStrides: array[8, int32],
                dimsMap: array[8, int32], ndim: int32, n: int32]
  for i in 0 ..< 8:
    pc.newStrides[i] = if i < ndim: newStrides[i] else: 1
    pc.oldStrides[i] = if i < ndim: oldStrides[i] else: 0
    pc.dimsMap[i] = if i < ndim: dimsMap[i] else: 0
  pc.ndim = int32(ndim); pc.n = int32(n)
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("permute", [src, dst], addr pc, uint32(sizeof(pc)), groups, 1, 1)

proc vulkanSessionPermuteBufEnc*(src, newStrides, oldStrides, dimsMap: VulkanBufferHandle,
                                  dst: var VulkanBufferHandle, ndim, n: int): bool =
  ## Same as vulkanSessionPermuteEnc but the stride tables are uploaded index buffers.
  var pc = [int32(ndim), int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("permute_buf", [src, newStrides, oldStrides, dimsMap, dst], addr pc[0], 8, groups, 1, 1)

proc vulkanSessionEmbeddingLookupEnc*(table, idx: VulkanBufferHandle, dst: var VulkanBufferHandle,
                                       dim, numIndices: int): bool =
  let n = numIndices * dim
  var pc = [int32(dim), int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("embedding", [table, idx, dst], addr pc[0], 8, groups, 1, 1)

proc vulkanSessionApfluEnc*(x: VulkanBufferHandle, y: var VulkanBufferHandle, n: int,
                             alpha, beta: float32): bool =
  var pc: tuple[alpha, beta: float32, n: int32] = (alpha, beta, int32(n))
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("apflu", [x, y], addr pc, 12, groups, 1, 1)

proc vulkanSessionApfluBackwardEnc*(x, dy: VulkanBufferHandle, dx: var VulkanBufferHandle, n: int,
                                     alpha, beta: float32): bool =
  var pc: tuple[alpha, beta: float32, n: int32] = (alpha, beta, int32(n))
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("apflu_bwd", [x, dy, dx], addr pc, 12, groups, 1, 1)

proc vulkanSessionLayernormBackwardEnc*(dy, x, gamma: VulkanBufferHandle,
                                         dx, dgamma, dbeta: var VulkanBufferHandle,
                                         rows, cols: int, eps: float32): bool =
  var stats = vulkanSessionAllocScratch(2 * rows)
  registerInternalScratch(stats)
  var pcs: tuple[rows, cols: int32, eps: float32] = (int32(rows), int32(cols), eps)
  if not dispatchGeneric("layernorm_bwd_stats", [x, stats], addr pcs, 12, uint32((rows + 63) div 64), 1, 1): return false
  var pc2 = [int32(rows), int32(cols)]
  if not dispatchGeneric("layernorm_bwd_dx", [dy, x, gamma, stats, dx], addr pc2[0], 8,
                         uint32((rows + 63) div 64), 1, 1): return false
  dispatchGeneric("layernorm_bwd_params", [dy, x, stats, dgamma, dbeta], addr pc2[0], 8,
                  uint32((cols + 63) div 64), 1, 1)

proc vulkanSessionSplitSliceEnc*(src: VulkanBufferHandle, dst: var VulkanBufferHandle,
                                  dimTotal, thisSize, offset, inner, n: int): bool =
  var pc = [int32(dimTotal), int32(thisSize), int32(offset), int32(inner), int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("split_slice", [src, dst], addr pc[0], 20, groups, 1, 1)

proc vulkanSessionConcatEnc*(flatSrc, base, dimSize, dimOffset: VulkanBufferHandle, dst: var VulkanBufferHandle,
                              numTensors, totalDim, inner, n: int): bool =
  var pc = [int32(numTensors), int32(totalDim), int32(inner), int32(n)]
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("concat_gather", [flatSrc, base, dimSize, dimOffset, dst], addr pc[0], 16, groups, 1, 1)

proc vulkanVecOp*(op: string, a, b: seq[float32]): seq[float32] =
  let opCode =
    case op
    of "add": 0
    of "sub": 1
    of "mul": 2
    of "div": 3
    else: 0
  let n = a.len
  result = newSeq[float32](n)
  gpuCheck(vulkanSessionBegin(), "vulkanVecOp: session begin that bai")
  var bufA = vulkanSessionUpload(a)
  var bufB = vulkanSessionUpload(b)
  var bufC = vulkanSessionAllocScratch(n)
  let ok = vulkanSessionVecOpEnc(bufA, bufB, bufC, opCode, n)
  gpuCheck(vulkanSessionEnd(), "vulkanVecOp: session end that bai")
  gpuCheck(ok, "vulkanVecOp: dispatch that bai")
  result = vulkanSessionRead(bufC, n)
  vulkanSessionFree(bufA)
  vulkanSessionFree(bufB)
  vulkanSessionFree(bufC)

proc vulkanMatmul*(a, b: seq[float32], m, k, n: int): seq[float32] =
  # FIX: Check for empty inputs before GPU allocation
  if a.len == 0 or b.len == 0 or m == 0 or k == 0 or n == 0:
    stderr.writeLine("[vulkanMatmul] WARNING: empty input detected, fallback to CPU, dims=(" & $m & "," & $k & "," & $n & "), a.len=" & $a.len & ", b.len=" & $b.len)
    result = newSeq[float32](m * n)
    return
  result = newSeq[float32](m * n)
  gpuCheck(vulkanSessionBegin(), "vulkanMatmul: session begin that bai")
  var bufA = vulkanSessionUpload(a)
  var bufB = vulkanSessionUpload(b)
  var bufC = vulkanSessionAllocScratch(m * n)
  let ok = vulkanSessionMatmulEnc(bufA, bufB, bufC, m, k, n)
  gpuCheck(vulkanSessionEnd(), "vulkanMatmul: session end that bai")
  gpuCheck(ok, "vulkanMatmul: dispatch that bai")
  result = vulkanSessionRead(bufC, m * n)
  vulkanSessionFree(bufA)
  vulkanSessionFree(bufB)
  vulkanSessionFree(bufC)

proc vulkanActivation*(op: string, x: seq[float32], p: float32 = 0'f32): seq[float32] =
  ## REAL GPU for every opcode - activation.comp already has 0=relu,1=sigmoid,
  ## 2=tanh,4=exp,5=log,6=sqrt,7=pow (with its own push-constant p for pow).
  ## NO CPU fallback for any op in this list - if the GPU dispatch
  ## fails it raises a real error (through gpuCheck) instead of silently computing on the CPU.
  let opCode =
    case op
    of "relu": 0
    of "sigmoid": 1
    of "tanh": 2
    of "exp": 4
    of "log": 5
    of "sqrt": 6
    of "pow": 7
    else: raise newException(ValueError, "vulkanActivation: unsupported op: " & op)
  let n = x.len
  gpuCheck(vulkanSessionBegin(), "vulkanActivation: session begin that bai")
  var bufX = vulkanSessionUpload(x)
  var bufY = vulkanSessionAllocScratch(n)
  let ok = vulkanSessionActivationEnc(opCode, bufX, bufY, n, p)
  gpuCheck(vulkanSessionEnd(), "vulkanActivation: session end that bai")
  gpuCheck(ok, "vulkanActivation: GPU dispatch failed for op=" & op & " (no CPU fallback)")
  result = vulkanSessionRead(bufY, n)
  vulkanSessionFree(bufX)
  vulkanSessionFree(bufY)

proc vulkanSoftmax*(x: seq[float32], rows, cols: int): seq[float32] =
  let n = rows * cols
  gpuCheck(vulkanSessionBegin(), "vulkanSoftmax: session begin that bai")
  var bufX = vulkanSessionUpload(x)
  var bufY = vulkanSessionAllocScratch(n)
  let ok = vulkanSessionSoftmaxEnc(bufX, bufY, rows, cols)
  gpuCheck(vulkanSessionEnd(), "vulkanSoftmax: session end that bai")
  gpuCheck(ok, "vulkanSoftmax: dispatch that bai")
  result = vulkanSessionRead(bufY, n)
  vulkanSessionFree(bufX)
  vulkanSessionFree(bufY)

proc vulkanLayernorm*(x, gamma, beta: seq[float32], rows, cols: int, eps: float32): seq[float32] =
  ## CPU fallback - there is no kernels/layernorm.comp yet (only rmsnorm.comp).
  result = newSeq[float32](rows * cols)
  for r in 0 ..< rows:
    let base = r * cols
    var mean = 0.0'f32
    for c in 0 ..< cols: mean += x[base + c]
    mean /= float32(cols)
    var variance = 0.0'f32
    for c in 0 ..< cols:
      let d = x[base + c] - mean
      variance += d * d
    variance /= float32(cols)
    let invStd = 1.0'f32 / sqrt(variance + eps)
    for c in 0 ..< cols:
      result[base + c] = gamma[c] * (x[base + c] - mean) * invStd + beta[c]

proc vulkanLayernormBackward*(dy, x, gamma, beta: seq[float32], rows, cols: int, eps: float32): tuple[dx, dgamma, dbeta: seq[float32]] =
  ## CPU fallback - the standard LayerNorm backward formula, same results
  ## as cudaLayernormBackward/metalLayernormBackward.
  var dx = newSeq[float32](rows * cols)
  var dgamma = newSeq[float32](cols)
  var dbeta = newSeq[float32](cols)
  for r in 0 ..< rows:
    let base = r * cols
    var mean = 0.0'f32
    for c in 0 ..< cols: mean += x[base + c]
    mean /= float32(cols)
    var variance = 0.0'f32
    for c in 0 ..< cols:
      let d = x[base + c] - mean
      variance += d * d
    variance /= float32(cols)
    let invStd = 1.0'f32 / sqrt(variance + eps)
    var xhat = newSeq[float32](cols)
    for c in 0 ..< cols:
      xhat[c] = (x[base + c] - mean) * invStd
    var sumDyG = 0.0'f32
    var sumDyGXhat = 0.0'f32
    for c in 0 ..< cols:
      let dyg = dy[base + c] * gamma[c]
      sumDyG += dyg
      sumDyGXhat += dyg * xhat[c]
      dgamma[c] += dy[base + c] * xhat[c]
      dbeta[c] += dy[base + c]
    for c in 0 ..< cols:
      let dyg = dy[base + c] * gamma[c]
      dx[base + c] = invStd * (dyg - sumDyG / float32(cols) - xhat[c] * sumDyGXhat / float32(cols))
  result = (dx, dgamma, dbeta)

proc vulkanEmbeddingLookup*(table: seq[float32], indices: seq[int32], vocab, dim: int): seq[float32] =
  ## CPU fallback - a simple gather, fast enough without the GPU.
  let num = indices.len
  result = newSeq[float32](num * dim)
  for i in 0 ..< num:
    let idx = int(indices[i])
    for d in 0 ..< dim:
      result[i * dim + d] = table[idx * dim + d]

proc vulkanApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
  ## CPU fallback - exactly the vecop_apflu formula in cuda kernels/vecop.ptx:
  ## x>0 -> x*(1+alpha*x); x<=0 -> beta*x*exp(x).
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    let v = x[i]
    if v > 0.0'f32:
      result[i] = v * (1.0'f32 + alpha * v)
    else:
      result[i] = beta * v * exp(v)

proc vulkanApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
  ## CPU fallback - the APFLU derivative: x>0 -> dy*(1+2*alpha*x);
  ## x<=0 -> dy*beta*exp(x)*(1+x).
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    let v = x[i]
    if v > 0.0'f32:
      result[i] = dy[i] * (1.0'f32 + 2.0'f32 * alpha * v)
    else:
      result[i] = dy[i] * beta * exp(v) * (1.0'f32 + v)

proc vulkanAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  ## Real GPU through kernels/attention_fused.comp (session Enc) - the mask
  ## is NOT used (same as cudaAttentionFused/metalAttentionFused,
  ## the shader applies the causal mask itself).
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  gpuCheck(vulkanSessionBegin(), "vulkanAttentionFused: session begin that bai")
  var bufQ = vulkanSessionUpload(q)
  var bufK = vulkanSessionUpload(k)
  var bufV = vulkanSessionUpload(v)
  var bufO = vulkanSessionAllocScratch(qkvLen)
  var bufS = vulkanSessionAllocScratch(sLen)
  let ok = vulkanSessionAttentionFusedEnc(bufQ, bufK, bufV, bufO, bufS, B, H, S, D, scale)
  gpuCheck(vulkanSessionEnd(), "vulkanAttentionFused: session end that bai")
  gpuCheck(ok, "vulkanAttentionFused: dispatch that bai")
  result.o = vulkanSessionRead(bufO, qkvLen)
  result.s_matrix = vulkanSessionRead(bufS, sLen)
  vulkanSessionFree(bufQ)
  vulkanSessionFree(bufK)
  vulkanSessionFree(bufV)
  vulkanSessionFree(bufO)
  vulkanSessionFree(bufS)

proc vulkanAttentionFusedBackward*(q, k, v, s_matrix, dy: seq[float32], B, H, S, D: int, scale: float32): tuple[dq, dk, dv: seq[float32]] =
  ## CHANGED: real GPU through kernels/attention_fused_backward.comp (session
  ## Enc, a 1:1 mirror of metal_attention_fused_backward - see the explanation in the
  ## .comp file), NO longer a pure CPU loop pretending to be GPU.
  ## Same S<=256 limit as the Metal version (local arrays softmaxW[256]/
  ## dSoftmax[256] in the shader) - checked explicitly here, SAME as
  ## metal_shim.m returning 0/false when S>256, to report a clear error instead of reading/
  ## writing past the shader arrays.
  if S > 256:
    raise newException(ValueError,
      "vulkanAttentionFusedBackward: S=" & $S & " exceeds the kernel limit of 256 (same limit as the Metal version)")
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  gpuCheck(vulkanSessionBegin(), "vulkanAttentionFusedBackward: session begin that bai")
  var bufQ = vulkanSessionUpload(q)
  var bufK = vulkanSessionUpload(k)
  var bufV = vulkanSessionUpload(v)
  var bufS = vulkanSessionUpload(s_matrix)
  var bufDy = vulkanSessionUpload(dy)
  # dq/dk/dv: the shader uses "+=" (dq) / atomic CAS-add (dk, dv) -> they MUST be
  # zero-initialised before dispatch, SAME as the memset in metal_shim.m -
  # vulkanSessionUpload with a zero seq instead of vulkanSessionAllocScratch
  # (not guaranteed zero) makes sure the buffers are clean before the shader accumulates.
  var bufDq = vulkanSessionUpload(newSeq[float32](qkvLen))
  var bufDk = vulkanSessionUpload(newSeq[float32](qkvLen))
  var bufDv = vulkanSessionUpload(newSeq[float32](qkvLen))
  let ok = vulkanSessionAttentionFusedBackwardEnc(bufQ, bufK, bufV, bufS, bufDy, bufDq, bufDk, bufDv, B, H, S, D, scale)
  gpuCheck(vulkanSessionEnd(), "vulkanAttentionFusedBackward: session end that bai")
  gpuCheck(ok, "vulkanAttentionFusedBackward: dispatch that bai")
  result.dq = vulkanSessionRead(bufDq, qkvLen)
  result.dk = vulkanSessionRead(bufDk, qkvLen)
  result.dv = vulkanSessionRead(bufDv, qkvLen)
  vulkanSessionFree(bufQ)
  vulkanSessionFree(bufK)
  vulkanSessionFree(bufV)
  vulkanSessionFree(bufS)
  vulkanSessionFree(bufDy)
  vulkanSessionFree(bufDq)
  vulkanSessionFree(bufDk)
  vulkanSessionFree(bufDv)
# =======================================================================
# REAL GPU PORT (round 2): GATHER / ARGMAX-INDEX / COMPARE / DROPOUT-RNG
# Shaders: kernels/gather.comp, argmax.comp, compare.comp, dropout_rng.comp
# - compiled by hand with glslc before running (see the note on getOrCreatePipeline
# above: it looks for kernels/<shaderName>.spv by name). Same session
# Begin/Enc/End/Read pattern as vulkanVecOp/vulkanMatmul above.
# =======================================================================

proc vulkanGather*(src: seq[float32], idx: seq[int32]): seq[float32] =
  let n = idx.len
  result = newSeq[float32](n)
  gpuCheck(vulkanSessionBegin(), "vulkanGather: session begin that bai")
  var bufSrc = vulkanSessionUpload(src)
  var idxBytes = newSeq[uint8](n * sizeof(int32))
  if n > 0: copyMem(addr idxBytes[0], unsafeAddr idx[0], n * sizeof(int32))
  var bufIdx = vulkanSessionUploadBytes(idxBytes)
  var bufDst = vulkanSessionAllocScratch(n)
  var pc = [int32(n)]
  let groups = uint32((n + 255) div 256)
  let ok = dispatchGeneric("gather", [bufSrc, bufIdx, bufDst], addr pc[0], 4, groups, 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanGather: session end that bai")
  gpuCheck(ok, "vulkanGather: dispatch that bai")
  result = vulkanSessionRead(bufDst, n)
  vulkanSessionFree(bufSrc); vulkanSessionFree(bufIdx); vulkanSessionFree(bufDst)

proc vulkanArgmaxIndex*(x: seq[float32], outer, dimSize, inner: int): tuple[val: seq[float32], idx: seq[int32]] =
  ## argmax.comp: local_size_x=1024, 1 workgroup / (o,i) pair, requires
  ## dimSize <= 1024.
  let pairs = outer * inner
  gpuCheck(vulkanSessionBegin(), "vulkanArgmaxIndex: session begin that bai")
  var bufX = vulkanSessionUpload(x)
  var bufVal = vulkanSessionAllocScratch(pairs)
  var idxScratchBytes = newSeq[uint8](pairs * sizeof(int32))
  var bufIdx = vulkanSessionUploadBytes(idxScratchBytes)  # scratch int32 (init 0)
  var pc = [int32(outer), int32(dimSize), int32(inner)]
  let ok = dispatchGeneric("argmax", [bufX, bufVal, bufIdx], addr pc[0], 12, uint32(pairs), 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanArgmaxIndex: session end that bai")
  gpuCheck(ok, "vulkanArgmaxIndex: dispatch that bai")
  let outVal = vulkanSessionRead(bufVal, pairs)
  var outIdx = newSeq[int32](pairs)
  let idxF = vulkanSessionRead(bufIdx, pairs)  # read back as float32, then reinterpret the bytes
  if pairs > 0: copyMem(addr outIdx[0], unsafeAddr idxF[0], pairs * sizeof(int32))
  vulkanSessionFree(bufX); vulkanSessionFree(bufVal); vulkanSessionFree(bufIdx)
  result = (val: outVal, idx: outIdx)

proc vulkanCompare*(a, b: seq[float32], op: int): seq[uint32] =
  let n = a.len
  gpuCheck(vulkanSessionBegin(), "vulkanCompare: session begin that bai")
  var bufA = vulkanSessionUpload(a)
  var bufB = vulkanSessionUpload(b)
  var maskScratchBytes = newSeq[uint8](n * sizeof(uint32))
  var bufMask = vulkanSessionUploadBytes(maskScratchBytes)
  var pc = [int32(op), int32(n)]
  let groups = uint32((n + 255) div 256)
  let ok = dispatchGeneric("compare", [bufA, bufB, bufMask], addr pc[0], 8, groups, 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanCompare: session end that bai")
  gpuCheck(ok, "vulkanCompare: dispatch that bai")
  let maskF = vulkanSessionRead(bufMask, n)
  result = newSeq[uint32](n)
  if n > 0: copyMem(addr result[0], unsafeAddr maskF[0], n * sizeof(uint32))
  vulkanSessionFree(bufA); vulkanSessionFree(bufB); vulkanSessionFree(bufMask)

proc vulkanDropoutMask*(n: int, p: float32, seed: uint32): seq[float32] =
  gpuCheck(vulkanSessionBegin(), "vulkanDropoutMask: session begin that bai")
  var bufMask = vulkanSessionAllocScratch(n)
  let scale = if p < 1'f32: 1'f32 / (1'f32 - p) else: 0'f32
  var pc = (seed: seed, p: p, scale: scale, n: int32(n))
  let groups = uint32((n + 255) div 256)
  let ok = dispatchGeneric("dropout_rng", [bufMask], addr pc, 16, groups, 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanDropoutMask: session end that bai")
  gpuCheck(ok, "vulkanDropoutMask: dispatch that bai")
  result = vulkanSessionRead(bufMask, n)
  vulkanSessionFree(bufMask)

# =======================================================================
# REAL GPU PORT (round 3) - LIKE TORCH: permute/split/concat no longer build
# O(n) index arrays on the CPU. permute.comp/split_slice.comp/concat_gather.comp
# decode the indices on the GPU from a small O(rank)/O(numTensors) table.
# =======================================================================

proc vulkanPermute*(src: seq[float32], newStrides, oldStrides, dimsMap: seq[int32], n: int): seq[float32] =
  let ndim = newStrides.len
  doAssert ndim <= 8, "vulkanPermute: rank > 8 not supported (fixed 8-entry push-constant array)"
  gpuCheck(vulkanSessionBegin(), "vulkanPermute: session begin that bai")
  var bufSrc = vulkanSessionUpload(src)
  var bufDst = vulkanSessionAllocScratch(n)
  var pc: tuple[newStrides: array[8, int32], oldStrides: array[8, int32],
                dimsMap: array[8, int32], ndim: int32, n: int32]
  for i in 0 ..< 8:
    pc.newStrides[i] = if i < ndim: newStrides[i] else: 1
    pc.oldStrides[i] = if i < ndim: oldStrides[i] else: 0
    pc.dimsMap[i] = if i < ndim: dimsMap[i] else: 0
  pc.ndim = int32(ndim); pc.n = int32(n)
  let groups = uint32((n + 255) div 256)
  let ok = dispatchGeneric("permute", [bufSrc, bufDst], addr pc, uint32(sizeof(pc)), groups, 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanPermute: session end that bai")
  gpuCheck(ok, "vulkanPermute: dispatch that bai")
  result = vulkanSessionRead(bufDst, n)
  vulkanSessionFree(bufSrc); vulkanSessionFree(bufDst)

proc vulkanSplitSlice*(src: seq[float32], dimTotal, thisSize, offset, inner, n: int): seq[float32] =
  gpuCheck(vulkanSessionBegin(), "vulkanSplitSlice: session begin that bai")
  var bufSrc = vulkanSessionUpload(src)
  var bufDst = vulkanSessionAllocScratch(n)
  var pc = [int32(dimTotal), int32(thisSize), int32(offset), int32(inner), int32(n)]
  let groups = uint32((n + 255) div 256)
  let ok = dispatchGeneric("split_slice", [bufSrc, bufDst], addr pc[0], 20, groups, 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanSplitSlice: session end that bai")
  gpuCheck(ok, "vulkanSplitSlice: dispatch that bai")
  result = vulkanSessionRead(bufDst, n)
  vulkanSessionFree(bufSrc); vulkanSessionFree(bufDst)

proc vulkanConcat*(flatSrc: seq[float32], srcBase, srcDimSize, srcDimOffset: seq[int32],
                    totalDim, inner, n: int): seq[float32] =
  ## SPIR-V has no array of device pointers -> one FLATTENED source buffer
  ## (flatSrc, the host only concatenates with memcpy - NO per-element index computation) +
  ## a small O(numTensors) table (srcBase/srcDimSize/srcDimOffset).
  let numTensors = srcBase.len
  gpuCheck(vulkanSessionBegin(), "vulkanConcat: session begin that bai")
  var bufSrc = vulkanSessionUpload(flatSrc)
  var baseBytes = newSeq[uint8](numTensors * sizeof(int32))
  if numTensors > 0: copyMem(addr baseBytes[0], unsafeAddr srcBase[0], numTensors * sizeof(int32))
  var bufBase = vulkanSessionUploadBytes(baseBytes)
  var sizeBytes = newSeq[uint8](numTensors * sizeof(int32))
  if numTensors > 0: copyMem(addr sizeBytes[0], unsafeAddr srcDimSize[0], numTensors * sizeof(int32))
  var bufSize = vulkanSessionUploadBytes(sizeBytes)
  var offBytes = newSeq[uint8](numTensors * sizeof(int32))
  if numTensors > 0: copyMem(addr offBytes[0], unsafeAddr srcDimOffset[0], numTensors * sizeof(int32))
  var bufOff = vulkanSessionUploadBytes(offBytes)
  var bufDst = vulkanSessionAllocScratch(n)
  var pc = [int32(numTensors), int32(totalDim), int32(inner), int32(n)]
  let groups = uint32((n + 255) div 256)
  let ok = dispatchGeneric("concat_gather", [bufSrc, bufBase, bufSize, bufOff, bufDst],
    addr pc[0], 16, groups, 1, 1)
  gpuCheck(vulkanSessionEnd(), "vulkanConcat: session end that bai")
  gpuCheck(ok, "vulkanConcat: dispatch that bai")
  result = vulkanSessionRead(bufDst, n)
  vulkanSessionFree(bufSrc); vulkanSessionFree(bufBase); vulkanSessionFree(bufSize)
  vulkanSessionFree(bufOff); vulkanSessionFree(bufDst)

# ── MicroVM launch (GLSL interpreter) ───────────────────────
# Requires: micro_vm.spv loaded (compile micro_vm.comp).
# Uses the dispatchGeneric pattern if the "micro_vm" pipeline is registered.

proc vulkanLaunchMicroVm*(code: seq[uint32]; memF32: var seq[float32];
                          nThreads: int; tgFloats: int = 1): bool =
  ## Best-effort: upload code+mem as buffers, dispatch micro_vm pipeline.
  if code.len == 0 or nThreads <= 0: return false
  # Encode code as float bit-pattern buffer is awkward; use byte upload.
  var codeBytes = newSeq[uint8](code.len * 4)
  if code.len > 0:
    copyMem(addr codeBytes[0], unsafeAddr code[0], code.len * 4)
  var bufCode = vulkanSessionUploadBytes(codeBytes)
  var bufMem = vulkanSessionUpload(memF32)
  var bufU8 = vulkanSessionAllocScratch(4)   # dummy
  var bufLut = vulkanSessionAllocScratch(256)
  var pc = [int32(code.len), int32(max(1, tgFloats))]
  let groups = uint32((nThreads + 255) div 256)
  let ok = dispatchGeneric("micro_vm", [bufCode, bufMem, bufU8, bufLut],
    addr pc[0], 8, groups, 1, 1)
  if ok:
    memF32 = vulkanSessionRead(bufMem, memF32.len)
  vulkanSessionFree(bufCode); vulkanSessionFree(bufMem)
  vulkanSessionFree(bufU8); vulkanSessionFree(bufLut)
  return ok

# =======================================================================
# MULTI-BUFFER JIT SESSION API (inference encode path)
# Parallel to Metal's sessionPipeline / sessionBindBuffer / sessionDispatch.
# Training uses 1-buffer vulkanSessionJitMicroVmEnc; inference needs N buffers
# (output, weight u8, LUT, input) with distinct bindings.
# =======================================================================

type
  VkBinding* = tuple[idx: int; buf: VulkanBufferHandle; offElems: int]

var gVkJitPending: tuple[name, src: string]
var gVkBindings: seq[VkBinding]
var gVkTgBytes: int = 0
# Unused descriptor slots share one process-lifetime buffer, like pipeline
# caches. It is never handed to callers or freed while queued work refers to it.
var gVkJitDummy: VulkanBufferHandle

proc vulkanJitBegin*(name, glslSrc: string): bool =
  ## Register pipeline name + GLSL source. Compile happens lazily at dispatch.
  ## Resets the pending binding list.
  gVkJitPending = (name, glslSrc)
  gVkBindings.setLen(0)
  gVkTgBytes = 0
  result = true

proc vulkanBindBuf*(idx: int; buf: VulkanBufferHandle; offElems = 0): bool =
  ## Stage one descriptor binding: slot `idx` → buffer at element offset.
  ## For float buffers offElems is in float32 elements; for u8 it's in bytes/4
  ## if the caller follows Metal convention (Metal multiplies by 4 for floats).
  ## Vulkan path stores element offset and converts to bytes at dispatch time
  ## based on the binding index convention of lowerBytecodeToGlsl:
  ##   binding 0 = MemF32 (float), 1 = MemU8 (uint/bytes), 2 = Lut (float),
  ##   3 = Input (float), 4 = MemU32 (uint).
  gVkBindings.add((idx, buf, offElems))
  result = true

proc vulkanSetThreadgroupBytes*(index, nBytes: int): bool =
  ## Record requested threadgroup/shared memory size (informational for now;
  ## GLSL shared float tgF32[1024] is fixed at compile time).
  discard index
  gVkTgBytes = nBytes
  result = true

proc vulkanMaxTgThreads*(): int =
  ensureInit()
  result = min(gDeviceCaps.maxThreads, 1024)
  let raw = getEnv("BYBY_VULKAN_MAX_TG")
  if raw.len > 0:
    try: result = min(result, max(1, parseInt(raw)))
    except ValueError: discard

proc vulkanSessionJitMultiBufEnc*(kernelName, source: string;
                                   bindings: seq[VkBinding];
                                   nThreads, tgSize: int;
                                   tgBytes = 0): bool =
  ## Core multi-buffer JIT dispatch. Builds a dense buffer array ordered by
  ## binding index (0..maxIdx), compiles/caches SPIR-V, then dispatches via
  ## dispatchGeneric. Pattern mirrors vulkanSessionJitMicroVmEnc but each
  ## logical buffer can be a distinct VkBuffer.
  let maxTg = vulkanMaxTgThreads()
  if not gSessionActive or nThreads <= 0 or tgSize <= 0 or tgSize > maxTg or
      tgBytes > gDeviceCaps.maxSharedBytes:
    stderr.writeLine "[vulkan_jit] multiBuf SKIP kernel=" & kernelName &
      " sessionActive=" & $gSessionActive & " nThreads=" & $nThreads &
      " tgSize=" & $tgSize & " maxTg=" & $maxTg
    return false
  if bindings.len == 0:
    stderr.writeLine "[vulkan_jit] multiBuf SKIP kernel=" & kernelName & " bindings=0"
    return false
  var maxIdx = 0
  for b in bindings:
    if b.idx > maxIdx: maxIdx = b.idx
  # Only missing slots need a dummy. Avoid a VkBuffer + VkDeviceMemory pair
  # for every dispatch, including dispatches whose bindings are already dense.
  var bufs = newSeq[VulkanBufferHandle](maxIdx + 1)
  var offs = newSeq[uint64](maxIdx + 1)
  for b in bindings:
    if b.buf.buf != nil:
      bufs[b.idx] = b.buf
      # GLSL: 0=MemF32, 1=MemU8, 2=Lut, 3=Input, 4=MemU32
      # Caller offElems: float/u32 = element index; MemU8 = byte offset/4 when following
      # quy uoc Metal (float*4). Weight/LUT thuong off=0.
      # offElems = float32 index (matches the Metal setBuffer offset/4). Weight U8 is always 0.
      offs[b.idx] = uint64(b.offElems) * uint64(sizeof(float32))
  let useCache = getEnv("BYBY_VULKAN_DISPATCH_CACHE", "1") != "0"
  var dummy: VulkanBufferHandle
  if not useCache:
    # Diagnostic A/B retains the previous allocation on every dispatch.
    dummy = vulkanSessionAllocScratch(4)
    registerInternalScratch(dummy)
    inc gJitDummyAllocations
  for i in 0 .. maxIdx:
    if bufs[i].buf == nil:
      if useCache:
        if gVkJitDummy.buf == nil:
          gVkJitDummy = vulkanSessionAllocScratch(4)
          inc gJitDummyAllocations
        bufs[i] = gVkJitDummy
      else:
        bufs[i] = dummy
  if getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") == "1":
    stderr.writeLine "[vulkan_jit] multiBuf kernel=" & kernelName &
      " bindings=" & $bindings.len & " maxIdx=" & $maxIdx &
      " nThreads=" & $nThreads & " tgSize=" & $tgSize & " tgBytes=" & $tgBytes &
      " offs=" & $offs
  let spvPath = compileVulkanJitShader(kernelName, source)
  let groups = uint32((nThreads + tgSize - 1) div tgSize)
  result = dispatchGeneric(kernelName, bufs, nil, 0, groups, 1, 1, tgSize, spvPath, offs, max(1, (tgBytes + 3) div 4))
  if not result:
    stderr.writeLine "[vulkan_jit] multiBuf FAIL kernel=" & kernelName
  discard tgBytes

proc vulkanKernelDispatch*(nThreads, tgSize: int; tgBytes = 0): bool =
  ## Build descriptor set from gVkBindings, compile/cache SPIR-V, dispatch.
  result = vulkanSessionJitMultiBufEnc(
    gVkJitPending.name, gVkJitPending.src,
    gVkBindings, nThreads, tgSize, tgBytes)
  gVkBindings.setLen(0)
  gVkTgBytes = 0

# Alias used by fp4 / session upload path
proc vulkanSessionUploadU8*(data: openArray[uint8]): VulkanBufferHandle =
  ## Discrete GPUs retain weights in device-local VRAM after one staging copy.
  vulkanSessionUploadBytesPrivate(data)
