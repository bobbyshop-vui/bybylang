## Vulkan/MoltenVK compute sessions and resident buffers.
## Optional KHR cooperative-matrix inference GEMM uses FP16 tiles and FP32
## accumulation. Device features/limits are queried before creating the device.
## CUDA libraries are not required by this backend.

import std/dynlib
import std/os
import std/math
import std/[osproc, hashes, strutils, sets, algorithm, monotimes, times]
import vulkan_caps
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
  VkResult* = int32

  # ---- Struct Vulkan tối thiểu cần cho Bước 1 (đúng thứ tự field như
  # vulkan_core.h thật - SAI THỨ TỰ SẼ GÂY LỖI NGẦM, không phải lỗi compile,
  # vì Nim object truyền theo layout khai báo, khớp C struct layout chỉ khi
  # thứ tự + kiểu field khớp CHÍNH XÁC). ----
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
    codeSize*: uint             # csize_t thực chất, dùng uint cho khớp size_t 64-bit
    pCode*: ptr uint32           # con trỏ tới bytecode SPIR-V (bội số 4 byte)

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
  VK_SUCCESS*: VkResult = 0

  # MoltenVK/macOS BẮT BUỘC 2 cờ portability này khi tạo instance (từ
  # Vulkan 1.3.216+ trở đi trên macOS) - thiếu sẽ lỗi VK_ERROR_INCOMPATIBLE_DRIVER.
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
    ## Vulkan CẦN CẢ HAI (VkBuffer + VkDeviceMemory) để dùng 1 buffer -
    ## khác Metal/CUDA/OpenCL chỉ cần 1 handle - đóng gói lại thành 1 struct
    ## cho khớp interface SessionHandle chung (chỉ có 1 field/backend).
    buf*: VkBuffer
    mem*: VkDeviceMemory
    sizeBytes*: uint64

proc gpuCheck(cond: bool, msg: string) =
  ## Giống hệt pattern trong cuda_driver.nim - raise CatchableError (không
  ## phải Defect) để lớp trên bắt được và fallback, không crash cả process.
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
var gFp8Fp4Logged = false
var gHostImportEnabled = false
var vkGetMemoryHostPointerPropertiesEXT: proc(dev: VkDevice; handleType: uint32;
  host: pointer; props: pointer): VkResult {.cdecl.}
var gFp16Fp4Logged = false
var gDevice: VkDevice
var gQueue: VkQueue
var gQueueFamilyIndex: uint32
var gCommandPool: VkCommandPool

proc tryLoad(): bool =
  if loaded: return lib.handle != nil
  loaded = true
  # MoltenVK có thể cài dưới 1 trong vài tên tuỳ cách cài (Vulkan SDK cho
  # macOS thường cài libMoltenVK.dylib + libvulkan.1.dylib làm loader trỏ
  # tới nó - thử cả 2 tên phổ biến nhất). Trên Linux, loader chuẩn tên
  # libvulkan.so.1 (package vulkan-loader/libvulkan1) - TRUOC DAY THIEU tên
  # này nên tryLoad() luôn thất bại trên Linux dù máy có driver Vulkan thật.
  for name in ["vulkan-1.dll", "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib",
               "libvulkan.so.1", "libvulkan.so"]:
    lib.handle = loadLib(name)
    if lib.handle != nil: break
  if lib.handle == nil:
    return false
  lib.vkGetInstanceProcAddr = cast[typeof(lib.vkGetInstanceProcAddr)](lib.handle.symAddr("vkGetInstanceProcAddr"))
  if lib.vkGetInstanceProcAddr == nil:
    return false
  # vkCreateInstance là hàm DUY NHẤT resolve được TRỰC TIẾP qua symAddr của
  # thư viện (chưa cần instance) - các hàm còn lại BẮT BUỘC resolve qua
  # vkGetInstanceProcAddr(instance, ...) SAU KHI đã có instance thật (đúng
  # theo spec Vulkan loader - KHÔNG phải tuỳ chọn, thiếu bước này 1 số hàm
  # sẽ trả về nil trên driver ICD thật dù dlsym trực tiếp "có vẻ" ra địa chỉ).
  lib.vkCreateInstance = cast[typeof(lib.vkCreateInstance)](lib.handle.symAddr("vkCreateInstance"))
  result = lib.vkCreateInstance != nil

proc loadInstanceProc[T](name: string): T =
  let p = lib.vkGetInstanceProcAddr(gInstance, name.cstring)
  gpuCheck(p != nil, "vkGetInstanceProcAddr tra ve nil cho ham: " & name)
  result = cast[T](p)

proc ensureInit() =
  if gInitialized: return
  gpuCheck(tryLoad(), "Khong tim thay/tai duoc thu vien Vulkan loader (libvulkan.so.1 tren Linux, libvulkan.1.dylib/libMoltenVK.dylib tren macOS) - kiem tra da cai Vulkan driver/ICD (vd. package vulkan-loader hoac vulkan-icd-loader) hoac MoltenVK/Vulkan SDK chua, va LD_LIBRARY_PATH/DYLD_LIBRARY_PATH co tro toi thu muc chua no khong")

  # ---- 1. Tạo VkInstance (kèm cờ portability bắt buộc cho MoltenVK) ----
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

  # THU LAI KHONG CO CO PORTABILITY: co VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
  # + extension "VK_KHR_portability_enumeration" la khai niem o TANG VULKAN LOADER
  # (de loader biet liet ke ca cac ICD "khong chuan" nhu MoltenVK). Neu may nay
  # KHONG co Vulkan Loader that (vd. libvulkan.1.dylib khong ton tai) va tryLoad()
  # da phai fallback load THANG vao libMoltenVK.dylib (ICD tran, bo qua loader),
  # thi ICD tran do KHONG hieu cai co/extension nay -> tra ve chinh xac
  # VK_ERROR_INCOMPATIBLE_DRIVER (-9) du driver GPU van hoan toan on. Truong hop
  # nay, tao lai instance lan 2 KHONG co co/extension portability.
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

  gpuCheckVk(rInst, "vkCreateInstance that bai ca 2 lan (co va khong co " &
    "portability extension). Neu -9=VK_ERROR_INCOMPATIBLE_DRIVER: may nay dang thieu Vulkan Loader/ICD that su " &
    "(khong phai loi logic tao instance nua) - can cai: 'brew install molten-vk vulkan-loader' " &
    "roi set VK_ICD_FILENAMES tro toi file MoltenVK_icd.json (xem 'brew --prefix molten-vk')")
  gInstance = inst

  # ---- 2. Resolve các hàm còn lại QUA vkGetInstanceProcAddr(instance,...) ----
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
  gpuCheck(devCount > 0, "Khong tim thay physical device Vulkan nao")
  var devices = newSeq[VkPhysicalDevice](devCount)
  gpuCheck(lib.vkEnumeratePhysicalDevices(gInstance, addr devCount, addr devices[0]) == VK_SUCCESS, "vkEnumeratePhysicalDevices (lay danh sach) that bai")
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

  # ---- 4. Tìm queue family hỗ trợ compute ----
  var qCount: uint32 = 0
  lib.vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevice, addr qCount, nil)
  gpuCheck(qCount > 0, "Physical device khong co queue family nao")
  var qProps = newSeq[VkQueueFamilyProperties](qCount)
  lib.vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevice, addr qCount, addr qProps[0])
  var foundQueue = false
  for i in 0 ..< int(qCount):
    if (qProps[i].queueFlags and VK_QUEUE_COMPUTE_BIT) != 0:
      gQueueFamilyIndex = uint32(i)
      foundQueue = true
      break
  gpuCheck(foundQueue, "Khong tim thay queue family ho tro compute")

  # ---- 5. Tạo logical device + lấy queue ----
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
  if gHostImportEnabled:
    vkGetMemoryHostPointerPropertiesEXT = cast[typeof(vkGetMemoryHostPointerPropertiesEXT)](
      lib.vkGetInstanceProcAddr(gInstance, "vkGetMemoryHostPointerPropertiesEXT"))
    gHostImportEnabled = vkGetMemoryHostPointerPropertiesEXT != nil
  lib.vkGetDeviceQueue(gDevice, gQueueFamilyIndex, 0, addr gQueue)

  # ---- 6. Command pool (dùng lại cho mọi lệnh compute sau này, giống
  # gSessionEncoder bên Metal) ----
  var poolInfo = VkCommandPoolCreateInfo(
    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: nil, flags: 0, queueFamilyIndex: gQueueFamilyIndex
  )
  gpuCheck(lib.vkCreateCommandPool(gDevice, addr poolInfo, nil, addr gCommandPool) == VK_SUCCESS, "vkCreateCommandPool that bai")

  gInitialized = true
  let weightMemory = if useDeviceLocalWeights(gDeviceCaps.properties.deviceType,
      getEnv("BYBY_VULKAN_DEVICE_LOCAL", "auto")): "device-local/VRAM"
    else: "host-visible/shared"
  stderr.writeLine "[vulkan] device=" & gDeviceCaps.deviceName &
    " type=" & deviceClass(gDeviceCaps.properties.deviceType) &
    " residentWeights=" & weightMemory &
    " subgroup=" & $gDeviceCaps.subgroupSize & " maxThreads=" & $gDeviceCaps.maxThreads &
    " sharedBytes=" & $gDeviceCaps.maxSharedBytes &
    " mmapImport=" & (if gHostImportEnabled: $gDeviceCaps.hostImportAlignment else: "off") &
    " matrix=" & (if gFp8CoopEnabled: "KHR_cooperative_matrix FP8-E4M3/FP32 + FP16/FP32"
      elif gCoopEnabled: "KHR_cooperative_matrix FP16/FP32" else: "scalar")
  stderr.writeLine "[vulkan] Khoi tao xong: instance+device+queue+commandPool san sang (queueFamily=" & $gQueueFamilyIndex & ")"

proc vulkanDeviceCaps*(): VulkanCaps =
  ensureInit()
  gDeviceCaps

proc vulkanTensorCoreAvailable*(): bool =
  ensureInit()
  gCoopEnabled

proc findMemoryType(typeFilter: uint32, properties: uint32): uint32 =
  var memProps: VkPhysicalDeviceMemoryProperties
  lib.vkGetPhysicalDeviceMemoryProperties(gPhysicalDevice, addr memProps)
  for i in 0 ..< int(memProps.memoryTypeCount):
    if (typeFilter and (1'u32 shl uint32(i))) != 0 and
       (memProps.memoryTypes[i].propertyFlags and properties) == properties:
      return uint32(i)
  gpuCheck(false, "Khong tim thay memory type phu hop (typeFilter=" & $typeFilter & ", can properties=" & $properties & ")")

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
# TEST TỐI GIẢN: vecAdd(a,b) = c, dùng để XÁC NHẬN toàn bộ pipeline
# instance->device->buffer->shader->pipeline->dispatch->readback chạy
# ĐÚNG trên máy thật, trước khi viết bất cứ kernel Q4 nào phức tạp hơn.
# Shader nguồn GLSL nằm ở kernels/vecadd.comp - PHẢI tự biên dịch sang
# SPIR-V bằng glslc/glslangValidator (có sẵn nếu đã cài Vulkan SDK) rồi
# copy file .spv ra CÙNG thư mục kernels/ với tên vecadd.spv:
#
#     glslc kernels/vecadd.comp -o kernels/vecadd.spv
#
# hoặc:
#
#     glslangValidator -V kernels/vecadd.comp -o kernels/vecadd.spv
# ---------------------------------------------------------------------
proc vulkanVecAddTest*(a, b: seq[float32]): seq[float32] =
  ensureInit()
  gpuCheck(a.len == b.len, "vulkanVecAddTest: a va b phai cung do dai")
  let n = a.len
  let bytes = uint64(n * sizeof(float32))

  let (bufA, memA) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufB, memB) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  let (bufC, memC) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  var aVar = a
  var bVar = b
  uploadToBuffer(memA, addr aVar[0], bytes)
  uploadToBuffer(memB, addr bVar[0], bytes)

  # ---- Nạp shader SPIR-V đã biên dịch sẵn (đọc lúc COMPILE Nim, giống
  # staticRead của Metal/CUDA - KHÔNG đọc file lúc runtime) ----
  let spirvPath = currentSourcePath().parentDir() / "kernels" / "vecadd.spv"
  gpuCheck(fileExists(spirvPath),
    "Chua thay " & spirvPath & " - can bien dich truoc: glslc kernels/vecadd.comp -o kernels/vecadd.spv")
  let spirvBytes = readFile(spirvPath)
  gpuCheck(spirvBytes.len mod 4 == 0, "File SPIR-V khong hop le (do dai khong chia het 4 byte)")

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

  # ---- Descriptor pool + set, trỏ 3 buffer vào 3 binding ----
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
  # local_size_x = 64 trong shader (xem vecadd.comp) - chia đủ nhóm để phủ n phần tử.
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
# KERNEL Q4 MATMUL THẬT - mirror chính xác thuật toán Metal/CUDA/OpenCL.
# Bản "immediate" (không session) - mỗi lần gọi tự upload/dispatch/đọc về,
# giống metal_matmul_q4 (bản non-session) chứ CHƯA có session API gộp
# nhiều lệnh như metal_matmul_q4_enc - đó là việc của bước sau (Bước 2/3),
# sau khi bản immediate này đã CHẠY ĐÚNG đã.
#
# BẮT BUỘC biên dịch shader trước:
#     glslc kernels/matmul_q4.comp -o kernels/matmul_q4.spv
# ---------------------------------------------------------------------
const MMQ4_NDST = 2  # SUA: 8 -> 2 (mirror ket qua da DO THAT tren Metal -
  # matmul_q4_kernel_ndst2_paired, xem giai thich day du trong
  # vecop_matmul.metal va kernels/matmul_q4.comp). PHAI dong bo VOI CA 2 noi
  # dung gia tri nay: #define MMQ4_NDST trong kernels/matmul_q4.comp (dong
  # tren cung file khong sua nhau tu dong - GLSL macro va Nim const la 2 the
  # gioi bien dich rieng biet). Neu lech nhau, rowBlocks tinh SAI -> ket qua
  # SAI/crash. CHUA CHAY THU TREN GPU VULKAN THAT.

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
    "Chua thay " & spirvPath & " - can bien dich truoc: glslc kernels/matmul_q4.comp -o kernels/matmul_q4.spv")
  let spirvBytes = readFile(spirvPath)
  gpuCheck(spirvBytes.len mod 4 == 0, "File SPIR-V khong hop le")

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

  # push constant: 6 int32 (M,K,N,groupSize,nGroupsPerRow,wqOff) = 24 byte, khớp
  # struct PushConsts trong matmul_q4.comp - SAI THỨ TỰ/KIỂU Ở ĐÂY SẼ ĐỌC
  # NHẦM GIÁ TRỊ bên shader (không lỗi compile, chỉ tính sai âm thầm).
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

  # push constants: PHẢI đúng thứ tự int32 khớp struct PushConsts trong .comp
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
# SESSION API ĐẦY ĐỦ - mirror kiến trúc session bên Metal (gSessionEncoder)
# nhưng dùng VkCommandBuffer làm đơn vị "session": sessionBegin() mở +
# bắt đầu record 1 command buffer, các hàm *Enc() RECORD lệnh vào đó
# (KHÔNG submit ngay), sessionEnd() kết thúc record + submit + CHỜ xong
# (fence) - ĐÚNG bài học rút ra từ Metal: sessionRead() chỉ an toàn SAU
# khi sessionEnd() đã hoàn tất, KHÔNG được đọc buffer giữa chừng session
# chưa commit.
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
var gSessionCmdBuf: VkCommandBuffer
var gSessionActive = false
var gSessionCleanupPools: seq[VkDescriptorPool] = @[]
# Internal-only session buffers (multiBuf dummy slots). Freed in
# vulkanSessionEnd. MUST NOT track buffers returned to callers that still
# read/free after End (vulkanMatmul/VecOp/Activation: AllocScratch → End →
# Read → Free) — that caused SIGSEGV use-after-free on first matmul.
var gSessionInternalBufs: seq[VulkanBufferHandle] = @[]
## Buffer bi sessionFree() KHI session con mo: command buffer dang ghi van tham chieu chung,
## destroy ngay => vkQueueSubmit dung VkBuffer da huy => SIGSEGV (Metal ref-count nen khong sao).
## Hoan huy toi sau khi vulkanSessionEnd da wait fence.
var gSessionDeferredFree: seq[VulkanBufferHandle] = @[]
var gVkLastError: string = ""   ## mo ta loi cua lan vulkanSessionEnd gan nhat that bai

proc getOrCreatePipeline(shaderName: string, numBindings: int, pushConstSize: uint32,
                         localSizeX = 0, spirvPathOverride = "", sharedFloats = 0): PipelineEntry =
  ## Cache pipeline theo tên shader + numBindings + localSizeX.
  ## Trước đây cacheKey chỉ gồm shaderName+localSizeX → cùng kernel với số
  ## binding khác nhau (microVm 5 vs multiBuf 3..5) trả về entry cũ →
  ## vkUpdateDescriptorSets ghi quá layout → UB / validation error.
  let cacheKey = shaderName & "_nb" & $numBindings &
    (if localSizeX > 0: "_local" & $localSizeX else: "") & "_sm" & $sharedFloats
  if gPipelineCache.hasKey(cacheKey):
    return gPipelineCache[cacheKey]

  let spirvPath = if spirvPathOverride.len > 0: spirvPathOverride
                  else: currentSourcePath().parentDir() / "kernels" / (shaderName & ".spv")
  gpuCheck(fileExists(spirvPath),
    "Chua thay " & spirvPath & " - can bien dich truoc: glslc kernels/" & shaderName & ".comp -o kernels/" & shaderName & ".spv")
  let spirvBytes = readFile(spirvPath)
  gpuCheck(spirvBytes.len mod 4 == 0, "File SPIR-V khong hop le: " & spirvPath)

  var shaderInfo = VkShaderModuleCreateInfo(
    sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext: nil, flags: 0,
    codeSize: uint(spirvBytes.len), pCode: cast[ptr uint32](unsafeAddr spirvBytes[0])
  )
  var shaderModule: VkShaderModule
  gpuCheck(lib.vkCreateShaderModule(gDevice, addr shaderInfo, nil, addr shaderModule) == VK_SUCCESS,
    "vkCreateShaderModule that bai cho " & shaderName)

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
    "vkCreateDescriptorSetLayout that bai cho " & shaderName)

  var pcRange = VkPushConstantRange(stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, offset: 0, size: pushConstSize)
  var pipelineLayoutInfo = VkPipelineLayoutCreateInfo(
    sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pNext: nil, flags: 0,
    setLayoutCount: 1, pSetLayouts: addr setLayout,
    pushConstantRangeCount: (if pushConstSize > 0: 1'u32 else: 0'u32),
    pPushConstantRanges: (if pushConstSize > 0: addr pcRange else: nil)
  )
  var pipelineLayout: VkPipelineLayout
  gpuCheck(lib.vkCreatePipelineLayout(gDevice, addr pipelineLayoutInfo, nil, addr pipelineLayout) == VK_SUCCESS,
    "vkCreatePipelineLayout that bai cho " & shaderName)

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
  gpuCheck(lib.vkCreateComputePipelines(gDevice, nil, 1, addr pipelineInfo, nil, addr pipeline) == VK_SUCCESS,
    "vkCreateComputePipelines that bai cho " & shaderName)

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
  ## 1 descriptor set trỏ đúng vào các buffer truyền vào. Set lấy từ pool
  ## của session (takeDescriptorPool) - pool chỉ bị destroy ở sessionEnd()
  ## sau khi fence xong, vì lệnh dispatch chưa chạy xong thì chưa được phép
  ## giải phóng set (giống lý do KHÔNG được sessionRead() giữa chừng session).
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
    # Offset vượt size buffer trước đây âm thầm dùng range=4 → đọc rác im lặng.
    # Fail-loud khi BYBY_VULKAN_STRICT=1; mặc định vẫn clamp nhưng log 1 lần.
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
  gSessionCleanupPools.setLen(0)
  gSessionInternalBufs.setLen(0)
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
  if lib.vkEndCommandBuffer(gSessionCmdBuf) != VK_SUCCESS:
    gVkLastError = "vkEndCommandBuffer that bai"
    return false
  var fenceInfo = VkFenceCreateInfo(sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: nil, flags: 0)
  var fence: VkFence
  if lib.vkCreateFence(gDevice, addr fenceInfo, nil, addr fence) != VK_SUCCESS:
    return false
  var cmdBufVar = gSessionCmdBuf
  var submitInfo = VkSubmitInfo(
    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext: nil,
    waitSemaphoreCount: 0, pWaitSemaphores: nil, pWaitDstStageMask: nil,
    commandBufferCount: 1, pCommandBuffers: addr cmdBufVar,
    signalSemaphoreCount: 0, pSignalSemaphores: nil
  )
  let submitOk = lib.vkQueueSubmit(gQueue, 1, addr submitInfo, fence) == VK_SUCCESS
  let waitOk = submitOk and (lib.vkWaitForFences(gDevice, 1, addr fence, 1, high(uint64)) == VK_SUCCESS)
  if not submitOk: gVkLastError = "vkQueueSubmit that bai"
  elif not waitOk: gVkLastError = "vkWaitForFences that bai"
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
  for i in 0 ..< gSessionInternalBufs.len:
    let h = gSessionInternalBufs[i]
    if h.buf != nil:
      lib.vkDestroyBuffer(gDevice, h.buf, nil)
    if h.mem != nil:
      lib.vkFreeMemory(gDevice, h.mem, nil)
  gSessionInternalBufs.setLen(0)
  # Huy cac buffer bi sessionFree() giua session (an toan vi da wait fence xong).
  for i in 0 ..< gSessionDeferredFree.len:
    let h = gSessionDeferredFree[i]
    if h.buf != nil: lib.vkDestroyBuffer(gDevice, h.buf, nil)
    if h.mem != nil: lib.vkFreeMemory(gDevice, h.mem, nil)
  gSessionDeferredFree.setLen(0)
  result = waitOk

proc vulkanGetLastError*(): string = gVkLastError

proc vulkanSessionCommitContinue*(): bool =
  ## Tuong duong metalSessionCommitContinue: chot command buffer hien tai (cho GPU xong) roi mo
  ## session moi de caller tiep tuc encode. Vulkan khong co "commit khong cho" trong driver nay.
  if not gSessionActive: return false
  if not vulkanSessionEnd(): return false
  result = vulkanSessionBegin()

proc vulkanRecoverQueue*(): bool =
  ## Vulkan khong co khai niem "queue bi poisoned" nhu MTLCommandQueue. Neu 1 session bi ket
  ## (gSessionActive = true sau loi), huy no de session ke tiep mo lai duoc. Tra true = san sang.
  if gSessionActive:
    gSessionActive = false
    discard lib.vkEndCommandBuffer(gSessionCmdBuf)
    var cb = gSessionCmdBuf
    lib.vkFreeCommandBuffers(gDevice, gCommandPool, 1, addr cb)
    if gDSetPool != nil:
      lib.vkDestroyDescriptorPool(gDevice, gDSetPool, nil)
      gDSetPool = nil
      gDSetPoolLeft = 0
    for p in gSessionCleanupPools:
      lib.vkDestroyDescriptorPool(gDevice, p, nil)
    gSessionCleanupPools.setLen(0)
    for i in 0 ..< gSessionInternalBufs.len:
      let h = gSessionInternalBufs[i]
      if h.buf != nil: lib.vkDestroyBuffer(gDevice, h.buf, nil)
      if h.mem != nil: lib.vkFreeMemory(gDevice, h.mem, nil)
    gSessionInternalBufs.setLen(0)
    for i in 0 ..< gSessionDeferredFree.len:
      let h = gSessionDeferredFree[i]
      if h.buf != nil: lib.vkDestroyBuffer(gDevice, h.buf, nil)
      if h.mem != nil: lib.vkFreeMemory(gDevice, h.mem, nil)
    gSessionDeferredFree.setLen(0)
  result = true

proc vulkanSessionUpload*(data: seq[float32]): VulkanBufferHandle =
  ensureInit()
  let bytes = uint64(data.len * sizeof(float32))
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
  let (buf, mem) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: bytes)

proc registerInternalScratch*(h: VulkanBufferHandle) =
  ## multiBuf dummy only — destroyed in vulkanSessionEnd. Do not register
  ## buffers the caller still reads after End.
  if gSessionActive:
    gSessionInternalBufs.add(h)

proc vulkanSessionRead*(h: VulkanBufferHandle, n: int): seq[float32] =
  ## CHỈ được gọi SAU vulkanSessionEnd() đã hoàn tất (đã wait fence) - đọc
  ## giữa chừng session chưa commit sẽ ra dữ liệu rác/cũ (bài học từ Metal).
  result = newSeq[float32](n)
  if n > 0:
    readFromBuffer(h.mem, addr result[0], uint64(n * sizeof(float32)))

proc vulkanSessionFree*(h: var VulkanBufferHandle) =
  if gSessionActive and (h.buf != nil or h.mem != nil):
    gSessionDeferredFree.add(h)   # van co the nam trong command buffer dang ghi -> hoan huy
    h.buf = nil
    h.mem = nil
    return
  if h.buf != nil:
    lib.vkDestroyBuffer(gDevice, h.buf, nil)
  if h.mem != nil:
    lib.vkFreeMemory(gDevice, h.mem, nil)
  h.buf = nil
  h.mem = nil

proc vulkanSessionCopyInto*(src: VulkanBufferHandle, dst: var VulkanBufferHandle,
                             srcOffsetElems, dstOffsetElems, n: int): bool =
  if not gSessionActive: return false
  var region = VkBufferCopy(
    srcOffset: uint64(srcOffsetElems * sizeof(float32)),
    dstOffset: uint64(dstOffsetElems * sizeof(float32)),
    size: uint64(n * sizeof(float32))
  )
  # Barrier: compute/transfer truoc -> transfer nay (src vua duoc shader ghi, dst vua duoc doc/ghi)
  var b1 = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: nil,
    srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT or VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT,
    dstAccessMask: VK_ACCESS_TRANSFER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT)
  lib.vkCmdPipelineBarrier(gSessionCmdBuf,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT or VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, addr b1, 0, nil, 0, nil)
  lib.vkCmdCopyBuffer(gSessionCmdBuf, src.buf, dst.buf, 1, addr region)
  # Barrier: transfer -> compute / transfer ke tiep
  var b2 = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: nil,
    srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
    dstAccessMask: VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT or
                   VK_ACCESS_TRANSFER_READ_BIT or VK_ACCESS_TRANSFER_WRITE_BIT)
  lib.vkCmdPipelineBarrier(gSessionCmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT or VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, addr b2, 0, nil, 0, nil)
  result = true

proc uploadResidentData(data: pointer; nBytes: uint64): VulkanBufferHandle =
  ensureInit()
  let localMode = getEnv("BYBY_VULKAN_DEVICE_LOCAL", "auto")
  let deviceLocal = useDeviceLocalWeights(gDeviceCaps.properties.deviceType, localMode)
  let allocBytes = max(4'u64, (nBytes + 3) and not 3'u64)
  let flags = if deviceLocal: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
              else: VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  let (buf, mem) = createHostVisibleBuffer(allocBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, flags)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: nBytes)
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
  uploadResidentData((if data.len > 0: cast[pointer](unsafeAddr data[0]) else: nil),
    uint64(data.len * sizeof(float32)))

proc vulkanSessionUploadBytesPrivate*(data: openArray[uint8]): VulkanBufferHandle =
  uploadResidentData((if data.len > 0: cast[pointer](unsafeAddr data[0]) else: nil), uint64(data.len))

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
  lib.vkCmdBindPipeline(gSessionCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, entry.pipeline)
  var dsetVar = dset
  lib.vkCmdBindDescriptorSets(gSessionCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, entry.layout, 0, 1, addr dsetVar, 0, nil)
  if pushConstSize > 0:
    lib.vkCmdPushConstants(gSessionCmdBuf, entry.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstSize, pushConstBytes)
  lib.vkCmdDispatch(gSessionCmdBuf, groupX, groupY, groupZ)
  # Metal's command encoder orders compute dispatches with a buffer barrier;
  # Vulkan requires an explicit memory dependency for write->read visibility.
  var barrier = VkMemoryBarrier(sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    pNext: nil, srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
    dstAccessMask: VK_ACCESS_SHADER_READ_BIT or VK_ACCESS_SHADER_WRITE_BIT)
  lib.vkCmdPipelineBarrier(gSessionCmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, addr barrier, 0, nil, 0, nil)
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
    let target = if source.contains("GL_KHR_cooperative_matrix"): " --target-env=vulkan1.3" else: ""
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
  ## matmul_fp4km_tiled.glsl: format 1 = packed FP4-KM, 0 = FP32 rows.
  const templateSrc = staticRead("kernels/matmul_fp4km_tiled.glsl")
  let name = if weightFormat == 1: "coop_matmul_fp4km_tiled" else: "coop_matmul_f32_tiled"
  (name, "#version 450\n#define WEIGHT_FORMAT " & $weightFormat & "\n" &
    templateSrc.replace("#version 450\n", ""))

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
    if gDeviceCaps.subgroupSize == 32 and gDeviceCaps.maxThreads >= 128:
      result.add tiledMatmulShader(1)
      result.add tiledMatmulShader(0)

proc tryCooperativeMatmul(a, w, c, lut, zeros: VulkanBufferHandle;
                         M, K, N, weightFormat: int; groupSize = 0;
                         groupsPerRow = 0): bool =
  ## Inference GEMM only. GEMV keeps its bandwidth-oriented scalar kernel.
  if not gCoopEnabled or not gDeviceCaps.cooperativeMatrix or not gSessionActive or
      M < 8 or K < 64 or N < 16: return false
  if a.buf == nil or w.buf == nil or c.buf == nil: return false
  if a.sizeBytes < uint64(M) * uint64(K) * 4 or c.sizeBytes < uint64(M) * uint64(N) * 4:
    return false
  let weightBytes = case weightFormat
    of 1: N * (K div 2) + N * (K div 32)
    of 2: N * ((K + 1) div 2)
    else: N * K * 4
  if weightBytes mod 4 != 0 or w.sizeBytes < uint64(weightBytes): return false
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
  var pc = [uint32(M), uint32(K), uint32(N), uint32(groupSize), uint32(groupsPerRow)]
  dispatchGeneric(kn, [a, w, c, lut, zeros], addr pc[0], 20,
    uint32((N + 15) div 16), uint32((M + 15) div 16), 1,
    gDeviceCaps.subgroupSize, spv)

proc tryFp4KmFp8Cooperative(a, w, c, lut: VulkanBufferHandle;
                            M, K, N: int): bool =
  ## Packed FP4-KM -> E4M3 is converted only for the current 16x16 MMA tile.
  ## No full-size dequantized weight buffer is created.
  if not gFp8CoopEnabled or not gSessionActive or M < 8 or K < 64 or N < 16:
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

proc tryTiledCooperative(a, w, c, lut: VulkanBufferHandle; M, K, N, weightFormat: int): bool =
  ## matmul_fp4km_tiled: weights converted once per 64 output rows instead of
  ## once per 16x16 tile. False when the device/shape does not fit its contract.
  if not gCoopEnabled or not gDeviceCaps.cooperativeMatrix or not gSessionActive or
      gDeviceCaps.subgroupSize != 32 or gDeviceCaps.maxThreads < 128 or
      M < 8 or K mod 64 != 0 or N < 16: return false
  if a.buf == nil or w.buf == nil or c.buf == nil or lut.buf == nil: return false
  let weightBytes = if weightFormat == 1: N * (K div 2) + N * (K div 32) else: N * K * 4
  if w.sizeBytes < uint64(weightBytes) or a.sizeBytes < uint64(M * K * 4) or
      c.sizeBytes < uint64(M * N * 4) or
      (weightFormat == 1 and lut.sizeBytes < 272'u64 * 4): return false
  let (kn, source) = tiledMatmulShader(weightFormat)
  var spv: string
  try:
    spv = compileVulkanJitShader(kn, source)
  except CatchableError as e:
    if weightFormat == 1: gFp4TiledEnabled = false else: gF32TiledEnabled = false
    stderr.writeLine "[vulkan] tiled cooperative shader unavailable: " & e.msg
    return false
  var pc = [uint32(M), uint32(K), uint32(N), 0'u32, 0'u32]
  dispatchGeneric(kn, [a, w, c, lut], addr pc[0], 20,
    uint32((N + 63) div 64), uint32((M + 63) div 64), 1, 128, spv)

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
  # THÊM: m==1 = GEMV - tile 16x16 ben duoi dispatch groupY=ceil(1/16)=1,
  # tuc chi 1/16 invocation/workgroup ra ket qua dung (giong het bug da fix
  # ben Metal matmul_kernel_t8, xem giai thich chi tiet o metal_shim.m).
  # Dung shader matmul_gemv.comp (subgroupAdd, khong barrier) khi M=1.
  if M == 1:
    dispatchGeneric("matmul_gemv", [a, b, c], addr pc[0], 12, uint32(N), 1, 1)
  else:
    const TILE = 16'u32
    dispatchGeneric("matmul", [a, b, c], addr pc[0], 12,
                    (uint32(N) + TILE - 1) div TILE, (uint32(M) + TILE - 1) div TILE, 1)

proc vulkanSessionMatmulNTEnc*(a, b: VulkanBufferHandle, c: var VulkanBufferHandle, M, K, N: int;
                               bOffsetElems = 0): bool =
  ## C(M x N) = A(M x K) @ B(N x K)^T - dung shader rieng matmul_nt.comp
  ## (doc B theo layout [N,K] tu nhien, khong transpose CPU truoc).
  ## bOffsetElems: B starts that many floats into its buffer.
  var pc = [int32(M), int32(K), int32(N), int32(bOffsetElems)]
  # THÊM: m==1 GEMV, xem giai thich o vulkanSessionMatmulEnc trên.
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
  ## Mirror attention_fused_backward_kernel (Metal) qua kernels/
  ## attention_fused_backward.comp - xem chi tiet trong file .comp. Dispatch
  ## 1 thread/(bh,ti): groupCount=(B*H,S,1) voi local_size=(1,1,1) trong
  ## shader, GIONG HET dispatchThreads cua Metal (khong group toi uu gi).
  var pc = (B: int32(B), H: int32(H), S: int32(S), D: int32(D), scale: scale)
  dispatchGeneric("attention_fused_backward", [q, k, v, sMatrix, dy, dq, dk, dv], addr pc, 20,
                  uint32(B * H), uint32(S), 1)

proc vulkanSessionAttentionDecodeGqaEnc*(q, k, v: VulkanBufferHandle, o, scores: var VulkanBufferHandle,
                                          H, S, D: int, scale: float32): bool =
  ## Mirror attention_decode_gqa_kernel.cu (CUDA)/Metal - 1 workgroup(128
  ## thread)/query head. CANH BAO: mirror tu ban CUDA CHUA CHAY PHAN CUNG
  ## THAT luc viet - PHAI verify bang BYBY_VERIFY_ATTN_DECODE=1 (xem
  ## nim_inference.nim) truoc khi tin dung ket qua tren may that.
  var pc = (H: int32(H), S: int32(S), D: int32(D), scale: scale)
  dispatchGeneric("attention_decode_gqa", [q, k, v, o, scores], addr pc, 16,
                  uint32(H), 1, 1)

const attnAllHeadsGlsl = staticRead(currentSourcePath().parentDir() / "kernels" / "attention_decode_all_heads.comp")
  ## Nhung san ma GLSL luc compile Nim -> neu chua co file .spv thi tu bien dich
  ## bang glslc luc chay (cache trong thu muc temp), khong bat buoc build tay.
var gAttnAllHeadsSpv = ""        ## duong dan .spv da chot ("" = chua chot)
var gAttnAllHeadsBroken = false  ## true = khong co .spv va glslc fail -> caller dung duong cu

proc vulkanSessionAttentionDecodeAllHeadsEnc*(q, kAll, vAll: VulkanBufferHandle,
                                               o: var VulkanBufferHandle,
                                               H, HKV, D, S, maxSeqLen: int): bool =
  ## Attention decode FUSED: TOAN BO H head trong DUNG 1 dispatch (1 workgroup/head).
  ## Mirror attention_decode_all_heads_kernel (Metal). Tra false (KHONG raise) khi
  ## khong chay duoc -> caller roi ve vong lap tung nhom head.
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
        stderr.writeLine "[vulkan] attention_decode_all_heads: khong co .spv va glslc fail (" &
          e.msg.splitLines()[0] & ") -> dung duong cu tung head"
        gAttnAllHeadsBroken = true
        return false
  var pc = [int32(H), int32(HKV), int32(D), int32(S), int32(maxSeqLen)]
  result = dispatchGeneric("attention_decode_all_heads", [q, kAll, vAll, o], addr pc[0], 20,
                           uint32(H), 1, 1, 0, gAttnAllHeadsSpv)

proc vulkanAvailable*(): bool =
  ## Dùng cho detectBackend() bên gpubackend.nim - thử load+init, bắt lỗi
  ## an toàn (KHÔNG raise ra ngoài) nếu máy không có Vulkan/MoltenVK.
  try:
    ensureInit()
    result = true
  except CatchableError:
    result = false

proc vulkanSessionUploadBytes*(data: seq[uint8]): VulkanBufferHandle =
  ## Ban byte tho (khac vulkanSessionUpload chi nhan seq[float32]) - dung
  ## cho weight int4 da pack (wq), mirror openclSessionUploadBytes.
  ensureInit()
  let bytes = uint64(data.len)
  let (buf, mem) = createHostVisibleBuffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
  if data.len > 0:
    var dataVar = data
    uploadToBuffer(mem, addr dataVar[0], bytes)
  result = VulkanBufferHandle(buf: buf, mem: mem, sizeBytes: bytes)

proc vulkanSessionActivationEnc*(opCode: int, x: VulkanBufferHandle, y: var VulkanBufferHandle, n: int, p: float32 = 0'f32): bool =
  ## PC layout phai khop struct PC {int op; int n; float p;} trong
  ## activation.comp (12 byte) - truoc day chi day 8 byte (thieu p) nen
  ## opcode 7 (pow) tren duong session khong bao gio nhan dung so mu.
  var pc: tuple[op, n: int32, p: float32] = (int32(opCode), int32(n), p)
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("activation", [x, y], addr pc, 12, groups, 1, 1)
# =======================================================================
# HIGH-LEVEL ONE-SHOT WRAPPERS - mirror interface cua metal_backend.nim va
# gpubackend.nim de cung cap API thong nhat cho Vulkan. Cac ham co shader
# .comp san thi dung GPU qua session encoder; cac ham chua co shader rieng
# van dung CPU fallback theo chinh sach cua caller.

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
  doAssert ndim <= 8, "vulkanSessionPermuteEnc: rank > 8 chưa hỗ trợ (mảng push-constant cố định 8)"
  var pc: tuple[newStrides: array[8, int32], oldStrides: array[8, int32],
                dimsMap: array[8, int32], ndim: int32, n: int32]
  for i in 0 ..< 8:
    pc.newStrides[i] = if i < ndim: newStrides[i] else: 1
    pc.oldStrides[i] = if i < ndim: oldStrides[i] else: 0
    pc.dimsMap[i] = if i < ndim: dimsMap[i] else: 0
  pc.ndim = int32(ndim); pc.n = int32(n)
  let groups = uint32((n + 255) div 256)
  dispatchGeneric("permute", [src, dst], addr pc, uint32(sizeof(pc)), groups, 1, 1)

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
  ## GPU THAT cho tat ca opcode - activation.comp da co san 0=relu,1=sigmoid,
  ## 2=tanh,4=exp,5=log,6=sqrt,7=pow (voi push-constant p rieng cho pow).
  ## KHONG fallback CPU cho bat ky op nao trong danh sach nay - neu dispatch
  ## GPU that bai thi raise loi that (qua gpuCheck) thay vi am tham tinh CPU.
  let opCode =
    case op
    of "relu": 0
    of "sigmoid": 1
    of "tanh": 2
    of "exp": 4
    of "log": 5
    of "sqrt": 6
    of "pow": 7
    else: raise newException(ValueError, "vulkanActivation: op khong ho tro: " & op)
  let n = x.len
  gpuCheck(vulkanSessionBegin(), "vulkanActivation: session begin that bai")
  var bufX = vulkanSessionUpload(x)
  var bufY = vulkanSessionAllocScratch(n)
  let ok = vulkanSessionActivationEnc(opCode, bufX, bufY, n, p)
  gpuCheck(vulkanSessionEnd(), "vulkanActivation: session end that bai")
  gpuCheck(ok, "vulkanActivation: dispatch GPU that bai cho op=" & op & " (khong fallback CPU)")
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
  ## CPU fallback - chua co kernels/layernorm.comp (chi moi co rmsnorm.comp).
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
  ## CPU fallback - cong thuc chuan cua LayerNorm backward, giong ket qua
  ## voi cudaLayernormBackward/metalLayernormBackward.
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
  ## CPU fallback - phep gather don gian, khong can GPU cung nhanh.
  let num = indices.len
  result = newSeq[float32](num * dim)
  for i in 0 ..< num:
    let idx = int(indices[i])
    for d in 0 ..< dim:
      result[i * dim + d] = table[idx * dim + d]

proc vulkanApflu*(x: seq[float32], alpha, beta: float32): seq[float32] =
  ## CPU fallback - cong thuc y het vecop_apflu trong cuda kernels/vecop.ptx:
  ## x>0 -> x*(1+alpha*x); x<=0 -> beta*x*exp(x).
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    let v = x[i]
    if v > 0.0'f32:
      result[i] = v * (1.0'f32 + alpha * v)
    else:
      result[i] = beta * v * exp(v)

proc vulkanApfluBackward*(x, dy: seq[float32], alpha, beta: float32): seq[float32] =
  ## CPU fallback - dao ham cua APFLU: x>0 -> dy*(1+2*alpha*x);
  ## x<=0 -> dy*beta*exp(x)*(1+x).
  result = newSeq[float32](x.len)
  for i in 0 ..< x.len:
    let v = x[i]
    if v > 0.0'f32:
      result[i] = dy[i] * (1.0'f32 + 2.0'f32 * alpha * v)
    else:
      result[i] = dy[i] * beta * exp(v) * (1.0'f32 + v)

proc vulkanAttentionFused*(q, k, v, mask: seq[float32], B, H, S, D: int, scale: float32): tuple[o, s_matrix: seq[float32]] =
  ## Dung that GPU qua kernels/attention_fused.comp (session Enc) - mask
  ## KHONG duoc dung toi (giong het cudaAttentionFused/metalAttentionFused,
  ## shader tu lam causal mask ben trong).
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
  ## SUA: dung that GPU qua kernels/attention_fused_backward.comp (session
  ## Enc, mirror 1:1 metal_attention_fused_backward - xem giai thich trong
  ## file .comp), KHONG con la vong lap CPU thuan nup danh GPU nhu truoc.
  ## Cung gioi han S<=256 nhu ban Metal (mang local softmaxW[256]/
  ## dSoftmax[256] trong shader) - kiem tra tuong minh o day, GIONG HET cach
  ## metal_shim.m tra ve 0/false khi S>256, de bao loi ro rang thay vi doc/
  ## ghi tran mang trong shader.
  if S > 256:
    raise newException(ValueError,
      "vulkanAttentionFusedBackward: S=" & $S & " vuot gioi han 256 cua kernel (mirror gioi han cua ban Metal)")
  let qkvLen = B * H * S * D
  let sLen = B * H * S * S
  gpuCheck(vulkanSessionBegin(), "vulkanAttentionFusedBackward: session begin that bai")
  var bufQ = vulkanSessionUpload(q)
  var bufK = vulkanSessionUpload(k)
  var bufV = vulkanSessionUpload(v)
  var bufS = vulkanSessionUpload(s_matrix)
  var bufDy = vulkanSessionUpload(dy)
  # dq/dk/dv: shader dung "+=" (dq) / atomic CAS-add (dk, dv) -> PHAI
  # zero-init truoc khi dispatch, GIONG HET memset trong metal_shim.m -
  # dung vulkanSessionUpload voi seq zero thay vi vulkanSessionAllocScratch
  # (khong dam bao zero) de chac chan buffer sach truoc khi shader cong don.
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
# GPU HOÁ THẬT (vòng 2): GATHER / ARGMAX-INDEX / COMPARE / DROPOUT-RNG
# Shader: kernels/gather.comp, argmax.comp, compare.comp, dropout_rng.comp
# - compile tay bằng glslc trước khi chạy (xem ghi chú getOrCreatePipeline
# ở trên: tự tìm file kernels/<shaderName>.spv theo tên). Cùng session
# Begin/Enc/End/Read pattern như vulkanVecOp/vulkanMatmul ở trên.
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
  ## argmax.comp: local_size_x=1024, 1 workgroup / cặp (o,i), yêu cầu
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
  let idxF = vulkanSessionRead(bufIdx, pairs)  # đọc lại dạng float32 rồi reinterpret bytes
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
# GPU HOÁ THẬT (vòng 3) - GIỐNG TORCH: permute/split/concat không còn build
# mảng chỉ số O(n) trên CPU. permute.comp/split_slice.comp/concat_gather.comp
# tự giải mã chỉ số trên GPU từ bảng nhỏ O(rank)/O(numTensors).
# =======================================================================

proc vulkanPermute*(src: seq[float32], newStrides, oldStrides, dimsMap: seq[int32], n: int): seq[float32] =
  let ndim = newStrides.len
  doAssert ndim <= 8, "vulkanPermute: rank > 8 chưa hỗ trợ (mảng push-constant cố định 8)"
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
  ## SPIR-V không có mảng con trỏ device -> dùng 1 buffer nguồn ĐÃ NỐI PHẲNG
  ## (flatSrc, host chỉ memcpy nối tiếp - KHÔNG tính chỉ số per-element) +
  ## bảng nhỏ O(numTensors) (srcBase/srcDimSize/srcDimOffset).
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
# Yêu cầu: micro_vm.spv đã load (compile micro_vm.comp).
# Dùng dispatchGeneric pattern nếu pipeline "micro_vm" đã đăng ký.

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
      # Caller offElems: float/u32 = element index; MemU8 = byte offset/4 neu theo
      # quy uoc Metal (float*4). Weight/LUT thuong off=0.
      # offElems = chi so float32 (khop Metal setBuffer offset/4). Weight U8 luon 0.
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
