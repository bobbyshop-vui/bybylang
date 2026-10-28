## Vulkan capability queries; layouts match the Vulkan 1.0/KHR ABI.
## No SDK headers or CUDA runtime are needed to build this module.
import std/sets

type
  VkPhysicalDeviceLimits* = object
    maxImageDimension1D*: uint32
    maxImageDimension2D*: uint32
    maxImageDimension3D*: uint32
    maxImageDimensionCube*: uint32
    maxImageArrayLayers*: uint32
    maxTexelBufferElements*: uint32
    maxUniformBufferRange*: uint32
    maxStorageBufferRange*: uint32
    maxPushConstantsSize*: uint32
    maxMemoryAllocationCount*: uint32
    maxSamplerAllocationCount*: uint32
    bufferImageGranularity*: uint64
    sparseAddressSpaceSize*: uint64
    maxBoundDescriptorSets*: uint32
    maxPerStageDescriptorSamplers*: uint32
    maxPerStageDescriptorUniformBuffers*: uint32
    maxPerStageDescriptorStorageBuffers*: uint32
    maxPerStageDescriptorSampledImages*: uint32
    maxPerStageDescriptorStorageImages*: uint32
    maxPerStageDescriptorInputAttachments*: uint32
    maxPerStageResources*: uint32
    maxDescriptorSetSamplers*: uint32
    maxDescriptorSetUniformBuffers*: uint32
    maxDescriptorSetUniformBuffersDynamic*: uint32
    maxDescriptorSetStorageBuffers*: uint32
    maxDescriptorSetStorageBuffersDynamic*: uint32
    maxDescriptorSetSampledImages*: uint32
    maxDescriptorSetStorageImages*: uint32
    maxDescriptorSetInputAttachments*: uint32
    maxVertexInputAttributes*: uint32
    maxVertexInputBindings*: uint32
    maxVertexInputAttributeOffset*: uint32
    maxVertexInputBindingStride*: uint32
    maxVertexOutputComponents*: uint32
    maxTessellationGenerationLevel*: uint32
    maxTessellationPatchSize*: uint32
    maxTessellationControlPerVertexInputComponents*: uint32
    maxTessellationControlPerVertexOutputComponents*: uint32
    maxTessellationControlPerPatchOutputComponents*: uint32
    maxTessellationControlTotalOutputComponents*: uint32
    maxTessellationEvaluationInputComponents*: uint32
    maxTessellationEvaluationOutputComponents*: uint32
    maxGeometryShaderInvocations*: uint32
    maxGeometryInputComponents*: uint32
    maxGeometryOutputComponents*: uint32
    maxGeometryOutputVertices*: uint32
    maxGeometryTotalOutputComponents*: uint32
    maxFragmentInputComponents*: uint32
    maxFragmentOutputAttachments*: uint32
    maxFragmentDualSrcAttachments*: uint32
    maxFragmentCombinedOutputResources*: uint32
    maxComputeSharedMemorySize*: uint32
    maxComputeWorkGroupCount*: array[3, uint32]
    maxComputeWorkGroupInvocations*: uint32
    maxComputeWorkGroupSize*: array[3, uint32]
    subPixelPrecisionBits*: uint32
    subTexelPrecisionBits*: uint32
    mipmapPrecisionBits*: uint32
    maxDrawIndexedIndexValue*: uint32
    maxDrawIndirectCount*: uint32
    maxSamplerLodBias*: float32
    maxSamplerAnisotropy*: float32
    maxViewports*: uint32
    maxViewportDimensions*: array[2, uint32]
    viewportBoundsRange*: array[2, float32]
    viewportSubPixelBits*: uint32
    minMemoryMapAlignment*: uint
    minTexelBufferOffsetAlignment*: uint64
    minUniformBufferOffsetAlignment*: uint64
    minStorageBufferOffsetAlignment*: uint64
    minTexelOffset*: int32
    maxTexelOffset*: uint32
    minTexelGatherOffset*: int32
    maxTexelGatherOffset*: uint32
    minInterpolationOffset*: float32
    maxInterpolationOffset*: float32
    subPixelInterpolationOffsetBits*: uint32
    maxFramebufferWidth*: uint32
    maxFramebufferHeight*: uint32
    maxFramebufferLayers*: uint32
    framebufferColorSampleCounts*: uint32
    framebufferDepthSampleCounts*: uint32
    framebufferStencilSampleCounts*: uint32
    framebufferNoAttachmentsSampleCounts*: uint32
    maxColorAttachments*: uint32
    sampledImageColorSampleCounts*: uint32
    sampledImageIntegerSampleCounts*: uint32
    sampledImageDepthSampleCounts*: uint32
    sampledImageStencilSampleCounts*: uint32
    storageImageSampleCounts*: uint32
    maxSampleMaskWords*: uint32
    timestampComputeAndGraphics*: uint32
    timestampPeriod*: float32
    maxClipDistances*: uint32
    maxCullDistances*: uint32
    maxCombinedClipAndCullDistances*: uint32
    discreteQueuePriorities*: uint32
    pointSizeRange*: array[2, float32]
    lineWidthRange*: array[2, float32]
    pointSizeGranularity*: float32
    lineWidthGranularity*: float32
    strictLines*: uint32
    standardSampleLocations*: uint32
    optimalBufferCopyOffsetAlignment*: uint64
    optimalBufferCopyRowPitchAlignment*: uint64
    nonCoherentAtomSize*: uint64
  VkPhysicalDeviceProperties* = object
    apiVersion*, driverVersion*, vendorID*, deviceID*, deviceType*: uint32
    deviceName*: array[256, char]
    pipelineCacheUUID*: array[16, uint8]
    limits*: VkPhysicalDeviceLimits
    sparseProperties*: array[5, uint32]
  VkPhysicalDeviceProperties2 = object
    sType: uint32
    pNext: pointer
    properties: VkPhysicalDeviceProperties
  VkSubgroupProperties = object
    sType: uint32
    pNext: pointer
    subgroupSize, supportedStages, supportedOperations, quadOperationsInAllStages: uint32
  VkCoopProperties = object
    sType: uint32
    pNext: pointer
    supportedStages: uint32
  VkFeatures2 = object
    sType: uint32
    pNext: pointer
    features: array[55, uint32]
  VkFloat16Int8Features* = object
    sType*: uint32
    pNext*: pointer
    shaderFloat16*, shaderInt8*: uint32
  VkStorage8Features* = object
    sType*: uint32
    pNext*: pointer
    storageBuffer8BitAccess*, uniformAndStorageBuffer8BitAccess*, storagePushConstant8*: uint32
  VkCoopFeatures* = object
    sType*: uint32
    pNext*: pointer
    cooperativeMatrix*, cooperativeMatrixRobustBufferAccess*: uint32
  VkMemoryModelFeatures* = object
    sType*: uint32
    pNext*: pointer
    vulkanMemoryModel*, vulkanMemoryModelDeviceScope*, vulkanMemoryModelAvailabilityVisibilityChains*: uint32
  VkCoopTile = object
    sType: uint32
    pNext: pointer
    m, n, k, aType, bType, cType, resultType, saturating, scope: uint32
  VkExtensionProperties = object
    extensionName: array[256, char]
    specVersion: uint32
  VkProcResolver* = proc(instance: pointer; name: cstring): pointer {.cdecl.}
  VulkanCaps* = object
    properties*: VkPhysicalDeviceProperties
    extensions*: HashSet[string]
    subgroupSize*: int
    float16*, int8*, storage8*, memoryModel*, cooperativeMatrix*: bool
    tileM*, tileN*, tileK*: int
    maxThreads*, maxSharedBytes*: int
    minStorageAlignment*: uint64

const
  Float16Int8Type* = 1000082000'u32
  Storage8Type* = 1000177000'u32
  CoopFeaturesType* = 1000506000'u32
  MemoryModelType* = 1000211000'u32

proc queryVulkanCaps*(resolve: VkProcResolver; instance, device: pointer): VulkanCaps =
  let getProps = cast[proc(d: pointer; p: ptr VkPhysicalDeviceProperties) {.cdecl.}](
    resolve(instance, "vkGetPhysicalDeviceProperties"))
  if getProps == nil: raise newException(ValueError, "Vulkan properties query is unavailable")
  getProps(device, addr result.properties)
  result.maxThreads = min(int(result.properties.limits.maxComputeWorkGroupInvocations),
    int(result.properties.limits.maxComputeWorkGroupSize[0]))
  result.maxSharedBytes = int(result.properties.limits.maxComputeSharedMemorySize)
  result.minStorageAlignment = result.properties.limits.minStorageBufferOffsetAlignment
  result.extensions = initHashSet[string]()
  let enumExt = cast[proc(d: pointer; layer: cstring; count: ptr uint32;
    p: ptr VkExtensionProperties): int32 {.cdecl.}](resolve(instance, "vkEnumerateDeviceExtensionProperties"))
  if enumExt != nil:
    var count: uint32
    if enumExt(device, nil, addr count, nil) == 0 and count > 0:
      var ext = newSeq[VkExtensionProperties](int(count))
      if enumExt(device, nil, addr count, addr ext[0]) == 0:
        for e in ext: result.extensions.incl($cast[cstring](unsafeAddr e.extensionName[0]))
  let hasCoop = "VK_KHR_cooperative_matrix" in result.extensions
  let getProps2 = cast[proc(d: pointer; p: ptr VkPhysicalDeviceProperties2) {.cdecl.}](
    resolve(instance, "vkGetPhysicalDeviceProperties2"))
  if getProps2 != nil:
    var coopProps = VkCoopProperties(sType: 1000506002)
    var subgroup = VkSubgroupProperties(sType: 1000094000,
      pNext: (if hasCoop: addr coopProps else: nil))
    var props = VkPhysicalDeviceProperties2(sType: 1000059001, pNext: addr subgroup)
    getProps2(device, addr props)
    if (subgroup.supportedStages and 0x20) != 0: result.subgroupSize = int(subgroup.subgroupSize)
    if hasCoop and (coopProps.supportedStages and 0x20) == 0: return
  let getFeatures2 = cast[proc(d: pointer; p: ptr VkFeatures2) {.cdecl.}](
    resolve(instance, "vkGetPhysicalDeviceFeatures2"))
  if getFeatures2 == nil: return
  var coop = VkCoopFeatures(sType: CoopFeaturesType)
  var memoryModel = VkMemoryModelFeatures(sType: MemoryModelType,
    pNext: (if hasCoop: addr coop else: nil))
  var storage = VkStorage8Features(sType: Storage8Type,
    pNext: addr memoryModel)
  var fp = VkFloat16Int8Features(sType: Float16Int8Type, pNext: addr storage)
  var features = VkFeatures2(sType: 1000059000, pNext: addr fp)
  getFeatures2(device, addr features)
  result.float16 = fp.shaderFloat16 != 0
  result.int8 = fp.shaderInt8 != 0
  result.storage8 = storage.storageBuffer8BitAccess != 0
  result.memoryModel = memoryModel.vulkanMemoryModel != 0
  if result.properties.apiVersion < 0x00403000'u32 or not hasCoop or
     coop.cooperativeMatrix == 0 or not result.float16 or not result.memoryModel or
     result.subgroupSize <= 0: return
  let getTiles = cast[proc(d: pointer; count: ptr uint32; p: ptr VkCoopTile): int32 {.cdecl.}](
    resolve(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"))
  if getTiles == nil: return
  var count: uint32
  if getTiles(device, addr count, nil) != 0 or count == 0: return
  var tiles = newSeq[VkCoopTile](int(count))
  for t in tiles.mitems: t.sType = 1000506001
  if getTiles(device, addr count, addr tiles[0]) != 0: return
  for t in tiles:
    # FP16 input, FP32 accumulate/output, subgroup scope, no saturation.
    # A single subgroup owns one output tile; cap scratch at 32 KiB.
    if t.aType == 0 and t.bType == 0 and t.cType == 1 and t.resultType == 1 and
       t.scope == 3 and t.saturating == 0 and t.m == 16 and t.n == 16 and t.k == 16:
      result.tileM = int(t.m); result.tileN = int(t.n); result.tileK = int(t.k)
      result.cooperativeMatrix = true
      break

proc deviceName*(caps: VulkanCaps): string =
  $cast[cstring](unsafeAddr caps.properties.deviceName[0])
