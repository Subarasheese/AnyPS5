#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeImages.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Native/include/CompileStatus.hpp"
#include "RuntimeAbi.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"

#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "BdaAbi.hpp"
#include <filesystem>
#include <fstream>
#include <spirv/unified1/spirv.hpp>
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineLibrary.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineEntryCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <set>
#include <optional>
#include <thread>
#include <condition_variable>
#include <deque>
#include <tuple>
#include <sys/ioctl.h>
#include <unistd.h>

namespace AgcDriver::Native {

namespace {

// Bumped by every GPU mapping change; SyncChangedMappings imports up to it.
std::atomic<std::uint64_t>& MappingGeneration() {
    static std::atomic<std::uint64_t> generation{0};
    return generation;
}

bool DrawBarriersRequested() {
    static const bool requested = std::getenv("APS5_NATIVE_DRAW_BARRIERS") != nullptr;
    return requested;
}

std::vector<VkDescriptorSetLayoutBinding> NativeGraphicsBindings(const Graphics::Context& context) {
    using Stage = ShaderRecompiler::RuntimeAbi::Stage;
    using Binding = ShaderRecompiler::RuntimeAbi::Binding;
    const std::array<std::pair<Stage, VkShaderStageFlags>, 4> stages{{
        {Stage::Main, VK_SHADER_STAGE_VERTEX_BIT | (context.meshShader ? VK_SHADER_STAGE_MESH_BIT_EXT : 0u)},
        {Stage::Fragment, VK_SHADER_STAGE_FRAGMENT_BIT},
        {Stage::TessellationControl, context.tessellationShader ? VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT : 0u},
        {Stage::TessellationEvaluation, context.tessellationShader ? VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT : 0u}
    }};
    constexpr std::array roles{Binding::Gds, Binding::BdaPagetable, Binding::FaultBuffer, Binding::FlattenedSrt, Binding::ShaderData};
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (const auto& [stage, flags] : stages) {
        if (flags == 0) continue;
        for (const auto role : roles) bindings.push_back({static_cast<std::uint32_t>(stage) * static_cast<std::uint32_t>(Binding::Count) + static_cast<std::uint32_t>(role), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags, nullptr});
    }
    return bindings;
}

// The device functions this file calls, resolved from the driver device (the driver's headers
// declare no Vulkan prototypes, so the usual names are free for them).
PFN_vkAllocateMemory vkAllocateMemory = nullptr;
PFN_vkCreateBuffer vkCreateBuffer = nullptr;
PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
PFN_vkQueueSubmit vkQueueSubmit = nullptr;
PFN_vkMapMemory vkMapMemory = nullptr;
PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT = nullptr;
PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = nullptr;
PFN_vkFreeMemory vkFreeMemory = nullptr;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
PFN_vkWaitForFences vkWaitForFences = nullptr;
PFN_vkResetFences vkResetFences = nullptr;
PFN_vkResetCommandBuffer vkResetCommandBuffer = nullptr;
PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR = nullptr;
PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
PFN_vkCreateFence vkCreateFence = nullptr;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = nullptr;
PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
PFN_vkCmdDispatch vkCmdDispatch = nullptr;
PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
PFN_vkMapMemory2KHR vkMapMemory2KHR = nullptr;
PFN_vkUnmapMemory2KHR vkUnmapMemory2KHR = nullptr;
PFN_vkUnmapMemory vkUnmapMemory = nullptr;

void LoadDeviceFunctions(const Graphics::Context& context) {
    vkAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(context.deviceProc(context.device, "vkAllocateMemory"));
    vkCreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(context.deviceProc(context.device, "vkCreateBuffer"));
    vkBindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(context.deviceProc(context.device, "vkBindBufferMemory"));
    vkGetBufferMemoryRequirements = reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(context.deviceProc(context.device, "vkGetBufferMemoryRequirements"));
    vkQueueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(context.deviceProc(context.device, "vkQueueSubmit"));
    vkMapMemory = reinterpret_cast<PFN_vkMapMemory>(context.deviceProc(context.device, "vkMapMemory"));
    vkGetMemoryHostPointerPropertiesEXT = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(context.deviceProc(context.device, "vkGetMemoryHostPointerPropertiesEXT"));
    vkGetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(context.deviceProc(context.device, "vkGetBufferDeviceAddress"));
    vkFreeMemory = reinterpret_cast<PFN_vkFreeMemory>(context.deviceProc(context.device, "vkFreeMemory"));
    vkCmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(context.deviceProc(context.device, "vkCmdPipelineBarrier"));
    vkWaitForFences = reinterpret_cast<PFN_vkWaitForFences>(context.deviceProc(context.device, "vkWaitForFences"));
    vkResetFences = reinterpret_cast<PFN_vkResetFences>(context.deviceProc(context.device, "vkResetFences"));
    vkResetCommandBuffer = reinterpret_cast<PFN_vkResetCommandBuffer>(context.deviceProc(context.device, "vkResetCommandBuffer"));
    vkGetMemoryFdPropertiesKHR = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(context.deviceProc(context.device, "vkGetMemoryFdPropertiesKHR"));
    vkUnmapMemory = reinterpret_cast<PFN_vkUnmapMemory>(context.deviceProc(context.device, "vkUnmapMemory"));
    if (context.placedMapAlignment != 0) {
        vkMapMemory2KHR = reinterpret_cast<PFN_vkMapMemory2KHR>(context.deviceProc(context.device, "vkMapMemory2KHR"));
        vkUnmapMemory2KHR = reinterpret_cast<PFN_vkUnmapMemory2KHR>(context.deviceProc(context.device, "vkUnmapMemory2KHR"));
    }
    vkEndCommandBuffer = reinterpret_cast<PFN_vkEndCommandBuffer>(context.deviceProc(context.device, "vkEndCommandBuffer"));
    vkDestroyShaderModule = reinterpret_cast<PFN_vkDestroyShaderModule>(context.deviceProc(context.device, "vkDestroyShaderModule"));
    vkDestroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(context.deviceProc(context.device, "vkDestroyBuffer"));
    vkCreateShaderModule = reinterpret_cast<PFN_vkCreateShaderModule>(context.deviceProc(context.device, "vkCreateShaderModule"));
    vkCreatePipelineLayout = reinterpret_cast<PFN_vkCreatePipelineLayout>(context.deviceProc(context.device, "vkCreatePipelineLayout"));
    vkCreateFence = reinterpret_cast<PFN_vkCreateFence>(context.deviceProc(context.device, "vkCreateFence"));
    vkCreateDescriptorSetLayout = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(context.deviceProc(context.device, "vkCreateDescriptorSetLayout"));
    vkCreateComputePipelines = reinterpret_cast<PFN_vkCreateComputePipelines>(context.deviceProc(context.device, "vkCreateComputePipelines"));
    vkCreateCommandPool = reinterpret_cast<PFN_vkCreateCommandPool>(context.deviceProc(context.device, "vkCreateCommandPool"));
    vkCmdPushDescriptorSetKHR = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(context.deviceProc(context.device, "vkCmdPushDescriptorSetKHR"));
    vkCmdPushConstants = reinterpret_cast<PFN_vkCmdPushConstants>(context.deviceProc(context.device, "vkCmdPushConstants"));
    vkCmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(context.deviceProc(context.device, "vkCmdDispatch"));
    vkCmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(context.deviceProc(context.device, "vkCmdBindPipeline"));
    vkBeginCommandBuffer = reinterpret_cast<PFN_vkBeginCommandBuffer>(context.deviceProc(context.device, "vkBeginCommandBuffer"));
    vkAllocateCommandBuffers = reinterpret_cast<PFN_vkAllocateCommandBuffers>(context.deviceProc(context.device, "vkAllocateCommandBuffers"));
    vkCmdBindDescriptorSets = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(context.deviceProc(context.device, "vkCmdBindDescriptorSets"));
}

struct Import {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    std::uint32_t permissions = 0;
};

std::atomic<NativeDevice*> device{nullptr};
std::mutex& CreateMutex() {
    static std::mutex mutex;
    return mutex;
}

}

struct NativeDevice::State {
    std::shared_ptr<VulkanDevice> driverDevice;
    Graphics::Context context{};

    VkDevice vk = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    VkDeviceSize importAlignment = 4096;

    mutable std::shared_mutex importsMutex;
    std::map<std::uint64_t, Import> imports;
    std::vector<Import> retired;
    std::size_t failedImports = 0;
    std::set<std::pair<std::uint64_t, std::uint64_t>> failed;
    // Chunks whose import failed whole: they are imported in SplitBytes pieces, so pages the driver
    // cannot pin cost only their own piece. Failed pieces are retried at each sync.
    static constexpr std::uint64_t SplitBytes = std::uint64_t{2} << 20u;
    std::set<std::uint64_t> split;
    std::uint64_t importedBytes = 0;

    // Page table: BdaAbi::Header + sorted BdaAbi::Range records, host visible.
    VkBuffer table = VK_NULL_HANDLE;
    VkDeviceMemory tableMemory = VK_NULL_HANDLE;
    void* tableMapping = nullptr;
    VkDeviceSize tableBytes = 0;
    VkDeviceAddress tableAddress = 0;
    std::uint64_t tableGeneration = 0;
    // The table's bound range must be exactly its header and records (the shader checks the count).
    VkDeviceSize tableUsedBytes = 0;

    // Native compute (v0: one batch at a time, submitted and waited for by Flush).
    struct ComputePipeline {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        bool pushConstants = false;
        bool heap = false;          // binds the image/sampler heap (NativeHeap::Set)
    };
    std::unique_ptr<NativeImages> images;
    VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout workgroupLayout = VK_NULL_HANDLE;
    struct WorkgroupScratch {
        Graphics::Context context;
        std::unique_ptr<Graphics::DeviceBuffer> buffer;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        ~WorkgroupScratch() {
            if (pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        }
    };
    std::shared_ptr<WorkgroupScratch> workgroupScratch;
    std::shared_ptr<WorkgroupScratch> workgroupFor(const std::array<std::uint32_t, 3>& groups, std::uint32_t dwords) {
        const auto stride = std::uint64_t{dwords} * sizeof(std::uint32_t);
        const auto limit = std::uint64_t{context.limits.maxStorageBufferRange};
        if (stride == 0 || stride > limit) throw std::runtime_error("native LDS stride exceeds storage-buffer range");
        std::uint64_t count = 1;
        for (const auto extent : groups) {
            if (extent != 0 && count > limit / stride / extent) throw std::runtime_error("native LDS dispatch exceeds storage-buffer range");
            count *= extent;
        }
        const auto bytes = std::max<std::uint64_t>(count, 1u) * stride;
        if (workgroupScratch && workgroupScratch->buffer->Size() >= bytes) return workgroupScratch;
        const auto size = std::min<std::uint64_t>((bytes + (1u << 20u) - 1u) & ~std::uint64_t{(1u << 20u) - 1u}, limit);
        auto made = std::make_shared<WorkgroupScratch>();
        made->context = context;
        made->buffer = std::make_unique<Graphics::DeviceBuffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        if (context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(vk, &poolInfo, nullptr, &made->pool) != VK_SUCCESS)
            throw std::runtime_error("native LDS descriptor pool");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = made->pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &workgroupLayout;
        if (context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(vk, &allocation, &made->set) != VK_SUCCESS)
            throw std::runtime_error("native LDS descriptor set");
        const VkDescriptorBufferInfo buffer{made->buffer->Handle(), 0, size};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = made->set;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer;
        context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(vk, 1, &write, 0, nullptr);
        if (Native::Logs()) std::fprintf(stderr, "[native] device-local LDS scratch: %llu bytes for %ux%ux%u groups (stride %u dwords)\n",
            static_cast<unsigned long long>(size), groups[0], groups[1], groups[2], dwords);
        workgroupScratch = made;
        return made;
    }
    VkDescriptorSetLayout graphicsLayout = VK_NULL_HANDLE;
    std::vector<std::uint32_t> graphicsLayoutKey;
    std::vector<std::shared_ptr<void>> kept;
    std::vector<std::function<void()>> afterBatch;
    // What the open batch records, for a batch that does not complete (the last 48 entries).
    std::vector<std::string> batchLog;
    // APS5_NATIVE_GPU_TIMES=1: GPU time per program (timestamps around each draw and dispatch).
    VkQueryPool timestamps = VK_NULL_HANDLE;
    std::vector<std::string> timedNames;
    std::map<std::string, std::pair<double, std::uint64_t>> gpuTimes;
    std::chrono::steady_clock::time_point timesReported = std::chrono::steady_clock::now();
    std::array<std::uint64_t, 6> barrierCounts{};
    std::uint64_t syncPackets = 0;
    std::uint64_t batchLogSize = 0;
    double lockWaitMs = 0.0;
    std::chrono::steady_clock::time_point lockReported = std::chrono::steady_clock::now();
    void noteLockWait(std::chrono::steady_clock::time_point start) {
        if (!Native::Logs()) return;
        const auto now = std::chrono::steady_clock::now();
        lockWaitMs += std::chrono::duration<double, std::milli>(now - start).count();
        if (now - lockReported < std::chrono::seconds(5)) return;
        std::fprintf(stderr, "[native] device lock waits: %.0f ms in %.1f s\n", lockWaitMs, std::chrono::duration<double>(now - lockReported).count());
        lockWaitMs = 0.0;
        lockReported = now;
    }
    bool passOpen = false;
    std::vector<std::uint64_t> passKey;
    std::int32_t passTimed = -1;
    std::set<std::uint64_t> writtenSinceBarrier;
    void endPass() {
        if (!passOpen) return;
        Graphics::EndRenderPass(context, commands);
        passOpen = false;
        passKey.clear();
        endTimed(passTimed);
        passTimed = -1;
    }
    static bool GpuTimes() {
        static const bool enabled = std::getenv("APS5_NATIVE_GPU_TIMES") != nullptr;
        return enabled;
    }
    std::int32_t beginTimed(const std::string& name) {
        if (!GpuTimes() || timedNames.size() >= 2048) return -1;
        if (timestamps == VK_NULL_HANDLE) {
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = 4096;
            context.Function<PFN_vkCreateQueryPool>("vkCreateQueryPool")(vk, &info, nullptr, &timestamps);
        }
        const auto index = static_cast<std::uint32_t>(timedNames.size());
        timedNames.push_back(name);
        context.Function<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool")(commands, timestamps, index * 2, 2);
        context.Function<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, index * 2);   // after earlier work completed: the delta is this work alone
        return static_cast<std::int32_t>(index);
    }
    void endTimed(std::int32_t index) {
        if (index < 0) return;
        context.Function<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps, static_cast<std::uint32_t>(index) * 2 + 1);
    }
    void collectTimes() {
        if (timedNames.empty()) return;
        std::vector<std::uint64_t> values(timedNames.size() * 2);
        context.Function<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(vk, timestamps, 0, static_cast<std::uint32_t>(values.size()), values.size() * 8, values.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        const double period = context.limits.timestampPeriod;
        for (std::size_t i = 0; i < timedNames.size(); ++i) {
            auto& entry = gpuTimes[timedNames[i]];
            entry.first += static_cast<double>(values[i * 2 + 1] - values[i * 2]) * period * 1e-6;
            ++entry.second;
        }
        timedNames.clear();
        const auto now = std::chrono::steady_clock::now();
        if (now - timesReported < std::chrono::seconds(5)) return;
        timesReported = now;
        std::vector<std::pair<double, std::string>> sorted;
        double total = 0.0;
        for (const auto& [name, value] : gpuTimes) sorted.push_back({value.first, name + " x" + std::to_string(value.second)}), total += value.first;
        std::sort(sorted.rbegin(), sorted.rend());
        std::fprintf(stderr, "[native] GPU time over 5 s: %.0f ms; draws %llu in %llu passes, barriers before draws: %llu after GPU writes, %llu target changes, %llu texture uploads, %llu sampled targets; top:\n", total, static_cast<unsigned long long>(barrierCounts[4]), static_cast<unsigned long long>(barrierCounts[5]), static_cast<unsigned long long>(barrierCounts[0]), static_cast<unsigned long long>(barrierCounts[1]), static_cast<unsigned long long>(barrierCounts[2]), static_cast<unsigned long long>(barrierCounts[3]));
        barrierCounts = {};
        for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 12); ++i) std::fprintf(stderr, "[native]   %8.1f ms  %s\n", sorted[i].first, sorted[i].second.c_str());
        gpuTimes.clear();
    }
    VkDeviceSize writeFlattenedSrt(std::span<const std::uint32_t> words, VkDeviceSize bytes) {
        const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16);
        const auto offset = (ringOffset + alignment - 1) & ~(alignment - 1);
        auto* target = static_cast<std::byte*>(ringMapping) + offset;
        std::memset(target, 0, static_cast<std::size_t>(bytes));
        std::memcpy(target, words.data(), words.size_bytes());
        ringOffset = offset + bytes;
        return offset;
    }
    // The image and sampler slots of a shader data block (IrBindingLayout::nativeImageSlots): the heap
    // indices of the T#s and S#s walked at the call site, unresolved while a resource is not created.
    void writeSlots(void* shaderData, std::size_t bytes, const ShaderRecompiler::RecompileResult& shader, std::span<const std::uint32_t> words) {
        const auto images = shader.nativeImageSources.size(), samplers = shader.nativeSamplerSources.size();
        const auto slots = 4u * (images + samplers + shader.nativeBaseKeys.size());
        const auto dwords = bytes / sizeof(std::uint32_t);
        if (images + samplers == 0 || dwords < slots + ShaderRecompiler::RuntimeAbi::NativeHeap::InvocationAddressDwords) return;
        auto* slot = static_cast<std::uint32_t*>(shaderData) + dwords - ShaderRecompiler::RuntimeAbi::NativeHeap::InvocationAddressDwords - slots;
        for (std::size_t index = 0; index < images; ++index, slot += 4) {
            std::uint32_t sampled = 0, storage = 0;
            const bool resolved = words.size() >= (index + 1) * 8 && this->images->ImageIndices(std::span<const std::uint32_t, 8>(words.data() + index * 8, 8), sampled, storage);
            slot[0] = sampled;
            slot[1] = storage;
            slot[2] = resolved ? 1u : 0u;
            slot[3] = 0;
        }
        for (std::size_t index = 0; index < samplers; ++index, slot += 4) {
            std::uint32_t sampler = 0;
            const auto first = images * 8 + index * 4;
            const bool resolved = words.size() >= first + 4 && this->images->SamplerIndex(std::span<const std::uint32_t, 4>(words.data() + first, 4), sampler);
            slot[0] = sampler;
            slot[1] = resolved ? 1u : 0u;
            slot[2] = slot[3] = 0;
        }
    }
    // Queues the slots' T#s and S#s that have no resource yet, for the Prepare before the work.
    void queueSlots(const ShaderRecompiler::RecompileResult& shader, std::span<const std::uint32_t> words) {
        std::uint32_t ignored[4];
        const auto images = shader.nativeImageSources.size();
        for (std::size_t index = 0; index < images && words.size() >= (index + 1) * 8; ++index) this->images->ImageIndices(std::span<const std::uint32_t, 8>(words.data() + index * 8, 8), ignored[0], ignored[1]);
        for (std::size_t index = 0; index < shader.nativeSamplerSources.size() && words.size() >= images * 8 + (index + 1) * 4; ++index) this->images->SamplerIndex(std::span<const std::uint32_t, 4>(words.data() + images * 8 + index * 4, 4), ignored[2]);
    }
    void writeInvocationAddresses(void* shaderData, std::size_t bytes, std::uint64_t program) const {
        constexpr auto tailBytes = ShaderRecompiler::RuntimeAbi::NativeHeap::InvocationAddressDwords * sizeof(std::uint32_t);
        if (bytes < tailBytes) throw std::runtime_error("native shader data lacks heap addresses");
        const auto image = images->ImageTable(), sampler = images->SamplerTable();
        const std::array<std::uint32_t, 6> words{
            static_cast<std::uint32_t>(image), static_cast<std::uint32_t>(image >> 32u),
            static_cast<std::uint32_t>(sampler), static_cast<std::uint32_t>(sampler >> 32u),
            static_cast<std::uint32_t>(program), static_cast<std::uint32_t>(program >> 32u)};
        std::memcpy(static_cast<std::byte*>(shaderData) + bytes - tailBytes, words.data(), tailBytes);
    }
    void note(std::string entry) {
        if (batchLog.size() >= 48) batchLog.erase(batchLog.begin());
        batchLog.push_back(std::move(entry));
    }
    // APS5_NATIVE_DRAW_STATS=<n>: pipeline statistics of the first n draws, printed per batch.
    VkQueryPool statistics = VK_NULL_HANDLE;
    std::uint32_t statisticsUsed = 0, statisticsLeft = 0;
    std::vector<std::string> statisticsNames;
    std::mutex workMutex;
    std::map<std::uint64_t, ComputePipeline> pipelines;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool recording = false;
    Import ring;                // host-visible shader data ring
    void* ringMapping = nullptr;
    VkDeviceSize ringBytes = 32ull << 20u;
    VkDeviceSize ringOffset = 0;
    std::unique_ptr<Graphics::DeviceBuffer> gds;
    bool gdsInitialized = false;
    void prepareGds() {
        if (!gds) gds = std::make_unique<Graphics::DeviceBuffer>(context, Pm4::GdsBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (gdsInitialized) return;
        endPass();
        context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, gds->Handle(), 0, Pm4::GdsBytes, 0);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        gdsInitialized = true;
    }
    Import fault;               // BdaAbi fault buffer
    Import zeroVertex;          // zeros for null vertex descriptors
    void* faultMapping = nullptr;

    Import hostBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, void*& mapping, bool gpuRead = false) {
        Import made;
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = bytes;
        bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        if (vkCreateBuffer(vk, &bufferInfo, nullptr, &made.buffer) != VK_SUCCESS) throw std::runtime_error("native device: buffer creation");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(vk, made.buffer, &requirements);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.pNext = &flags;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (gpuRead) made.memory = allocateVram(allocation, requirements.memoryTypeBits);
        if (made.memory == VK_NULL_HANDLE && (allocation.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk, &allocation, nullptr, &made.memory) != VK_SUCCESS)) throw std::runtime_error("native device: buffer memory");
        vkBindBufferMemory(vk, made.buffer, made.memory, 0);
        vkMapMemory(vk, made.memory, 0, VK_WHOLE_SIZE, 0, &mapping);
        made.begin = 0;
        made.end = bytes;
        return made;
    }

    void prepareCompute() {
        if (commandPool != VK_NULL_HANDLE) return;
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = context.queueFamilyIndex;
        if (vkCreateCommandPool(vk, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) throw std::runtime_error("native device: command pool");
        VkCommandBufferAllocateInfo bufferInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        bufferInfo.commandPool = commandPool;
        bufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        bufferInfo.commandBufferCount = 1;
        for (auto& slot : slots) {
            vkAllocateCommandBuffers(vk, &bufferInfo, &slot.commands);
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(vk, &fenceInfo, nullptr, &slot.fence);
            slot.ring = hostBuffer(ringBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, slot.ringMapping, true);
        }
        commands = slots[0].commands;
        ring = slots[0].ring;
        ringMapping = slots[0].ringMapping;
        context.detiler->SetBatchClock([this] { return openSerial; }, [this] { return completedSerial.load(std::memory_order_acquire); });
        completer = std::thread([this] { completeBatches(); });
        fault = hostBuffer(ShaderRecompiler::BdaAbi::NativeFaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, faultMapping);
        std::memset(faultMapping, 0, ShaderRecompiler::BdaAbi::NativeFaultBufferBytes);
    }

    // Cold pipelines build on workers. The originating work waits for readiness
    // with workMutex released, preserving work and allowing other queues to run.
    std::condition_variable pipelineReady;
    std::set<std::uint64_t> building;
    std::set<std::uint64_t> buildingGraphics;
    std::map<std::uint64_t, std::string> failedGraphics;
    std::map<std::uint64_t, std::string> failedPipelines;

    // The pipeline of a compiled compute shader: one push-descriptor set of its bindings. Under
    // workMutex; only an actual build failure returns null.
    ComputePipeline* pipelineFor(const ShaderRecompiler::RecompileResult& shader, std::uint64_t key, std::string& failure, std::unique_lock<std::mutex>& lock) {
        if (const auto found = pipelines.find(key); found != pipelines.end()) return &found->second;
        if (const auto failed = failedPipelines.find(key); failed != failedPipelines.end()) {
            failure = failed->second;
            return nullptr;
        }
        if (building.insert(key).second) {
            std::thread([this, &shader, key] {
                BackgroundSlot slot(true);
                std::string reason;
                std::optional<ComputePipeline> made;
                try { made = buildPipeline(shader, reason); }
                catch (const std::exception& error) { reason = error.what(); }
                std::lock_guard done(workMutex);
                building.erase(key);
                if (made) pipelines.emplace(key, *made);
                else failedPipelines.emplace(key, reason);
                pipelineReady.notify_all();
            }).detach();
        }
        {
            const CompileWait waiting;
            pipelineReady.wait(lock, [&] { return !building.contains(key); });
        }
        if (const auto found = pipelines.find(key); found != pipelines.end()) return &found->second;
        failure = failedPipelines.at(key);
        return nullptr;
    }

    std::optional<ComputePipeline> buildPipeline(const ShaderRecompiler::RecompileResult& shader, std::string& failure) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bool heap = false;
        for (const auto& binding : shader.bindings) {
            if (binding.descriptorSet == ShaderRecompiler::RuntimeAbi::NativeHeap::Set && (binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers)) {
                heap = true;
                continue;
            }
            switch (binding.role) {
                case ShaderRecompiler::DescriptorRole::Gds:
                case ShaderRecompiler::DescriptorRole::BdaPagetable:
                case ShaderRecompiler::DescriptorRole::FaultBuffer:
                case ShaderRecompiler::DescriptorRole::FlattenedSrt:
                case ShaderRecompiler::DescriptorRole::ShaderData:
                    bindings.push_back({binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
                    break;
                default:
                    failure = "binding role " + std::to_string(static_cast<int>(binding.role)) + " is not native yet";
                    return std::nullopt;
            }
        }
        ComputePipeline made;
        VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        setInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        setInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(vk, &setInfo, nullptr, &made.setLayout) != VK_SUCCESS) {
            failure = "vkCreateDescriptorSetLayout";
            return std::nullopt;
        }
        made.pushConstants = !shader.pushConstants.empty() || shader.pushDataStartDword != ~0u;
        made.heap = heap;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        static_assert(ShaderRecompiler::RuntimeAbi::NativeHeap::Set == 2);
        const std::array<VkDescriptorSetLayout, 3> setLayouts{made.setLayout, shader.workgroupMemoryDwords != 0 ? workgroupLayout : emptyLayout, images->HeapLayout()};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = heap ? 3u : shader.workgroupMemoryDwords != 0 ? 2u : 1u;
        layoutInfo.pSetLayouts = setLayouts.data();
        layoutInfo.pushConstantRangeCount = made.pushConstants ? 1u : 0u;
        layoutInfo.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(vk, &layoutInfo, nullptr, &made.layout) != VK_SUCCESS) {
            failure = "vkCreatePipelineLayout";
            return std::nullopt;
        }
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = shader.spirv.size() * sizeof(std::uint32_t);
        moduleInfo.pCode = shader.spirv.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(vk, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
            failure = "vkCreateShaderModule";
            return std::nullopt;
        }
        std::vector<VkSpecializationMapEntry> entries;
        for (std::size_t i = 0; i < shader.specialization.size(); ++i) entries.push_back({shader.specialization[i].id, static_cast<std::uint32_t>(i * sizeof(ShaderRecompiler::PipelineSpecializationConstant) + offsetof(ShaderRecompiler::PipelineSpecializationConstant, value)), sizeof(std::uint32_t)});
        VkSpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()), entries.data(), shader.specialization.size() * sizeof(ShaderRecompiler::PipelineSpecializationConstant), shader.specialization.data()};
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.stage.pSpecializationInfo = entries.empty() ? nullptr : &specialization;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        subgroupSize.requiredSubgroupSize = 32u;
        // The driver device enables subgroup size control where it can (VulkanDevice::ComputeTarget asks wave32 only then).
        if (shader.hostSubgroupSize == 32u) info.stage.pNext = &subgroupSize;
        if (shader.fullSubgroups) info.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.layout = made.layout;
        if (context.pipelineExecutableInfo) info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        const auto started = std::chrono::steady_clock::now();
        if (Native::Logs()) std::fprintf(stderr, "[native] building compute pipeline: %zu SPIR-V words, %zu bindings\n", shader.spirv.size(), shader.bindings.size());
        VkResult result;
        {
            auto cacheLock = context.LockPipelineCacheForUse();
            std::vector<std::byte> entryKey(std::as_bytes(std::span(shader.spirv)).begin(), std::as_bytes(std::span(shader.spirv)).end());
            const auto specialization = std::as_bytes(std::span(shader.specialization));
            entryKey.insert(entryKey.end(), specialization.begin(), specialization.end());
            for (const auto value : {shader.hostSubgroupSize, static_cast<std::uint32_t>(shader.fullSubgroups), static_cast<std::uint32_t>(shader.bindings.size()), shader.workgroupMemoryDwords}) {
                const auto bytes = std::as_bytes(std::span(&value, 1));
                entryKey.insert(entryKey.end(), bytes.begin(), bytes.end());
            }
            result = Graphics::CreateWithEntryCache(context, entryKey, [&](VkPipelineCache entry) { return vkCreateComputePipelines(vk, entry, 1, &info, nullptr, &made.pipeline); });
        }
        if (Native::Logs()) std::fprintf(stderr, "[native] compute pipeline built in %.0f ms (%d)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(), static_cast<int>(result));
        if (result == VK_SUCCESS && context.pipelineExecutableInfo) {
            std::fprintf(stderr, "[pipeline-stats] native compute pipeline, %zu SPIR-V words\n", shader.spirv.size());
            Graphics::LogPipelineStatistics_nid_no_patch(context, made.pipeline);
        }
        vkDestroyShaderModule(vk, module, nullptr);
        if (result != VK_SUCCESS) {
            failure = "vkCreateComputePipelines " + std::to_string(static_cast<int>(result));
            return std::nullopt;
        }
        return made;
    }

    struct Slot {
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        Import ring;
        void* ringMapping = nullptr;
        std::uint64_t serial = 0;
        bool submitted = false;
        bool done = false;
        bool used = false;
        VkResult result = VK_SUCCESS;
        std::vector<std::function<void()>> afterBatch;
        std::vector<std::shared_ptr<void>> kept;
        std::vector<std::shared_ptr<void>> resources;
        std::vector<Import> retired;
        std::vector<std::string> log;
        std::chrono::steady_clock::time_point submittedAt{};
    };
    static constexpr std::size_t SlotCount = 8;
    std::array<Slot, SlotCount> slots;
    std::size_t current = 0;
    std::uint64_t openSerial = 0;
    std::atomic<std::uint64_t> completedSerial{0};
    std::deque<std::size_t> submittedOrder;
    std::mutex completionMutex;
    std::condition_variable completionChanged;
    std::deque<std::size_t> completionQueue;
    bool stopping = false;
    std::thread completer;

    void completeBatches() {
        for (;;) {
            std::size_t index = 0;
            {
                std::unique_lock lock(completionMutex);
                completionChanged.wait(lock, [&] { return stopping || !completionQueue.empty(); });
                if (completionQueue.empty()) return;
                index = completionQueue.front();
            }
            auto& slot = slots[index];
            const auto result = vkWaitForFences(vk, 1, &slot.fence, VK_TRUE, 10'000'000'000ull);
            Native::TimelineEvent("gpu done", slot.serial);
            if (Native::Logs()) {
                static std::chrono::steady_clock::time_point lastDone{}, reported = std::chrono::steady_clock::now();
                static double busyMs = 0.0;
                const auto done = std::chrono::steady_clock::now();
                const auto start = std::max(slot.submittedAt, lastDone);
                busyMs += std::chrono::duration<double, std::milli>(done - start).count();
                lastDone = done;
                const auto window = std::chrono::duration<double, std::milli>(done - reported).count();
                if (window >= 5000.0) {
                    std::fprintf(stderr, "[native] GPU busy %.0f%% of %.1f s (submit to completion)\n", 100.0 * busyMs / window, window / 1000.0);
                    busyMs = 0.0;
                    reported = done;
                }
            }
            for (;;) {
                std::vector<std::function<void()>> actions;
                {
                    std::lock_guard lock(completionMutex);
                    if (result != VK_SUCCESS || slot.afterBatch.empty()) {
                        slot.result = result;
                        slot.done = true;
                        completionQueue.pop_front();
                        if (result == VK_SUCCESS) completedSerial.store(slot.serial, std::memory_order_release);
                        break;
                    }
                    actions = std::move(slot.afterBatch);
                    slot.afterBatch.clear();
                }
                for (auto& action : actions) action();
            }
            completionChanged.notify_all();
        }
    }

    bool reapOldest(bool block) {
        auto& s = *this;
        if (s.submittedOrder.empty()) return false;
        auto& slot = s.slots[s.submittedOrder.front()];
        {
            std::unique_lock lock(s.completionMutex);
            if (!slot.done && !block) return false;
            const auto waitStart = std::chrono::steady_clock::now();
            s.completionChanged.wait(lock, [&] { return slot.done; });
            static double waitedMs = 0.0;
            static std::uint64_t reaped = 0;
            static auto reported = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            waitedMs += std::chrono::duration<double, std::milli>(now - waitStart).count();
            ++reaped;
            if (now - reported > std::chrono::seconds(5)) {
                if (Native::Logs()) std::fprintf(stderr, "[native] batches: %llu in 5 s, executor waited %.0f ms for the GPU\n", static_cast<unsigned long long>(reaped), waitedMs);
                waitedMs = 0.0;
                reaped = 0;
                reported = now;
            }
        }
        for (const auto word : {0u, ShaderRecompiler::BdaAbi::NativeDiagnosticWord}) {
            auto* faultRecord = reinterpret_cast<ShaderRecompiler::BdaAbi::Fault*>(static_cast<std::uint32_t*>(s.faultMapping) + word);
            if (faultRecord->state == ShaderRecompiler::BdaAbi::FaultState::Ready) {
                static constexpr const char* Reasons[] = {"unknown", "read of unmapped memory", "access without permission", "address overflow", "invalid page table", "invalid rectangle", "shader loop limit reached", "unaligned access", "invalid descriptor", "image table fault", "unlinked call", "direct load mismatch"};
                const auto reason = static_cast<std::uint32_t>(faultRecord->reason);
                char detail[192];
                if (faultRecord->reason == ShaderRecompiler::BdaAbi::FaultReason::LoopLimit) std::snprintf(detail, sizeof(detail), "%s in program 0x%llx", Reasons[6], static_cast<unsigned long long>(faultRecord->address));
                else std::snprintf(detail, sizeof(detail), "%s (stage %u, pc 0x%x, address 0x%llx, %u bytes)", reason < std::size(Reasons) ? Reasons[reason] : Reasons[0], faultRecord->stage, faultRecord->instruction, static_cast<unsigned long long>(faultRecord->address), faultRecord->bytes);
                ReportGraphicsIssue("GPU fault", detail);
                if (slot.result == VK_SUCCESS) std::memset(faultRecord, 0, sizeof(*faultRecord));
            }
        }
        if (slot.result != VK_SUCCESS) {
            if (Native::Logs()) {
                std::fprintf(stderr, "[native] batch failed (%d); its last work:\n", static_cast<int>(slot.result));
                for (const auto& entry : slot.log) std::fprintf(stderr, "[native]   %s\n", entry.c_str());
            }
            throw std::runtime_error(slot.result == VK_ERROR_DEVICE_LOST ? std::string("the graphics device was lost (VK_ERROR_DEVICE_LOST)")
                : "GPU work did not complete (VkResult " + std::to_string(static_cast<int>(slot.result)) + ")");
        }
        s.images->Completed();
        s.collectTimes();
        if (s.statisticsUsed != 0) {
            std::vector<std::uint64_t> values(s.statisticsUsed * 6u);
            s.context.Function<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(s.vk, s.statistics, 0, s.statisticsUsed, values.size() * 8u, values.data(), 48, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            for (std::uint32_t q = 0; q < s.statisticsUsed; ++q) std::fprintf(stderr, "[native] draw stats %s: vertices %llu, VS %llu, clipping in %llu out %llu, FS %llu, TES %llu\n", s.statisticsNames[q].c_str(), static_cast<unsigned long long>(values[q * 6]), static_cast<unsigned long long>(values[q * 6 + 1]), static_cast<unsigned long long>(values[q * 6 + 2]), static_cast<unsigned long long>(values[q * 6 + 3]), static_cast<unsigned long long>(values[q * 6 + 4]), static_cast<unsigned long long>(values[q * 6 + 5]));
            s.statisticsUsed = 0;
            s.statisticsNames.clear();
        }
        slot.kept.clear();
        slot.resources.clear();
        {
            std::unique_lock imports(s.importsMutex);
            for (auto& import : slot.retired) s.destroy(import);
            slot.retired.clear();
        }
        slot.log.clear();
        slot.submitted = false;
        slot.done = false;
        s.submittedOrder.pop_front();
        return true;
    }

    void drain() {
        while (reapOldest(true)) {}
    }

    void flushLocked(bool wait = true) {
        auto& s = *this;
        if (!s.recording) {
            if (wait) s.drain();
            return;
        }
        s.endPass();
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(s.commands);
        s.recording = false;
        auto& slot = s.slots[s.current];
        slot.kept = std::move(s.kept);
        s.kept.clear();
        slot.resources = s.images->TakeBatchResources();
        s.batchLogSize = s.batchLog.size();
        slot.log = std::move(s.batchLog);
        s.batchLog.clear();
        {
            std::unique_lock imports(s.importsMutex);
            slot.retired = std::move(s.retired);
            s.retired.clear();
        }
        {
            std::lock_guard lock(s.completionMutex);
            slot.afterBatch = std::move(s.afterBatch);
            slot.done = false;
        }
        s.afterBatch.clear();
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &s.commands;
        {
            // The driver device's queue is shared with its presentation: submissions hold the GPU lock.
            const auto lockStart = std::chrono::steady_clock::now();
            std::lock_guard queue(GuestMemory::GpuMutex());
            const auto locked = std::chrono::steady_clock::now();
            if (vkQueueSubmit(s.context.queue, 1, &submit, slot.fence) != VK_SUCCESS) throw std::runtime_error("native device: vkQueueSubmit");
            if (Native::Logs()) {
                static double lockMs = 0.0, submitMs = 0.0;
                static auto reported = std::chrono::steady_clock::now();
                const auto now = std::chrono::steady_clock::now();
                lockMs += std::chrono::duration<double, std::milli>(locked - lockStart).count();
                submitMs += std::chrono::duration<double, std::milli>(now - locked).count();
                if (now - reported > std::chrono::seconds(5)) {
                    std::fprintf(stderr, "[native] batch submits over 5 s: %.0f ms waiting for the queue lock, %.0f ms in vkQueueSubmit\n", lockMs, submitMs);
                    lockMs = submitMs = 0.0;
                    reported = now;
                }
            }
        }
        slot.submitted = true;
        slot.used = true;
        slot.submittedAt = std::chrono::steady_clock::now();
        Native::TimelineEvent("gpu submit", slot.serial, s.batchLogSize);
        s.submittedOrder.push_back(s.current);
        {
            std::lock_guard lock(s.completionMutex);
            s.completionQueue.push_back(s.current);
        }
        s.completionChanged.notify_all();
        s.current = (s.current + 1) % SlotCount;
        if (wait || GpuTimes() || s.statisticsUsed != 0) s.drain();
    }

    bool graphicsBarrier = true;
    bool tracing = false;
    std::vector<std::string> frameTrace;
    std::vector<std::uint64_t> requestedDumps;
    std::uint64_t barrierRecordings = 0;
    std::vector<VkImageView> lastViews;
    VkExtent2D lastExtent{};
    std::map<std::pair<std::uint64_t, std::uint64_t>, bool> memoryWriters;

    bool writesGuestMemory(const ShaderRecompiler::RecompileResult& shader) {
        const auto identity = shader.Identity();
        const std::pair<std::uint64_t, std::uint64_t> key{identity.low, identity.high};
        if (const auto found = memoryWriters.find(key); found != memoryWriters.end() && !identity.Empty()) return found->second;
        const auto code = std::span(shader.spirv);
        std::set<std::uint32_t> pointers, bookkeeping;
        for (std::size_t i = 5; i < code.size();) {
            const auto count = code[i] >> 16u;
            if (count == 0 || i + count > code.size()) break;
            if ((code[i] & 0xffffu) == 5u && count >= 3) {
                const std::string_view name(reinterpret_cast<const char*>(&code[i + 2]), strnlen(reinterpret_cast<const char*>(&code[i + 2]), (count - 2) * 4u));
                if (name == "native_image_lookup" || name == "native_sampler_lookup" || name == "record_bda_fault") bookkeeping.insert(code[i + 1]);
            }
            i += count;
        }
        bool writes = false, skipping = false;
        for (std::size_t i = 5; i < code.size() && !writes;) {
            const auto count = code[i] >> 16u;
            const auto opcode = code[i] & 0xffffu;
            if (count == 0 || i + count > code.size()) break;
            if (opcode == 54u && count >= 5) skipping = bookkeeping.contains(code[i + 2]);
            else if (opcode == 56u) skipping = false;
            if (skipping) {
                i += count;
                continue;
            }
            if (opcode == 120u && count >= 4) pointers.insert(code[i + 2]);
            else if ((opcode == 65u || opcode == 66u || opcode == 67u || opcode == 70u || opcode == 124u || opcode == 83u) && count >= 4 && pointers.contains(code[i + 3])) pointers.insert(code[i + 2]);
            else if ((opcode == 62u || opcode == 228u) && count >= 3 && pointers.contains(code[i + 1])) writes = true;
            else if (opcode >= 229u && opcode <= 242u && count >= 4 && pointers.contains(code[i + 3])) writes = true;
            else if (opcode == 99u) writes = true;
            i += count;
        }
        if (!identity.Empty()) memoryWriters.emplace(key, writes);
        return writes;
    }

    // Work recorded into the open batch; the batch is submitted every RecordedPerBatch items so the
    // GPU runs the frame's earlier passes while later ones are recorded.
    std::uint32_t recordedInBatch = 0;
    bool frameScanPending = true;
    // Graphics pipelines whose build outlasted CompileWaitLimit: their draws are skipped until built.
    std::set<std::uint64_t> slowGraphics;
    void noteRecorded() {
        static const std::uint32_t perBatch = [] { const char* text = std::getenv("APS5_NATIVE_BATCH_WORK"); return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : 192u; }();
        if (perBatch != 0 && ++recordedInBatch >= perBatch) flushLocked(false);
    }

    void begin() {
        if (recording) return;
        recordedInBatch = 0;
        auto& slot = slots[current];
        while (slot.submitted) reapOldest(true);
        while (reapOldest(false)) {}
        if (slot.used) vkResetFences(vk, 1, &slot.fence);
        slot.serial = ++openSerial;
        commands = slot.commands;
        ring = slot.ring;
        ringMapping = slot.ringMapping;
        graphicsBarrier = true;
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(commands, 0);
        vkBeginCommandBuffer(commands, &beginInfo);
        recording = true;
        ringOffset = 0;
    }

    std::uint32_t memoryType(std::uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) != 0 && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        return UINT32_MAX;
    }

    VkDeviceMemory allocateVram(VkMemoryAllocateInfo allocation, std::uint32_t bits) const {
        allocation.memoryTypeIndex = memoryType(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (allocation.memoryTypeIndex == UINT32_MAX) return VK_NULL_HANDLE;
        VkDeviceMemory made = VK_NULL_HANDLE;
        if (vkAllocateMemory(vk, &allocation, nullptr, &made) == VK_SUCCESS) return made;
        for (std::uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
            if (memory.memoryTypes[type].propertyFlags != 0) continue;
            VkMemoryAllocateInfo small{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            small.allocationSize = 65536;
            small.memoryTypeIndex = type;
            VkDeviceMemory scratch = VK_NULL_HANDLE;
            if (vkAllocateMemory(vk, &small, nullptr, &scratch) == VK_SUCCESS) vkFreeMemory(vk, scratch, nullptr);
            if (vkAllocateMemory(vk, &allocation, nullptr, &made) == VK_SUCCESS) return made;
        }
        return VK_NULL_HANDLE;
    }

    // Diagnostic: whether the device can still allocate 1 MiB of host-visible memory.
    void probe(const char* where) {
        static int probes = 0;
        if (++probes > 12) return;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = 1u << 20u;
        allocation.memoryTypeIndex = memoryType(~0u, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory memory = VK_NULL_HANDLE;
        const auto result = vkAllocateMemory(vk, &allocation, nullptr, &memory);
        if (memory != VK_NULL_HANDLE) vkFreeMemory(vk, memory, nullptr);
        if (Native::Logs()) std::fprintf(stderr, "[native] allocation probe at %s: %d (thread %ld)\n", where, static_cast<int>(result), static_cast<long>(gettid()));
    }

    void destroy(Import& import) const {
        if (import.buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk, import.buffer, nullptr);
        if (import.memory != VK_NULL_HANDLE) vkFreeMemory(vk, import.memory, nullptr);
        import = {};
    }

    // Guest pools in video memory (GuestArena::GpuPlacement), keyed by their begin in `imports`.
    // Placement and eviction run under the guest memory lock, so they hand their import changes to
    // SyncGpuMemory (which takes the imports lock first) through `placements`.
    std::set<std::uint64_t> placed;
    std::uint64_t placedBytes = 0;
    std::mutex placementMutex;
    std::map<std::uint64_t, Import> placedNow;
    std::vector<std::uint64_t> evictedNow;
    std::map<std::uint64_t, VkDeviceMemory> placedMemory;

    bool placeVideoMemory(std::uint64_t address, std::uint64_t bytes) {
        const auto alignment = context.placedMapAlignment;
        if (alignment == 0 || vkMapMemory2KHR == nullptr || address % alignment != 0 || bytes % alignment != 0) return false;
        Import made;
        made.begin = address;
        made.end = address + bytes;
        made.permissions = ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write;
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = bytes;
        bufferInfo.usage = ImportUsage;
        if (vkCreateBuffer(vk, &bufferInfo, nullptr, &made.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(vk, made.buffer, &requirements);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.pNext = &flags;
        allocation.allocationSize = bytes;
        if (requirements.size != bytes || (made.memory = allocateVram(allocation, requirements.memoryTypeBits)) == VK_NULL_HANDLE || vkBindBufferMemory(vk, made.buffer, made.memory, 0) != VK_SUCCESS) {
            destroy(made);
            return false;
        }
        // The pool's contents so far, then the memory mapped in place of the host pages.
        void* staging = nullptr;
        if (vkMapMemory(vk, made.memory, 0, VK_WHOLE_SIZE, 0, &staging) != VK_SUCCESS) {
            destroy(made);
            return false;
        }
        std::memcpy(staging, reinterpret_cast<const void*>(address), bytes);
        vkUnmapMemory(vk, made.memory);
        VkMemoryMapPlacedInfoEXT placement{VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT};
        placement.pPlacedAddress = reinterpret_cast<void*>(address);
        VkMemoryMapInfoKHR mapInfo{VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR, &placement};
        mapInfo.flags = VK_MEMORY_MAP_PLACED_BIT_EXT;
        mapInfo.memory = made.memory;
        mapInfo.size = VK_WHOLE_SIZE;
        void* mapped = nullptr;
        if (vkMapMemory2KHR(vk, &mapInfo, &mapped) != VK_SUCCESS || mapped != reinterpret_cast<void*>(address)) {
            destroy(made);
            throw std::runtime_error("native device: video memory could not be placed at a guest pool");
        }
        VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addressInfo.buffer = made.buffer;
        made.address = vkGetBufferDeviceAddress(vk, &addressInfo);
        std::lock_guard lock(placementMutex);
        placedMemory[address] = made.memory;
        placedNow[address] = made;
        return true;
    }

    void evictVideoMemory(std::uint64_t address, std::uint64_t) {
        std::lock_guard lock(placementMutex);
        const auto memory = placedMemory.find(address);
        if (memory == placedMemory.end()) return;
        VkMemoryUnmapInfoKHR unmapInfo{VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR};
        unmapInfo.flags = VK_MEMORY_UNMAP_RESERVE_BIT_EXT;
        unmapInfo.memory = memory->second;
        vkUnmapMemory2KHR(vk, &unmapInfo);
        placedMemory.erase(memory);
        evictedNow.push_back(address);
    }

    // Under the exclusive imports lock: the pools placed and evicted since the last call.
    bool takePlacements() {
        std::lock_guard lock(placementMutex);
        bool changed = false;
        for (const auto address : evictedNow) {
            if (const auto placedNowEntry = placedNow.find(address); placedNowEntry != placedNow.end()) {
                retired.push_back(placedNowEntry->second);
                placedNow.erase(placedNowEntry);
                continue;
            }
            const auto found = imports.find(address);
            if (found == imports.end() || !placed.contains(address)) continue;
            retired.push_back(found->second);
            importedBytes -= found->second.end - found->second.begin;
            placedBytes -= found->second.end - found->second.begin;
            placed.erase(address);
            imports.erase(found);
            changed = true;
        }
        evictedNow.clear();
        for (auto& [address, made] : placedNow) {
            imports[address] = made;
            placed.insert(address);
            importedBytes += made.end - made.begin;
            placedBytes += made.end - made.begin;
            changed = true;
            if (Native::Logs()) std::fprintf(stderr, "[native] GPU pool 0x%llx+0x%llx in video memory (%.0f MiB placed)\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(made.end - made.begin), placedBytes / 1048576.0);
        }
        placedNow.clear();
        return changed;
    }

    static constexpr VkBufferUsageFlags ImportUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // One VkBuffer over [begin, begin + bytes) of the game's pages: a host-pointer import of private
    // memory, else a dma-buf of the shared memory file backing it (udmabuf), as AnyPS5's importer.
    bool importChunk(std::uint64_t begin, std::uint64_t bytes, bool writable, Import& out) {
        Import made;
        made.begin = begin;
        made.end = begin + bytes;
        made.permissions = ShaderRecompiler::BdaAbi::Read | (writable ? ShaderRecompiler::BdaAbi::Write : 0u);
        const auto bind = [&](VkExternalMemoryHandleTypeFlagBits type, const void* importInfo, std::uint32_t typeBits) {
            VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            external.handleTypes = type;
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.pNext = &external;
            bufferInfo.size = bytes;
            bufferInfo.usage = ImportUsage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(vk, &bufferInfo, nullptr, &made.buffer) != VK_SUCCESS) { step = "vkCreateBuffer"; return false; }
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(vk, made.buffer, &requirements);
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.pNext = importInfo;
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.pNext = &flags;
            allocation.allocationSize = bytes;
            allocation.memoryTypeIndex = memoryType(typeBits & requirements.memoryTypeBits, 0);
            VkResult result = VK_SUCCESS;
            if (allocation.memoryTypeIndex == UINT32_MAX) step = "memory type";
            else if ((result = vkAllocateMemory(vk, &allocation, nullptr, &made.memory)) != VK_SUCCESS) step = "vkAllocateMemory";
            else if ((result = vkBindBufferMemory(vk, made.buffer, made.memory, 0)) != VK_SUCCESS) step = "vkBindBufferMemory";
            else return true;
            std::snprintf(stepText, sizeof(stepText), "%s (%d; requirement 0x%llx, types 0x%x/0x%x)", step, static_cast<int>(result), static_cast<unsigned long long>(requirements.size), typeBits, requirements.memoryTypeBits);
            step = stepText;
            destroy(made);
            return false;
        };
        bool bound = false;
        step = "vkGetMemoryFdPropertiesKHR";
        VkMemoryHostPointerPropertiesEXT pointerProperties{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        if (vkGetMemoryHostPointerPropertiesEXT(vk, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, reinterpret_cast<void*>(begin), &pointerProperties) == VK_SUCCESS) {
            VkImportMemoryHostPointerInfoEXT hostPointer{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
            hostPointer.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            hostPointer.pHostPointer = reinterpret_cast<void*>(begin);
            bound = bind(VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, &hostPointer, pointerProperties.memoryTypeBits);
            if (!bound) std::snprintf(hostText, sizeof(hostText), "host pointer: %s", step);
        } else {
            std::snprintf(hostText, sizeof(hostText), "host pointer: vkGetMemoryHostPointerPropertiesEXT");
        }
        int file = -1;
        std::uint64_t offset = 0;
        if (!bound && udmabuf >= 0 && vkGetMemoryFdPropertiesKHR != nullptr && GuestArena::GuestArenaSharedBacking_nid_postfix(static_cast<std::uintptr_t>(begin), static_cast<std::size_t>(bytes), &file, &offset)) {
            udmabuf_create request{};
            request.memfd = static_cast<std::uint32_t>(file);
            request.flags = UDMABUF_FLAGS_CLOEXEC;
            request.offset = offset;
            request.size = bytes;
            const int buffer = ioctl(udmabuf, UDMABUF_CREATE, &request);
            if (buffer < 0) std::snprintf(stepText, sizeof(stepText), "UDMABUF_CREATE: %s (offset 0x%llx)", std::strerror(errno), static_cast<unsigned long long>(offset)), step = stepText;
            close(file);
            if (buffer >= 0) {
                VkMemoryFdPropertiesKHR properties{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
                VkImportMemoryFdInfoKHR fdImport{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
                fdImport.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                fdImport.fd = buffer;
                // A successful vkAllocateMemory owns the descriptor.
                if (vkGetMemoryFdPropertiesKHR(vk, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, buffer, &properties) == VK_SUCCESS) bound = bind(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, &fdImport, properties.memoryTypeBits);
                if (!bound) close(buffer);
            }
        }
        if (!bound) {
            std::snprintf(failureText, sizeof(failureText), "%s; dma-buf: %s", hostText, file < 0 ? "no shared backing" : udmabuf < 0 ? "no /dev/udmabuf" : step);
            lastFailure = failureText;
            return false;
        }
        VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addressInfo.buffer = made.buffer;
        made.address = vkGetBufferDeviceAddress(vk, &addressInfo);
        out = made;
        return true;
    }

    // Chunks of at most 64 MiB (udmabuf's default size limit); each chunk is one page-table record.
    static constexpr std::uint64_t ChunkBytes = std::uint64_t{64} << 20u;
    int udmabuf = -1;
    const char* lastFailure = "";
    const char* step = "";
    char stepText[160] = "";
    char hostText[200] = "";
    char failureText[400] = "";

    // Rewrites the page table from the imports (under the exclusive lock), in the native renderer's
    // direct format (RuntimeAbi::NativePageTable): two dependent loads per guest address.
    void rebuildTable() {
        namespace Table = ShaderRecompiler::RuntimeAbi::NativePageTable;
        constexpr std::uint64_t pageBytes = std::uint64_t{1} << Table::PageShift;
        std::uint64_t highest = 0;
        for (const auto& [begin, import] : imports) highest = std::max(highest, import.end);
        const auto level1Count = static_cast<std::uint32_t>(highest == 0 ? 1 : ((highest - 1) >> Table::BlockShift) + 1);
        std::vector<std::uint32_t> level1(level1Count, 0u);
        std::uint32_t blocks = 0;
        for (const auto& [begin, import] : imports) {
            for (auto region = import.begin >> Table::BlockShift; region <= (import.end - 1) >> Table::BlockShift; ++region) {
                if (level1[region] == 0) level1[region] = ++blocks;
            }
        }
        const auto level2Word = Table::Level2Word(level1Count);
        const auto needed = (static_cast<std::size_t>(level2Word) + static_cast<std::size_t>(blocks) * Table::PagesPerBlock * 4u) * sizeof(std::uint32_t);
        // A new table per change: recorded work keeps the table (and the exact range) it was bound
        // with; the previous buffer is retired and freed once no batch can use it (Flush).
        {
            if (table != VK_NULL_HANDLE) retired.push_back({0, 0, table, tableMemory, 0, 0});
            tableBytes = std::max<VkDeviceSize>(needed, 4096);
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = tableBytes;
            bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            if (vkCreateBuffer(vk, &bufferInfo, nullptr, &table) != VK_SUCCESS) throw std::runtime_error("native device: page table buffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(vk, table, &requirements);
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.pNext = &flags;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            tableMemory = allocateVram(allocation, requirements.memoryTypeBits);
            if (tableMemory == VK_NULL_HANDLE && (allocation.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk, &allocation, nullptr, &tableMemory) != VK_SUCCESS)) throw std::runtime_error("native device: page table memory");
            vkBindBufferMemory(vk, table, tableMemory, 0);
            vkMapMemory(vk, tableMemory, 0, VK_WHOLE_SIZE, 0, &tableMapping);
            VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addressInfo.buffer = table;
            tableAddress = vkGetBufferDeviceAddress(vk, &addressInfo);
        }
        std::vector<std::uint32_t> built(needed / sizeof(std::uint32_t), 0u);
        auto* words = built.data();
        words[0] = Table::Magic;
        words[1] = level1Count;
        std::memcpy(words + Table::HeaderWords, level1.data(), level1.size() * sizeof(std::uint32_t));
        std::size_t shared = 0;
        for (const auto& [begin, import] : imports) {
            for (auto page = import.begin & ~(pageBytes - 1); page < import.end; page += pageBytes) {
                const auto block = level1[page >> Table::BlockShift];
                auto* entry = reinterpret_cast<Table::Entry*>(words + level2Word + (static_cast<std::size_t>(block - 1) * Table::PagesPerBlock + ((page >> Table::PageShift) & (Table::PagesPerBlock - 1))) * 4u);
                if (entry->deviceLow != 0 || entry->deviceHigh != 0) ++shared;
                const auto device = import.address + page - import.begin;   // wraps for a page starting before the import; only offsets >= validBegin are used
                entry->deviceLow = static_cast<std::uint32_t>(device);
                entry->deviceHigh = static_cast<std::uint32_t>(device >> 32u);
                entry->validBegin = static_cast<std::uint32_t>(import.begin > page ? import.begin - page : 0);
                entry->validBytes = static_cast<std::uint32_t>(std::min<std::uint64_t>(import.end - page, Table::BytesMask)) | (import.permissions << Table::PermissionShift);
            }
        }
        if (shared != 0) {
            static std::size_t reported = 0;
            if (reported++ < 4 && Native::Logs()) std::fprintf(stderr, "[native] page table: %zu 64 KiB pages shared by two imports (the later one maps them)\n", shared);
        }
        std::memcpy(tableMapping, built.data(), needed);
        tableUsedBytes = needed;
        ++tableGeneration;
    }
};

std::vector<std::uint64_t> AddressList(const char* name) {
    std::vector<std::uint64_t> addresses;
    if (const char* text = std::getenv(name)) {
        while (*text) {
            char* end = nullptr;
            const auto address = std::strtoull(text, &end, 0);
            if (end == text) break;
            addresses.push_back(address);
            text = *end == ',' ? end + 1 : end;
        }
    }
    return addresses;
}

template <typename StateType>
void DumpTargetSurfaces(StateType& s, const std::filesystem::path& root, const std::vector<NativeImages::DrawnTarget>& targets, const std::string& prefix);

// Debug aid: the per-image draw census APS5_DEBUG_TARGET_CENSUS fills, keyed by (target address,
// VkImage), holding the draw count and the first pixel program that drew into it.
// Debug aid: draws into the APS5_NATIVE_DUMP_ADDRESSES targets since the last flip, so
// APS5_DEBUG_DUMP_AT_COUNT can read a target back at chosen points inside one frame.
std::atomic<std::uint64_t>& FrameTargetDraws() {
    static std::atomic<std::uint64_t> draws{0};
    return draws;
}

// Set at a flip once APS5_NATIVE_DUMP_TARGETS_AT has passed, so a count-based dump always counts
// from the start of a frame.
std::atomic<bool>& FrameCountGate() {
    static std::atomic<bool> open{false};
    return open;
}

std::mutex& TargetCensusMutex() {
    static std::mutex mutex;
    return mutex;
}
std::map<std::pair<std::uint64_t, std::uintptr_t>, std::pair<std::uint64_t, std::uint64_t>>& TargetCensus() {
    static std::map<std::pair<std::uint64_t, std::uintptr_t>, std::pair<std::uint64_t, std::uint64_t>> census;
    return census;
}

NativeDevice& NativeDevice::Create(std::shared_ptr<VulkanDevice> driverDevice) {
    std::lock_guard lock(CreateMutex());
    if (auto* existing = device.load()) return *existing;
    auto* made = new NativeDevice(std::move(driverDevice));
    device.store(made);
    static const GuestArena::GpuPlacement placement{
        [](std::uintptr_t address, std::size_t bytes) {
            static const std::vector<std::uint64_t> pools = AddressList("APS5_VIDEO_MEMORY_POOLS");
            if (!pools.empty() && pools.front() > 1 && std::find(pools.begin(), pools.end(), address) == pools.end()) return false;
            auto* native = NativeDevice::Get();
            return native != nullptr && native->state->placeVideoMemory(address, bytes);
        },
        [] {
            if (auto* native = NativeDevice::Get()) {
                std::lock_guard lock(native->state->workMutex);
                native->state->flushLocked(true);
            }
        },
        [](std::uintptr_t address, std::size_t bytes) {
            if (auto* native = NativeDevice::Get()) native->state->evictVideoMemory(address, bytes);
        }};
    // APS5_VIDEO_MEMORY_POOLS=1: whole GPU pools in video memory. GT7 keeps CPU data in the same
    // pools, which its code then reads uncached, so the default keeps pools in host memory.
    if (made->state->context.placedMapAlignment != 0 && std::getenv("APS5_VIDEO_MEMORY_POOLS") != nullptr) GuestArena::GuestArenaSetGpuPlacement_nid_postfix(&placement);
    made->SyncGpuMemory();
    const auto stats = made->Stats();
    if (Native::Logs()) std::fprintf(stderr, "[native] device ready: %zu GPU ranges imported (%.1f MiB), %zu failed\n", stats.imports, stats.importedBytes / 1048576.0, stats.failedImports);
    return *made;
}

NativeDevice* NativeDevice::Get() {
    return device.load(std::memory_order_acquire);
}

NativeDevice::NativeDevice(std::shared_ptr<VulkanDevice> driverDevice) : state(std::make_unique<State>()) {
    if (driverDevice == nullptr) throw std::runtime_error("native device: no driver device");
    state->driverDevice = std::move(driverDevice);
    state->context = state->driverDevice->NativeContext();
    LoadDeviceFunctions(state->context);
    state->vk = state->context.device;
    state->physical = state->context.physical;
    state->memory = state->context.memory;
    if (state->context.hostImportAlignment != 0) state->importAlignment = state->context.hostImportAlignment;
    state->udmabuf = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (vkGetMemoryHostPointerPropertiesEXT == nullptr) throw std::runtime_error("native device: VK_EXT_external_memory_host is unavailable");
    if (vkCmdPushDescriptorSetKHR == nullptr) throw std::runtime_error("native device: VK_KHR_push_descriptor is unavailable");
    state->images = std::make_unique<NativeImages>(state->context, [this](std::uint64_t address, VkBuffer& buffer, VkDeviceSize& offset) -> std::uint64_t {
        std::shared_lock lock(state->importsMutex);
        auto it = state->imports.upper_bound(address);
        if (it == state->imports.begin()) return 0;
        --it;
        if (address >= it->second.end) return 0;
        buffer = it->second.buffer;
        offset = address - it->second.begin;
        return it->second.end - address;
    });
    state->images->BeforeRecording([this] { state->endPass(); });
    VkDescriptorSetLayoutCreateInfo emptyInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    if (vkCreateDescriptorSetLayout(state->vk, &emptyInfo, nullptr, &state->emptyLayout) != VK_SUCCESS) throw std::runtime_error("native device: empty set layout");
    const VkDescriptorSetLayoutBinding workgroupBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo workgroupInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    workgroupInfo.bindingCount = 1;
    workgroupInfo.pBindings = &workgroupBinding;
    if (vkCreateDescriptorSetLayout(state->vk, &workgroupInfo, nullptr, &state->workgroupLayout) != VK_SUCCESS)
        throw std::runtime_error("native LDS descriptor layout");
    const auto graphicsBindings = NativeGraphicsBindings(state->context);
    for (const auto& binding : graphicsBindings) state->graphicsLayoutKey.insert(state->graphicsLayoutKey.end(), {binding.binding, binding.stageFlags});
    state->graphicsLayoutKey.push_back(0x4e48u);
    VkDescriptorSetLayoutCreateInfo graphicsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    graphicsInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    graphicsInfo.bindingCount = static_cast<std::uint32_t>(graphicsBindings.size());
    graphicsInfo.pBindings = graphicsBindings.data();
    if (vkCreateDescriptorSetLayout(state->vk, &graphicsInfo, nullptr, &state->graphicsLayout) != VK_SUCCESS) throw std::runtime_error("native graphics descriptor layout");
    if (Native::Logs()) std::fprintf(stderr, "[native] native device on %s, host import alignment %llu\n", state->driverDevice->DeviceName().c_str(), static_cast<unsigned long long>(state->importAlignment));
}

NativeDevice::~NativeDevice() {
    if (state->completer.joinable()) {
        {
            std::lock_guard lock(state->completionMutex);
            state->stopping = true;
        }
        state->completionChanged.notify_all();
        state->completer.join();
    }
    for (auto& [begin, import] : state->imports) state->destroy(import);
    for (auto& import : state->retired) state->destroy(import);
}

void NativeDevice::SyncGpuMemory() {
    // The registry's GPU ranges now, as import chunks; the lease is held only for the comparison.
    struct Chunk {
        std::uint64_t begin, bytes;
        bool writable;
        // The registry range the chunk came from (diagnostics).
        std::uint64_t range = 0, rangeBytes = 0, allocation = 0;
    };
    std::vector<Chunk> wanted;
    {
        const auto lease = GuestAllocations::GuestAllocationsAcquireAll_nid_postfix();
        for (const auto& range : lease) {
            // GPU-only memory (no CPU access, e.g. GT7's tile-mapped textures) is imported too:
            // `readable` is the CPU's view; the host mapping of GPU memory is always readable.
            if (range == nullptr || !range->gpu || range->bytes == 0) continue;
            // Committed runs only: reserved pages of heaps the game commits on demand cannot be pinned.
            for (const auto& [runBegin, runEnd] : GuestMemory::CommittedRanges(range->address, range->bytes)) {
                const auto begin = runBegin & ~(state->importAlignment - 1);
                const auto end = runEnd & ~(state->importAlignment - 1);
                for (auto at = begin; at < end; at += State::ChunkBytes) wanted.push_back({at, std::min(State::ChunkBytes, end - at), range->writable, range->address, range->bytes, range->allocationAddress});
            }
        }
    }
    std::sort(wanted.begin(), wanted.end(), [](const Chunk& a, const Chunk& b) { return a.begin < b.begin; });
    std::unique_lock lock(state->importsMutex);
    if (!state->split.empty()) {
        std::vector<Chunk> pieces;
        for (const auto& chunk : wanted) {
            if (!state->split.contains(chunk.begin)) {
                pieces.push_back(chunk);
                continue;
            }
            for (auto at = chunk.begin; at < chunk.begin + chunk.bytes; at += State::SplitBytes) pieces.push_back({at, std::min(State::SplitBytes, chunk.begin + chunk.bytes - at), chunk.writable, chunk.range, chunk.rangeBytes, chunk.allocation});
        }
        wanted = std::move(pieces);
    }
    state->failed.clear();
    const bool placements = state->takePlacements();
    std::erase_if(wanted, [&](const Chunk& chunk) {
        auto it = state->imports.upper_bound(chunk.begin);
        if (it == state->imports.begin()) return false;
        --it;
        return state->placed.contains(it->first) && chunk.begin < it->second.end;
    });
    bool changed = placements;
    for (auto it = state->imports.begin(); it != state->imports.end();) {
        const auto found = std::lower_bound(wanted.begin(), wanted.end(), it->second.begin, [](const Chunk& chunk, std::uint64_t begin) { return chunk.begin < begin; });
        if ((found != wanted.end() && found->begin == it->second.begin && found->begin + found->bytes == it->second.end) || state->placed.contains(it->first)) {
            ++it;
            continue;
        }
        // Unmapped: recorded work may still read the import, so it is freed at the next idle point.
        state->importedBytes -= it->second.end - it->second.begin;
        state->retired.push_back(it->second);
        it = state->imports.erase(it);
        changed = true;
    }
    // Debug aid: APS5_DEBUG_IMPORT_AT=<address> reports whether the registry offers that address and
    // whether an import covers it after this sync, for memory a texture upload could not read.
    static const std::uint64_t watchImport = [] { const char* text = std::getenv("APS5_DEBUG_IMPORT_AT"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
    if (watchImport != 0) {
        static int reports = 0;
        const auto covering = std::find_if(wanted.begin(), wanted.end(), [&](const Chunk& chunk) { return watchImport >= chunk.begin && watchImport < chunk.begin + chunk.bytes; });
        if (reports < 12) {
            ++reports;
            if (covering == wanted.end()) {
                const auto after = std::find_if(wanted.begin(), wanted.end(), [&](const Chunk& chunk) { return chunk.begin > watchImport; });
                std::fprintf(stderr, "[native] import watch 0x%llx: the registry offers no chunk for it (%zu chunks; nearest above 0x%llx)\n", static_cast<unsigned long long>(watchImport), wanted.size(),
                    static_cast<unsigned long long>(after == wanted.end() ? 0ull : after->begin));
            } else {
                std::fprintf(stderr, "[native] import watch 0x%llx: registry chunk 0x%llx+0x%llx writable %d; already imported %d; failed %d\n", static_cast<unsigned long long>(watchImport),
                    static_cast<unsigned long long>(covering->begin), static_cast<unsigned long long>(covering->bytes), covering->writable ? 1 : 0,
                    state->imports.count(covering->begin) != 0 ? 1 : 0, state->failed.count({covering->begin, covering->bytes}) != 0 ? 1 : 0);
            }
        }
    }
    for (const auto& chunk : wanted) {
        if (const auto it = state->imports.find(chunk.begin); it != state->imports.end() && it->second.end == chunk.begin + chunk.bytes) continue;
        if (state->failed.count({chunk.begin, chunk.bytes}) != 0) continue;
        const auto import = [&](std::uint64_t begin, std::uint64_t bytes) {
            Import made;
            if (!state->importChunk(begin, bytes, chunk.writable, made)) return false;
            state->imports[begin] = made;
            state->importedBytes += bytes;
            changed = true;
            return true;
        };
        if (import(chunk.begin, chunk.bytes)) continue;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> unpinned{{chunk.begin, chunk.bytes}};
        if (chunk.bytes > State::SplitBytes) {
            state->split.insert(chunk.begin);
            unpinned.clear();
            for (auto at = chunk.begin; at < chunk.begin + chunk.bytes; at += State::SplitBytes) {
                const auto bytes = std::min(State::SplitBytes, chunk.begin + chunk.bytes - at);
                if (!import(at, bytes)) unpinned.push_back({at, bytes});
            }
        }
        for (const auto& piece : unpinned) {
            ++state->failedImports;
            state->failed.insert(piece);
            static std::set<std::uint64_t> reported;
            if (!reported.insert(piece.first).second) continue;
            state->probe("failed import");
            // The host mapping holding the piece: its kind decides which import path can pin it.
            std::string mapping = "not in /proc/self/maps";
            if (std::FILE* maps = std::fopen("/proc/self/maps", "r")) {
                char line[512];
                while (std::fgets(line, sizeof(line), maps) != nullptr) {
                    unsigned long long low = 0, high = 0;
                    if (std::sscanf(line, "%llx-%llx", &low, &high) == 2 && piece.first >= low && piece.first < high) {
                        mapping = line;
                        if (!mapping.empty() && mapping.back() == '\n') mapping.pop_back();
                        break;
                    }
                }
                std::fclose(maps);
            }
            std::fprintf(stderr, "[native] GPU memory 0x%llx+0x%llx could not be imported: %s; mapping %s; registry range 0x%llx+0x%llx (allocation 0x%llx)\n", static_cast<unsigned long long>(piece.first), static_cast<unsigned long long>(piece.second), state->lastFailure, mapping.c_str(), static_cast<unsigned long long>(chunk.range), static_cast<unsigned long long>(chunk.rangeBytes), static_cast<unsigned long long>(chunk.allocation));
        }
    }
    if (changed || state->table == VK_NULL_HANDLE) state->rebuildTable();
}

void NativeDevice::SyncChangedMappings() const {
    static std::mutex syncing;
    static std::uint64_t synced = 0;
    if (MappingGeneration().load(std::memory_order_acquire) == synced) return;
    std::lock_guard lock(syncing);
    const auto generation = MappingGeneration().load(std::memory_order_acquire);
    if (generation == synced) return;
    try {
        const_cast<NativeDevice*>(this)->SyncGpuMemory();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[native] GPU memory import failed: %s\n", error.what());
    }
    synced = generation;
}

std::uint64_t NativeDevice::DeviceAddress(std::uint64_t address, std::uint64_t bytes) const {
    SyncChangedMappings();
    std::shared_lock lock(state->importsMutex);
    auto it = state->imports.upper_bound(address);
    if (it == state->imports.begin()) return 0;
    --it;
    if (address + bytes > it->second.end) return 0;
    return it->second.address + (address - it->second.begin);
}

std::uint64_t NativeDevice::DeviceSpan(std::uint64_t address, std::uint64_t& bytes) const {
    bytes = 0;
    SyncChangedMappings();
    std::shared_lock lock(state->importsMutex);
    auto it = state->imports.upper_bound(address);
    if (it == state->imports.begin()) return 0;
    --it;
    if (address >= it->second.end || (it->second.permissions & ShaderRecompiler::BdaAbi::Read) == 0) return 0;
    bytes = it->second.end - address;
    return it->second.address + (address - it->second.begin);
}

NativeDevice::Statistics NativeDevice::Stats() const {
    std::shared_lock lock(state->importsMutex);
    return {state->imports.size(), state->importedBytes, state->failedImports, state->tableGeneration};
}

bool NativeDevice::TransferOnGpu(std::span<const std::uint32_t> packet) {
    SyncChangedMappings();
    const auto opcode = (packet[0] >> 8u) & 0xffu;
    const auto address = [](std::uint32_t low, std::uint32_t high) { return low | (std::uint64_t{high} << 32u); };
    std::uint64_t destination = 0, source = 0, bytes = 0;
    bool fill = false, update = false;
    std::vector<std::uint32_t> words;
    if (opcode == 0x37u) {
        if (packet.size() < 5 || ((packet[1] >> 8u) & 0xfu) == 0u) return false;
        destination = address(packet[2], packet[3]);
        if ((packet[1] & 0x10000u) != 0u) words.push_back(packet.back());
        else words.assign(packet.begin() + 4, packet.end());
        bytes = words.size() * 4u;
        update = true;
        if ((destination & 3u) != 0u || bytes > 65536u) return false;
    } else if (opcode == 0x50u) {
        if (packet.size() < 7) return false;
        const auto from = ((packet[1] >> 29u) & 3u) | ((packet[6] >> 24u) & 4u) | ((packet[6] >> 25u) & 8u);
        const auto to = ((packet[1] >> 20u) & 3u) | ((packet[6] >> 25u) & 4u) | ((packet[6] >> 26u) & 8u);
        if (from == 1u || to == 1u) return DmaGds(packet, false);
        bytes = packet[6] & 0x3ffffffu;
        if (bytes == 0u || to == 2u) return true;
        if (to != 0u && to != 3u) return false;
        destination = address(packet[4], packet[5]);
        if (from == 2u) {
            fill = true;
            if ((destination & 3u) != 0u || (bytes & 3u) != 0u) return false;
        } else if (from == 0u || from == 3u) {
            source = address(packet[2], packet[3]);
        } else {
            return false;
        }
    } else {
        return false;
    }
    auto& s = *state;
    std::lock_guard lock(s.workMutex);
    const auto locate = [&](std::uint64_t at, VkDeviceSize& offset) {
        std::shared_lock imports(s.importsMutex);
        auto found = s.imports.upper_bound(at);
        if (found == s.imports.begin() || at > std::prev(found)->second.end || bytes > std::prev(found)->second.end - at) return VkBuffer{VK_NULL_HANDLE};
        --found;
        offset = at - found->second.begin;
        return found->second.buffer;
    };
    VkDeviceSize destinationOffset = 0, sourceOffset = 0;
    const auto destinationBuffer = locate(destination, destinationOffset);
    if (destinationBuffer == VK_NULL_HANDLE) return false;
    VkBuffer sourceBuffer = VK_NULL_HANDLE;
    if (!fill && !update) {
        sourceBuffer = locate(source, sourceOffset);
        if (sourceBuffer == VK_NULL_HANDLE || (sourceBuffer == destinationBuffer && source < destination + bytes && destination < source + bytes)) return false;
    }
    s.prepareCompute();
    s.begin();
    s.endPass();
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    if (update) {
        s.context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer")(s.commands, destinationBuffer, destinationOffset, bytes, words.data());
    } else if (fill) {
        s.context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(s.commands, destinationBuffer, destinationOffset, bytes, packet[2]);
    } else {
        const VkBufferCopy copy{sourceOffset, destinationOffset, bytes};
        s.context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(s.commands, sourceBuffer, destinationBuffer, 1, &copy);
    }
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    s.graphicsBarrier = true;
    s.images->MarkGpuWritten(destination, destination + bytes);
    s.note(std::string(update ? "WRITE_DATA " : fill ? "DMA fill " : "DMA copy ") + std::to_string(bytes) + " bytes");
    return true;
}

bool NativeDevice::DmaGds(std::span<const std::uint32_t> packet, bool wait) {
    SyncChangedMappings();
    if (packet.size() < 7) return false;
    const auto source = ((packet[1] >> 29u) & 3u) | ((packet[6] >> 24u) & 4u) | ((packet[6] >> 25u) & 8u);
    const auto destination = ((packet[1] >> 20u) & 3u) | ((packet[6] >> 25u) & 4u) | ((packet[6] >> 26u) & 8u);
    if (source != 1u && destination != 1u) return false;
    const auto bytes = packet[6] & 0x3ffffffu;
    if (bytes == 0 || destination == 2u) return true;
    const auto address = [](std::uint32_t low, std::uint32_t high) { return low | (std::uint64_t{high} << 32u); };
    const auto src = address(packet[2], packet[3]), dst = address(packet[4], packet[5]);
    if ((bytes & 3u) != 0 || (source != 2u && (src & 3u) != 0) || (dst & 3u) != 0)
        throw std::runtime_error("native GDS DMA requires dword-aligned ranges");
    if ((source == 1u && (src > Pm4::GdsBytes || bytes > Pm4::GdsBytes - src)) ||
        (destination == 1u && (dst > Pm4::GdsBytes || bytes > Pm4::GdsBytes - dst)))
        throw std::runtime_error("native GDS DMA exceeds GDS range");
    auto& s = *state;
    std::lock_guard lock(s.workMutex);
    s.prepareCompute();
    s.begin();
    s.endPass();
    s.prepareGds();
    s.graphicsBarrier = true;
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    const auto bufferFor = [&](std::uint32_t selector, std::uint64_t at, VkDeviceSize& offset) {
        if (selector == 1u) { offset = at; return s.gds->Handle(); }
        if (selector != 0u && selector != 3u) throw std::runtime_error("unsupported native GDS DMA selector");
        std::shared_lock imports(s.importsMutex);
        auto found = s.imports.upper_bound(at);
        if (found == s.imports.begin() || at > std::prev(found)->second.end || bytes > std::prev(found)->second.end - at)
            return VkBuffer{VK_NULL_HANDLE};
        --found;
        offset = at - found->second.begin;
        return found->second.buffer;
    };
    VkDeviceSize destinationOffset = 0;
    const auto destinationBuffer = bufferFor(destination, dst, destinationOffset);
    if (destinationBuffer == VK_NULL_HANDLE) throw std::runtime_error("GDS DMA destination outside imported memory");
    if (source == 2u) {
        s.context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(s.commands, destinationBuffer, destinationOffset, bytes, packet[2]);
    } else {
        VkDeviceSize sourceOffset = 0;
        const auto sourceBuffer = bufferFor(source, src, sourceOffset);
        if (sourceBuffer == VK_NULL_HANDLE) {
            // Explicit upload of a CPU constant table (not a GPU allocation).
            // GDS is at most 64 KiB, within vkCmdUpdateBuffer's payload limit.
            std::vector<std::byte> data(bytes);
            GuestMemory::Read(src, data);
            s.context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer")(s.commands, destinationBuffer, destinationOffset, bytes, data.data());
        } else {
            if (sourceBuffer == destinationBuffer && src < dst + bytes && dst < src + bytes)
                throw std::runtime_error("overlapping native GDS DMA");
            const VkBufferCopy copy{sourceOffset, destinationOffset, bytes};
            s.context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(s.commands, sourceBuffer, destinationBuffer, 1, &copy);
        }
    }
    s.note("GDS DMA " + std::to_string(bytes) + " bytes");
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    if (wait) s.flushLocked();
    return true;
}

// APS5_NATIVE_LOG: where a recorded draw's time goes (lock, state and pipeline, images, commands).
struct DrawPhases {
    std::chrono::steady_clock::time_point start, last;
    explicit DrawPhases(std::chrono::steady_clock::time_point started) : start(started), last(std::chrono::steady_clock::now()) {
        Mark(0, last);
    }
    void Mark(std::size_t phase) {
        const auto now = std::chrono::steady_clock::now();
        Mark(phase, now);
    }
    void Mark(std::size_t phase, std::chrono::steady_clock::time_point now) {
        if (!Native::Logs()) return;
        static std::mutex mutex;
        static std::array<double, 4> us{};
        static std::uint64_t draws = 0;
        static auto reported = std::chrono::steady_clock::now();
        std::lock_guard lock(mutex);
        us[phase] += std::chrono::duration<double, std::micro>(phase == 0 ? last - start : now - last).count();
        last = now;
        if (phase != 3) return;
        ++draws;
        if (now - reported < std::chrono::seconds(5)) return;
        std::fprintf(stderr, "[native] recorded draws: %llu in 5 s, us per draw: lock %.1f, state and pipeline %.1f, images %.1f, commands %.1f\n", static_cast<unsigned long long>(draws), us[0] / draws, us[1] / draws, us[2] / draws, us[3] / draws);
        us = {};
        draws = 0;
        reported = now;
    }
};

namespace {
std::uint32_t TexelBytes(VkFormat format);
}

bool NativeDevice::Dispatch(const ComputeWork& work, std::string& failure) {
    auto& s = *state;
    SyncChangedMappings();
    const auto lockStart = std::chrono::steady_clock::now();
    std::unique_lock lock(s.workMutex);
    s.noteLockWait(lockStart);
    s.prepareCompute();
    const auto& shader = *work.shader;
    auto* pipeline = s.pipelineFor(shader, work.pipelineKey, failure, lock);
    if (pipeline == nullptr) return false;
    std::shared_ptr<State::WorkgroupScratch> scratch;
    if (shader.workgroupMemoryDwords != 0) {
        auto groups = work.groups;
        if (work.indirectArguments != 0) {
            // The host sizes backing storage; only these 12 bytes are read.
            // Retire native producers before observing the indirect arguments.
            if (s.recording) s.flushLocked();
            GuestMemory::Read(work.indirectArguments, std::as_writable_bytes(std::span(groups)), 4);
        }
        scratch = s.workgroupFor(groups, shader.workgroupMemoryDwords);
    }
    const auto dataBytes = work.shaderData.size_bytes();
    const bool pushed = shader.pushDataStartDword != ~0u;
    const bool flattened = std::any_of(shader.bindings.begin(), shader.bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; });
    const auto flatBytes = flattened ? std::max<VkDeviceSize>(work.flattenedSrt.size_bytes(), 16) : 0;
    VkDeviceSize dataOffset = 0, flatOffset = 0;
    if ((!pushed && dataBytes != 0) || flattened) {
        const auto alignment = std::max<VkDeviceSize>(s.context.limits.minStorageBufferOffsetAlignment, 16);
        if (s.recording && s.ringOffset + 2 * alignment + (pushed ? 0 : dataBytes) + flatBytes > s.ringBytes) s.flushLocked(false);
    }
    s.begin();
    s.endPass();
    {
        const auto uploads = State::GpuTimes() ? s.beginTimed("image creation and uploads") : -1;
        s.queueSlots(shader, work.slotWords);
        s.images->Prepare(s.commands);
        s.endTimed(uploads);
    }
    if (!pushed && dataBytes != 0) {
        const auto alignment = std::max<VkDeviceSize>(s.context.limits.minStorageBufferOffsetAlignment, 16);
        dataOffset = (s.ringOffset + alignment - 1) & ~(alignment - 1);
        std::memcpy(static_cast<std::byte*>(s.ringMapping) + dataOffset, work.shaderData.data(), dataBytes);
        s.writeInvocationAddresses(static_cast<std::byte*>(s.ringMapping) + dataOffset, dataBytes, work.programAddress);
        s.writeSlots(static_cast<std::byte*>(s.ringMapping) + dataOffset, dataBytes, shader, work.slotWords);
        s.ringOffset = dataOffset + dataBytes;
    }
    // Debug aid: APS5_DEBUG_PROGRAM_IMAGES=<program address> lists the image slots of that compute
    // program's first dispatches: T# base, format words, size, and the resolved heap indices.
    // Debug aid: APS5_DEBUG_DUMP_PROGRAM_IMAGES=<program,...> dumps every image a listed compute
    // program binds, once per program, at the first flip after APS5_NATIVE_DUMP_TARGETS_AT seconds.
    {
        static const std::vector<std::uint64_t> dumpPrograms = AddressList("APS5_DEBUG_DUMP_PROGRAM_IMAGES");
        static const double dumpAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto started = std::chrono::steady_clock::now();
        static std::set<std::uint64_t> dumpedPrograms;
        if (!dumpPrograms.empty() && std::find(dumpPrograms.begin(), dumpPrograms.end(), work.programAddress) != dumpPrograms.end() && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= dumpAfter && dumpedPrograms.insert(work.programAddress).second) {
            for (std::size_t index = 0; index < shader.nativeImageSources.size() && work.slotWords.size() >= (index + 1) * 8; ++index) {
                const auto* t = work.slotWords.data() + index * 8;
                const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
                if (base != 0) s.requestedDumps.push_back(base);
                std::fprintf(stderr, "[native] dump request: program 0x%llx image slot %zu base 0x%llx\n", static_cast<unsigned long long>(work.programAddress), index, static_cast<unsigned long long>(base));
            }
        }
    }
    // Debug aid: APS5_DEBUG_WATCH_IMAGE=<address,...> logs each compute dispatch (program, slot)
    // binding an image within 256 MB above a listed address (once per program and base), with the time since the first such dispatch.
    {
        static const std::vector<std::uint64_t> watched = AddressList("APS5_DEBUG_WATCH_IMAGE");
        static const auto started = std::chrono::steady_clock::now();
        static int logged = 0;
        for (std::size_t index = 0; (!watched.empty() || std::getenv("APS5_DEBUG_WATCH_RANGE") != nullptr) && logged < 400 && index < shader.nativeImageSources.size() && work.slotWords.size() >= (index + 1) * 8; ++index) {
            const auto* t = work.slotWords.data() + index * 8;
            const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
            // APS5_DEBUG_WATCH_IMAGE=<a>,<b>,...: exact bases; a pair written <low>-<high> watches a range.
            static const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = [] {
                std::vector<std::pair<std::uint64_t, std::uint64_t>> found;
                if (const char* text = std::getenv("APS5_DEBUG_WATCH_RANGE")) {
                    char* end = nullptr;
                    const auto low = std::strtoull(text, &end, 0);
                    if (*end == '-') found.emplace_back(low, std::strtoull(end + 1, nullptr, 0));
                }
                return found;
            }();
            const bool inRange = std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return base >= range.first && base < range.second; });
            if (!inRange && std::none_of(watched.begin(), watched.end(), [&](std::uint64_t first) { return base == first; })) continue;
            static std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
            if (!seen.insert({work.programAddress ^ (static_cast<std::uint64_t>(index) << 56u), base}).second && logged > 40) continue;
            ++logged;
            std::fprintf(stderr, "[native] watch 0x%llx: dispatch 0x%llx slot %zu groups %ux%ux%u at %.2f s\n", static_cast<unsigned long long>(base), static_cast<unsigned long long>(work.programAddress), index, work.groups[0], work.groups[1], work.groups[2], std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }
    }
    // Debug experiment: APS5_DEBUG_SKIP_DISPATCH_TO=<address,...> drops every dispatch that binds an
    // image at one of those addresses, to see which compute pass owns the content there. Not a fix.
    {
        static const std::vector<std::uint64_t> skipTo = AddressList("APS5_DEBUG_SKIP_DISPATCH_TO");
        for (std::size_t index = 0; !skipTo.empty() && index < shader.nativeImageSources.size() && work.slotWords.size() >= (index + 1) * 8; ++index) {
            const auto* t = work.slotWords.data() + index * 8;
            const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
            if (std::none_of(skipTo.begin(), skipTo.end(), [&](std::uint64_t first) { return base == first; })) continue;
            static std::set<std::uint64_t> skipSeen;
            if (skipSeen.insert(work.programAddress).second)
                std::fprintf(stderr, "[native] skipping dispatches of cs 0x%llx (binds 0x%llx at slot %zu)\n", static_cast<unsigned long long>(work.programAddress), static_cast<unsigned long long>(base), index);
            return true;
        }
    }
    static const std::uint64_t debugProgram = [] { const char* text = std::getenv("APS5_DEBUG_PROGRAM_IMAGES"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
    static int debugDispatches = 0;
    if (debugProgram != 0 && work.programAddress == debugProgram && debugDispatches++ < 3) {
        for (std::size_t index = 0; index < shader.nativeImageSources.size() && work.slotWords.size() >= (index + 1) * 8; ++index) {
            const auto* t = work.slotWords.data() + index * 8;
            std::uint32_t sampled = 0, storage = 0;
            const bool resolved = s.images->ImageIndices(std::span<const std::uint32_t, 8>(t, 8), sampled, storage);
            const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
            std::fprintf(stderr, "[native] program 0x%llx image slot %zu: base 0x%llx T# %08x %08x %08x %08x %08x %08x %08x %08x -> sampled %u storage %u resolved %d\n", static_cast<unsigned long long>(work.programAddress), index, static_cast<unsigned long long>(base), t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], sampled, storage, resolved);
        }
    }
    if (flattened) flatOffset = s.writeFlattenedSrt(work.flattenedSrt, flatBytes);
    std::vector<VkDescriptorBufferInfo> infos;
    std::vector<VkWriteDescriptorSet> writes;
    infos.reserve(shader.bindings.size());
    for (const auto& binding : shader.bindings) {
        VkDescriptorBufferInfo info{};
        switch (binding.role) {
            case ShaderRecompiler::DescriptorRole::BdaPagetable: {
                std::shared_lock imports(s.importsMutex);
                info = {s.table, 0, s.tableUsedBytes};
                break;
            }
            case ShaderRecompiler::DescriptorRole::FaultBuffer: info = {s.fault.buffer, 0, VK_WHOLE_SIZE}; break;
            case ShaderRecompiler::DescriptorRole::ShaderData: info = {s.ring.buffer, dataOffset, std::max<VkDeviceSize>(dataBytes, 16)}; break;
            case ShaderRecompiler::DescriptorRole::FlattenedSrt: info = {s.ring.buffer, flatOffset, flatBytes}; break;
            case ShaderRecompiler::DescriptorRole::Gds: s.prepareGds(); info = {s.gds->Handle(), 0, Pm4::GdsBytes}; break;
            default: continue;
        }
        infos.push_back(info);
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstBinding = binding.binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &infos.back();
        writes.push_back(write);
    }
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(s.commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    if (scratch) {
        vkCmdBindDescriptorSets(s.commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout,
            ShaderRecompiler::WorkgroupMemoryDescriptorSet, 1, &scratch->set, 0, nullptr);
        s.kept.push_back(scratch);
    }
    if (pipeline->heap) {
        const auto heap = s.images->HeapSet();
        vkCmdBindDescriptorSets(s.commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, ShaderRecompiler::RuntimeAbi::NativeHeap::Set, 1, &heap, 0, nullptr);
    }
    if (!writes.empty()) vkCmdPushDescriptorSetKHR(s.commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    if (pipeline->pushConstants) {
        std::array<std::byte, 128> push{};
        std::memcpy(push.data(), shader.pushConstants.data(), std::min<std::size_t>(shader.pushConstants.size(), push.size()));
        if (pushed) {
            const auto start = static_cast<std::size_t>(shader.pushDataStartDword) * 4u;
            if (start > push.size() || dataBytes > push.size() - start) throw std::runtime_error("native compute shader data exceeds push block");
            std::memcpy(push.data() + start, work.shaderData.data(), dataBytes);
            s.writeInvocationAddresses(push.data() + start, dataBytes, work.programAddress);
            s.writeSlots(push.data() + start, dataBytes, shader, work.slotWords);
        }
        vkCmdPushConstants(s.commands, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, push.data());
    }
    if (Native::Logs()) {
        char entry[96];
        std::snprintf(entry, sizeof(entry), "dispatch 0x%llx %ux%ux%u", static_cast<unsigned long long>(work.programAddress), work.groups[0], work.groups[1], work.groups[2]);
        s.note(entry);
    }
    char timedName[64] = "";
    if (State::GpuTimes() || s.tracing) std::snprintf(timedName, sizeof(timedName), "dispatch 0x%llx", static_cast<unsigned long long>(work.programAddress));
    const auto timed = State::GpuTimes() ? s.beginTimed(timedName) : -1;
    if (s.tracing) {
        char line[256];
        std::snprintf(line, sizeof(line), "%s groups %ux%ux%u%s", timedName, work.groups[0], work.groups[1], work.groups[2], work.indirectArguments != 0 ? " indirect" : "");
        s.frameTrace.emplace_back(line);
    }
    if (work.indirectArguments != 0) {
        std::shared_lock imports(s.importsMutex);
        auto it = s.imports.upper_bound(work.indirectArguments);
        if (it == s.imports.begin() || work.indirectArguments + 12 > std::prev(it)->second.end) {
            failure = "indirect dispatch arguments outside imported memory";
            s.endTimed(timed);
            return false;
        }
        --it;
        VkMemoryBarrier indirect{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        indirect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
        indirect.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &indirect, 0, nullptr, 0, nullptr);
        s.context.Function<PFN_vkCmdDispatchIndirect>("vkCmdDispatchIndirect")(s.commands, it->second.buffer, work.indirectArguments - it->second.begin);
    } else {
        vkCmdDispatch(s.commands, work.groups[0], work.groups[1], work.groups[2]);
    }
    s.graphicsBarrier = true;
    s.endTimed(timed);
    s.noteRecorded();
    // Debug aid: APS5_DEBUG_AFTER_DISPATCH=<program>:<image slot> reads that slot's surfaces back
    // right after the program's first dispatches from APS5_NATIVE_DUMP_TARGETS_AT seconds and prints
    // how many texels of level 0, layer 0 are nonzero (written by the dispatch, or not).
    {
        static const std::array<std::uint64_t, 2> after = [] {
            std::array<std::uint64_t, 2> values{};
            if (const char* text = std::getenv("APS5_DEBUG_AFTER_DISPATCH")) {
                char* end = nullptr;
                values[0] = std::strtoull(text, &end, 0);
                if (*end == ':') values[1] = std::strtoull(end + 1, nullptr, 0);
            }
            return values;
        }();
        static const double dumpAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto started = std::chrono::steady_clock::now();
        static int checks = 0;
        if (after[0] != 0 && work.programAddress == after[0] && checks < 4 && work.slotWords.size() >= (after[1] + 1) * 8 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= dumpAfter) {
            ++checks;
            const auto* t = work.slotWords.data() + after[1] * 8;
            const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
            for (const auto& target : s.images->SurfacesAt(base)) {
                const auto texel = TexelBytes(target.format);
                if (texel == 0) continue;
                void* mapping = nullptr;
                const auto bytes = static_cast<VkDeviceSize>(target.width) * target.height * texel;
                auto buffer = s.hostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, mapping);
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                VkBufferImageCopy region{};
                region.imageSubresource = {static_cast<VkImageAspectFlags>(target.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
                region.imageExtent = {target.width, target.height, 1};
                s.context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(s.commands, target.image, VK_IMAGE_LAYOUT_GENERAL, buffer.buffer, 1, &region);
                s.flushLocked();
                const auto* words = static_cast<const std::uint32_t*>(mapping);
                std::size_t nonzero = 0;
                for (VkDeviceSize i = 0; i < bytes / 4; ++i) nonzero += words[i] != 0;
                std::fprintf(stderr, "[native] after dispatch 0x%llx (groups %ux%ux%u): slot %llu surface 0x%llx %ux%u VkFormat %d: %zu of %llu dwords nonzero, first %08x\n", static_cast<unsigned long long>(work.programAddress), work.groups[0], work.groups[1], work.groups[2], static_cast<unsigned long long>(after[1]), static_cast<unsigned long long>(base), target.width, target.height, static_cast<int>(target.format), nonzero, static_cast<unsigned long long>(bytes / 4), words[0]);
                s.context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(s.vk, buffer.buffer, nullptr);
                s.context.Function<PFN_vkFreeMemory>("vkFreeMemory")(s.vk, buffer.memory, nullptr);
            }
        }
    }
    return true;
}

bool NativeDevice::Draw(const DrawWork& work, std::string& failure) {
    auto& s = *state;
    SyncChangedMappings();
    const auto lockStart = std::chrono::steady_clock::now();
    std::unique_lock lock(s.workMutex);
    s.noteLockWait(lockStart);
    DrawPhases phases(lockStart);
    s.prepareCompute();
    const auto& graphics = *work.state;
    if (graphics.colors.empty() && !graphics.depth) {
        failure = "draw without color or depth targets";
        return false;
    }
    // The stages' buffer bindings, one push-descriptor set (every stage's bindings have their own
    // numbers, RuntimeAbi::BindingNumber), and the shader data blocks that do not fit push constants.
    const auto& layoutKey = s.graphicsLayoutKey;
    const auto setLayout = s.graphicsLayout;
    bool heap = false;
    std::size_t ringBytes = 0;
    std::size_t bufferBindingCount = 0;
    for (const auto& stage : work.stages) {
        if (stage.shader->pushDataStartDword == ~0u) ringBytes += (stage.shaderData.size() * 4u + 255u) & ~std::size_t{255};
        ringBytes += (std::max<std::size_t>(stage.flattenedSrt.size() * 4u, 16) + 255u) & ~std::size_t{255};
        for (const auto& binding : stage.shader->bindings) {
            if (binding.descriptorSet == ShaderRecompiler::RuntimeAbi::NativeHeap::Set && (binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers)) {
                heap = true;
                continue;
            }
            if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer && binding.role != ShaderRecompiler::DescriptorRole::ShaderData && binding.role != ShaderRecompiler::DescriptorRole::Gds && binding.role != ShaderRecompiler::DescriptorRole::FlattenedSrt) {
                failure = "graphics binding role " + std::to_string(static_cast<int>(binding.role)) + " is not native yet";
                return false;
            }
            if (binding.descriptorSet != 0) {
                failure = "graphics stage buffers outside set 0";
                return false;
            }
            ++bufferBindingCount;
        }
    }
    std::vector<Graphics::CompiledShader> compiled;
    for (const auto& stage : work.stages) compiled.push_back({stage.stage, stage.shader, stage.pushOffset});
    Graphics::PipelineSetLayouts layouts{setLayout, VK_NULL_HANDLE, false, layoutKey, {}, s.images->HeapLayout(), s.emptyLayout};
    std::shared_ptr<Graphics::Pipeline> pipeline;
    // Vertex buffers: one binding per attribute, in the game's imported memory (a null descriptor
    // reads zeros).
    std::vector<ShaderRecompiler::VertexAttribute> attributes;
    for (const auto& stage : work.stages) attributes.insert(attributes.end(), stage.attributes.begin(), stage.attributes.end());
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceSize> vertexOffsets;
    Graphics::VertexInputLayout vertexInput;
    try {
        vertexInput = Graphics::BuildVertexInputLayout(s.context, attributes);
        if (s.zeroVertex.buffer == VK_NULL_HANDLE) {
            void* mapping = nullptr;
            s.zeroVertex = s.hostBuffer(4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mapping);
            std::memset(mapping, 0, 4096);
        }
        std::shared_lock imports(s.importsMutex);
        for (const auto& attribute : attributes) {
            const auto& fields = attribute.resource.fields;
            const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
            NotePoolUse("vertex", address);
            auto it = s.imports.upper_bound(address);
            static int reported = 0;
            if (reported < 24) {
                ++reported;
                if (Native::Logs()) std::fprintf(stderr, "[native] vertex attribute %u: V# %08x %08x %08x %08x address 0x%llx %s; viewport %.0f,%.0f %.0fx%.0f scissor %d,%d %ux%u\n", attribute.location, fields[0], fields[1], fields[2], fields[3], static_cast<unsigned long long>(address), Graphics::NullVertexDescriptor(attribute) ? "null" : (it == s.imports.begin() || address >= std::prev(it)->second.end) ? "NOT IMPORTED" : "imported", graphics.viewport.x, graphics.viewport.y, graphics.viewport.width, graphics.viewport.height, graphics.scissor.offset.x, graphics.scissor.offset.y, graphics.scissor.extent.width, graphics.scissor.extent.height);
            }
            if (Graphics::NullVertexDescriptor(attribute) || address == 0 || it == s.imports.begin() || address >= std::prev(it)->second.end) {
                vertexBuffers.push_back(s.zeroVertex.buffer);
                vertexOffsets.push_back(0);
                continue;
            }
            --it;
            vertexBuffers.push_back(it->second.buffer);
            vertexOffsets.push_back(address - it->second.begin);
        }
    } catch (const std::exception& error) {
        failure = std::string("vertex input: ") + error.what();
        return false;
    }
    try {
        std::uint64_t hash = 0;
        pipeline = Graphics::FindCachedPipeline(s.context, graphics, vertexInput, layouts, compiled, VK_IMAGE_LAYOUT_GENERAL, hash);
        if (pipeline == nullptr && hash == 0) pipeline = Graphics::CachedPipeline(s.context, graphics, vertexInput, layouts, compiled, VK_IMAGE_LAYOUT_GENERAL);
        if (pipeline == nullptr) {
            // Keep this draw pending until its pipeline is ready.
            if (const auto failed = s.failedGraphics.find(hash); failed != s.failedGraphics.end()) {
                failure = failed->second;
                return false;
            }
            if (s.buildingGraphics.insert(hash).second) {
                std::thread([&s, hash, context = s.context, state = graphics, vertexInput, layoutKey, setLayout, compiled, heapLayout = s.images->HeapLayout(), empty = s.emptyLayout] {
                    BackgroundSlot slot;
                    std::string reason;
                    try {
                        const Graphics::PipelineSetLayouts copied{setLayout, VK_NULL_HANDLE, false, layoutKey, {}, heapLayout, empty};
                        const auto started = std::chrono::steady_clock::now();
                        const auto before = Graphics::PipelineLibraryCountersOf(context.device);
                        static_cast<void>(Graphics::BuildCachedPipeline(context, state, vertexInput, copied, compiled, VK_IMAGE_LAYOUT_GENERAL));
                        if (Native::Logs()) {
                            const auto after = Graphics::PipelineLibraryCountersOf(context.device);
                            std::fprintf(stderr, "[native] graphics pipeline %016llx built in %.0f ms (libraries: vertex input %llu, pre-rasterization %llu, fragment shader %llu, fragment output %llu)\n",
                                static_cast<unsigned long long>(hash), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
                                static_cast<unsigned long long>(after.built[0] - before.built[0]), static_cast<unsigned long long>(after.built[1] - before.built[1]),
                                static_cast<unsigned long long>(after.built[2] - before.built[2]), static_cast<unsigned long long>(after.built[3] - before.built[3]));
                        }
                    } catch (const std::exception& error) {
                        reason = std::string("graphics pipeline: ") + error.what();
                    }
                    std::lock_guard lock(s.workMutex);
                    s.buildingGraphics.erase(hash);
                    if (!reason.empty()) s.failedGraphics.emplace(hash, reason);
                    s.pipelineReady.notify_all();
                }).detach();
            }
            {
                const CompileWait waiting;
                const auto built = [&] { return !s.buildingGraphics.contains(hash); };
                if (CompileWaitLimit().count() == 0) s.pipelineReady.wait(lock, built);
                else if (s.slowGraphics.contains(hash) ? !built() : !s.pipelineReady.wait_for(lock, CompileWaitNow(), built)) {
                    s.slowGraphics.insert(hash);
                    NoteCompileWaitTimedOut();
                    failure = "graphics pipeline still compiling";
                    return false;
                }
            }
            if (const auto failed = s.failedGraphics.find(hash); failed != s.failedGraphics.end()) {
                failure = failed->second;
                return false;
            }
            pipeline = Graphics::FindCachedPipeline(s.context, graphics, vertexInput, layouts, compiled, VK_IMAGE_LAYOUT_GENERAL, hash);
            if (pipeline == nullptr) pipeline = Graphics::BuildCachedPipeline(s.context, graphics, vertexInput, layouts, compiled, VK_IMAGE_LAYOUT_GENERAL);
        }
    } catch (const std::exception& error) {
        failure = std::string("graphics pipeline: ") + error.what();
        return false;
    }
    phases.Mark(1);
    if (s.recording && ringBytes != 0 && s.ringOffset + ringBytes + 256 > s.ringBytes) s.flushLocked(false);
    s.begin();
    for (const auto& stage : work.stages) s.queueSlots(*stage.shader, stage.slotWords);
    // Debug aid: APS5_DEBUG_DUMP_TARGET_IMAGES=<color target,...> logs and dumps (once per base) the
    // images that draws into a listed color target bind, from APS5_NATIVE_DUMP_TARGETS_AT seconds.
    {
        static const std::vector<std::uint64_t> dumpTargets = AddressList("APS5_DEBUG_DUMP_TARGET_IMAGES");
        static const double dumpAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto started = std::chrono::steady_clock::now();
        static std::set<std::uint64_t> dumpedBases;
        if (!dumpTargets.empty() && !graphics.colors.empty() && std::find(dumpTargets.begin(), dumpTargets.end(), graphics.colors.front().address) != dumpTargets.end() && dumpedBases.size() < 300 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= dumpAfter) {
            for (const auto& stage : work.stages) {
                for (std::size_t index = 0; index < stage.shader->nativeImageSources.size() && stage.slotWords.size() >= (index + 1) * 8; ++index) {
                    const auto* t = stage.slotWords.data() + index * 8;
                    const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
                    if (base == 0 || !dumpedBases.insert(base).second) continue;
                    s.requestedDumps.push_back(base);
                    std::fprintf(stderr, "[native] draw image: target 0x%llx program 0x%llx (stage %d) slot %zu base 0x%llx T# %08x %08x %08x %08x %08x %08x %08x %08x\n", static_cast<unsigned long long>(graphics.colors.front().address), static_cast<unsigned long long>(stage.programAddress), static_cast<int>(stage.stage), index, static_cast<unsigned long long>(base), t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
                }
            }
        }
    }
    // Debug aid: APS5_DEBUG_LOG_DST_ALPHA=<color target> prints the textures bound by each draw into
    // that target that multiplies the destination by the source alpha, so the texture whose alpha
    // erases the target can be named.
    {
        static const std::uint64_t wanted = [] { const char* text = std::getenv("APS5_DEBUG_LOG_DST_ALPHA"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        if (wanted != 0 && !graphics.colors.empty() && graphics.colors.front().address == wanted && !graphics.blends.empty() &&
            graphics.blends.front().blendEnable && graphics.blends.front().srcColorBlendFactor == VK_BLEND_FACTOR_ZERO && graphics.blends.front().dstColorBlendFactor == VK_BLEND_FACTOR_SRC_ALPHA) {
            static std::mutex alphaMutex;
            static std::set<std::uint64_t> alphaSeen;
            std::lock_guard alphaLock(alphaMutex);
            if (!work.stages.empty() && alphaSeen.insert(work.stages.back().programAddress).second) {
                for (const auto& stage : work.stages) {
                    for (std::size_t index = 0; index < stage.shader->nativeImageSources.size() && stage.slotWords.size() >= (index + 1) * 8; ++index) {
                        const auto* t = stage.slotWords.data() + index * 8;
                        const auto base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
                        std::uint32_t sampled = 0, storage = 0;
                        const bool resolved = s.images->ImageIndices(std::span<const std::uint32_t, 8>(t, 8), sampled, storage);
                        std::fprintf(stderr, "[native] dst-alpha draw ps 0x%llx stage %d image slot %zu base 0x%llx T# %08x %08x %08x %08x -> sampled %u storage %u resolved %d\n",
                            static_cast<unsigned long long>(work.stages.back().programAddress), static_cast<int>(stage.stage), index, static_cast<unsigned long long>(base), t[0], t[1], t[2], t[3], sampled, storage, resolved ? 1 : 0);
                    }
                }
            }
        }
    }
    // Debug aid: APS5_DEBUG_DUMP_AT_COUNT=<n,...> writes the APS5_NATIVE_DUMP_ADDRESSES surfaces
    // before the n-th draw of the frame into one of them, so one frame can be read as a sequence.
    {
        static const std::vector<std::uint64_t> atCounts = AddressList("APS5_DEBUG_DUMP_AT_COUNT");
        static const std::vector<std::uint64_t> named = AddressList("APS5_NATIVE_DUMP_ADDRESSES");
        static const double countAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto countStarted = std::chrono::steady_clock::now();
        static int countEvents = 0;
        if (!atCounts.empty() && !named.empty() && !graphics.colors.empty() &&
            std::find(named.begin(), named.end(), graphics.colors.front().address) != named.end()) {
            const auto number = FrameTargetDraws().fetch_add(1) + 1;
            if (countEvents < 6 && std::find(atCounts.begin(), atCounts.end(), number) != atCounts.end() && FrameCountGate().load()) {
                std::vector<NativeImages::DrawnTarget> surfaces;
                for (const auto address : named) {
                    const auto found = s.images->SurfacesAt(address);
                    surfaces.insert(surfaces.end(), found.begin(), found.end());
                }
                const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR");
                char prefix[32];
                std::snprintf(prefix, sizeof(prefix), "count%llu_", static_cast<unsigned long long>(number));
                ++countEvents;
                std::fprintf(stderr, "[native] frame draw %llu into 0x%llx (ps 0x%llx): dumping\n", static_cast<unsigned long long>(number),
                    static_cast<unsigned long long>(graphics.colors.front().address), static_cast<unsigned long long>(work.stages.empty() ? 0ull : work.stages.back().programAddress));
                DumpTargetSurfaces(s, std::filesystem::path(directory != nullptr ? directory : "native-targets"), surfaces, prefix);
            }
        }
    }
    // Debug aid: APS5_DEBUG_DUMP_AT_PS=<pixel program,...> writes the APS5_NATIVE_DUMP_ADDRESSES
    // surfaces just before that program's draw, so a target can be read mid-frame instead of at the
    // flip, where the next frame has already cleared it.
    {
        static const std::vector<std::uint64_t> atPs = AddressList("APS5_DEBUG_DUMP_AT_PS");
        static const std::vector<std::uint64_t> named = AddressList("APS5_NATIVE_DUMP_ADDRESSES");
        static const double atAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto atStarted = std::chrono::steady_clock::now();
        static int atEvents = 0;
        if (!atPs.empty() && !named.empty() && atEvents < 3 && !work.stages.empty() && std::find(atPs.begin(), atPs.end(), work.stages.back().programAddress) != atPs.end() &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - atStarted).count() >= atAfter) {
            std::vector<NativeImages::DrawnTarget> surfaces;
            for (const auto address : named) {
                const auto found = s.images->SurfacesAt(address);
                surfaces.insert(surfaces.end(), found.begin(), found.end());
            }
            const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR");
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "atps%d_", atEvents++);
            DumpTargetSurfaces(s, std::filesystem::path(directory != nullptr ? directory : "native-targets"), surfaces, prefix);
        }
    }
    // Debug aid: APS5_DEBUG_TARGET_STATE=<address,...> prints, once per pixel program, the whole
    // color-target and blend state of a draw into one of those targets, so a draw that lands can be
    // compared with one that does not.
    {
        static const std::vector<std::uint64_t> wanted = AddressList("APS5_DEBUG_TARGET_STATE");
        if (!wanted.empty() && !graphics.colors.empty() && !work.stages.empty()) {
            const bool hit = std::any_of(graphics.colors.begin(), graphics.colors.end(), [&](const auto& color) { return std::find(wanted.begin(), wanted.end(), color.address) != wanted.end(); });
            static std::mutex stateMutex;
            static std::set<std::uint64_t> stateSeen;
            std::unique_lock stateLock(stateMutex, std::defer_lock);
            if (hit) stateLock.lock();
            if (hit && stateSeen.size() < 24 && stateSeen.insert(work.stages.back().programAddress).second) {
                std::fprintf(stderr, "[native] target state ps 0x%llx: %zu colors, extent %ux%u, scissor %dx%d+%d+%d, samples %u, depth test %d write %d, a2c %d\n",
                    static_cast<unsigned long long>(work.stages.back().programAddress), graphics.colors.size(), graphics.renderExtent.width, graphics.renderExtent.height,
                    graphics.scissor.extent.width, graphics.scissor.extent.height, graphics.scissor.offset.x, graphics.scissor.offset.y, graphics.samples,
                    graphics.depthTest ? 1 : 0, graphics.depthWrite ? 1 : 0, graphics.alphaToCoverage ? 1 : 0);
                for (std::size_t index = 0; index < graphics.colors.size(); ++index) {
                    const auto& color = graphics.colors[index];
                    const auto& blend = index < graphics.blends.size() ? graphics.blends[index] : VkPipelineColorBlendAttachmentState{};
                    std::fprintf(stderr, "[native]   color %zu: 0x%llx slot %u export %u VkFormat %d guest %u map 0x%x packing %d uint %d extent %ux%u surface 0x%llx %ux%u mip %u/%u slice %u layers %u tile %u dcc 0x%llx cmask 0x%llx fast %d clear %08x %08x | blend %d mask %x src %d dst %d op %d\n",
                        index, static_cast<unsigned long long>(color.address), color.slot, color.exportIndex, static_cast<int>(color.format), color.guestFormat, color.componentMapping,
                        static_cast<int>(color.packing), color.uintExport ? 1 : 0, color.extent.width, color.extent.height, static_cast<unsigned long long>(color.surfaceAddress),
                        color.surfaceExtent.width, color.surfaceExtent.height, color.mip, color.mipCount, color.arraySlice, color.layers, static_cast<unsigned>(color.tileMode),
                        static_cast<unsigned long long>(color.dccAddress), static_cast<unsigned long long>(color.cmaskAddress), color.cmaskFastClear ? 1 : 0, color.clearWords[0], color.clearWords[1],
                        blend.blendEnable ? 1 : 0, blend.colorWriteMask, static_cast<int>(blend.srcColorBlendFactor), static_cast<int>(blend.dstColorBlendFactor), static_cast<int>(blend.colorBlendOp));
                }
            }
        }
    }
    s.images->Prepare(s.commands);
    phases.Mark(2);
    std::vector<VkImageView> views;
    VkImage firstColorImage = VK_NULL_HANDLE;
    try {
        for (const auto& color : graphics.colors) {
            VkImage image = VK_NULL_HANDLE;
            views.push_back(s.images->TargetView(s.commands, color, image));
            if (&color == &graphics.colors.front()) firstColorImage = image;
            // Debug aid: APS5_DEBUG_TARGET_CENSUS=<address,...> counts the draws into each distinct
            // VkImage at a listed target address, so an aliased target shows up as several images.
            {
                static const std::vector<std::uint64_t> census = AddressList("APS5_DEBUG_TARGET_CENSUS");
                if (!census.empty() && std::find(census.begin(), census.end(), color.address) != census.end() && image != VK_NULL_HANDLE) {
                    std::lock_guard censusLock(TargetCensusMutex());
                    auto& entry = TargetCensus()[std::make_pair(color.address, reinterpret_cast<std::uintptr_t>(image))];
                    ++entry.first;
                    if (entry.second == 0 && !work.stages.empty()) entry.second = work.stages.back().programAddress;
                }
            }
        }
        if (graphics.depth) views.push_back(s.images->DepthView(s.commands, *graphics.depth));
    } catch (const std::exception& error) {
        failure = std::string("render target: ") + error.what();
        return false;
    }
    {
        static std::set<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, int, bool, std::size_t>> seen;
        Graphics::ColorTarget none{};
        const auto& color = graphics.colors.empty() ? none : graphics.colors.front();
        // APS5_DEBUG_LOG_AFTER=<s> holds this log back until then, so its slots are spent on the frame
        // of interest instead of the first draws of the run.
        static const double logAfter = [] { const char* text = std::getenv("APS5_DEBUG_LOG_AFTER"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto logStarted = std::chrono::steady_clock::now();
        const bool logDue = std::chrono::duration<double>(std::chrono::steady_clock::now() - logStarted).count() >= logAfter;
        if (logDue && seen.size() < 200 && seen.insert({color.address, graphics.renderExtent.width, graphics.renderExtent.height, static_cast<int>(color.format), heap, work.stages.size()}).second) {
            std::string stages;
            for (const auto& stage : work.stages) {
                stages += " stage" + std::to_string(static_cast<int>(stage.stage)) + "(push@" + std::to_string(stage.pushOffset) + " data@" + std::to_string(static_cast<int>(stage.shader->pushDataStartDword)) + " roles";
                for (const auto& binding : stage.shader->bindings) stages += " " + std::to_string(static_cast<int>(binding.role)) + "/" + std::to_string(binding.descriptorSet) + ":" + std::to_string(binding.binding);
                stages += " data";
                for (const auto word : stage.shaderData) {
                    char text[16];
                    std::snprintf(text, sizeof(text), " %x", word);
                    stages += text;
                }
                stages += ")";
            }
            stages += " topology " + std::to_string(static_cast<int>(graphics.topology)) + " rect " + std::to_string(graphics.rectList) + " cull " + std::to_string(static_cast<int>(graphics.cullMode)) + " front " + std::to_string(static_cast<int>(graphics.frontFace)) + " mesh " + std::to_string(graphics.stages.mesh.has_value());
            if (Native::Logs()) std::fprintf(stderr, "[native] draw into 0x%llx image %p ps 0x%llx (%zu targets) %ux%u VkFormat %d blend %u mask 0x%x heap %d count %u:%s\n", static_cast<unsigned long long>(color.address),
                static_cast<const void*>(firstColorImage), static_cast<unsigned long long>(work.programAddresses[1]), graphics.colors.size(), graphics.renderExtent.width, graphics.renderExtent.height, static_cast<int>(color.format), graphics.blends.empty() ? 0u : graphics.blends.front().blendEnable, graphics.blends.empty() ? 0u : graphics.blends.front().colorWriteMask, heap, work.count, stages.c_str());
        }
    }
    const std::vector<std::shared_ptr<Graphics::StorageTexture>> owners(graphics.colors.size());
    auto framebuffer = pipeline->AcquireFramebuffer(views, owners, graphics.renderExtent, graphics.renderLayers);
    s.kept.push_back(framebuffer);
    s.kept.push_back(pipeline);
    // Shader data blocks: push constants (pushDataStartDword) or the ring.
    auto block = Graphics::AssemblePushConstants(compiled);
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(bufferBindingCount);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(bufferBindingCount);
    const auto alignment = std::max<VkDeviceSize>(s.context.limits.minStorageBufferOffsetAlignment, 16);
    for (const auto& stage : work.stages) {
        const auto dataBytes = stage.shaderData.size() * sizeof(std::uint32_t);
        VkDeviceSize dataOffset = 0;
        if (stage.shader->pushDataStartDword != ~0u) {
            const auto start = static_cast<std::size_t>(stage.shader->pushDataStartDword) * 4u;
            if (start + dataBytes > block.size()) {
                failure = "graphics shader data exceeds the push block";
                return false;
            }
            std::memcpy(block.data() + start, stage.shaderData.data(), dataBytes);
            s.writeInvocationAddresses(block.data() + start, dataBytes, stage.programAddress);
            s.writeSlots(block.data() + start, dataBytes, *stage.shader, stage.slotWords);
        } else if (dataBytes != 0) {
            dataOffset = (s.ringOffset + alignment - 1) & ~(alignment - 1);
            std::memcpy(static_cast<std::byte*>(s.ringMapping) + dataOffset, stage.shaderData.data(), dataBytes);
            s.writeInvocationAddresses(static_cast<std::byte*>(s.ringMapping) + dataOffset, dataBytes, stage.programAddress);
            s.writeSlots(static_cast<std::byte*>(s.ringMapping) + dataOffset, dataBytes, *stage.shader, stage.slotWords);
            s.ringOffset = dataOffset + dataBytes;
        }
        const auto flatBytes = std::max<VkDeviceSize>(stage.flattenedSrt.size() * sizeof(std::uint32_t), 16);
        VkDeviceSize flatOffset = 0;
        if (std::any_of(stage.shader->bindings.begin(), stage.shader->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; })) flatOffset = s.writeFlattenedSrt(stage.flattenedSrt, flatBytes);
        for (const auto& binding : stage.shader->bindings) {
            VkDescriptorBufferInfo info{};
            switch (binding.role) {
                case ShaderRecompiler::DescriptorRole::BdaPagetable: {
                    std::shared_lock imports(s.importsMutex);
                    info = {s.table, 0, s.tableUsedBytes};
                    break;
                }
                case ShaderRecompiler::DescriptorRole::FaultBuffer: info = {s.fault.buffer, 0, VK_WHOLE_SIZE}; break;
                case ShaderRecompiler::DescriptorRole::ShaderData: info = {s.ring.buffer, dataOffset, std::max<VkDeviceSize>(dataBytes, 16)}; break;
                case ShaderRecompiler::DescriptorRole::FlattenedSrt: info = {s.ring.buffer, flatOffset, flatBytes}; break;
                case ShaderRecompiler::DescriptorRole::Gds: s.prepareGds(); info = {s.gds->Handle(), 0, Pm4::GdsBytes}; break;
                default: continue;
            }
            if (std::any_of(writes.begin(), writes.end(), [&](const VkWriteDescriptorSet& w) { return w.dstBinding == binding.binding; })) continue;
            infos.push_back(info);
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstBinding = binding.binding;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &infos.back();
            writes.push_back(write);
        }
    }
    // Index buffer: the game's own memory, imported.
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceSize indexOffset = 0;
    if (work.indexed && !graphics.stages.mesh) {
        std::shared_lock imports(s.importsMutex);
        NotePoolUse("index", work.indexAddress);
        auto it = s.imports.upper_bound(work.indexAddress);
        if (it == s.imports.begin()) {
            failure = "index buffer outside imported memory";
            return false;
        }
        --it;
        if (work.indexAddress + static_cast<std::uint64_t>(work.count) * work.indexSize > it->second.end) {
            failure = "index buffer outside imported memory";
            return false;
        }
        indexBuffer = it->second.buffer;
        indexOffset = work.indexAddress - it->second.begin;
    }
    // Indirect draws: the records (and their count) in the game's imported memory.
    VkBuffer recordBuffer = VK_NULL_HANDLE, countBuffer = VK_NULL_HANDLE;
    VkDeviceSize recordOffset = 0, countOffset = 0;
    if (work.indirectArguments != 0) {
        std::shared_lock imports(s.importsMutex);
        const auto locate = [&](std::uint64_t address, std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset) {
            auto it = s.imports.upper_bound(address);
            if (it == s.imports.begin() || address + bytes > std::prev(it)->second.end) return false;
            --it;
            buffer = it->second.buffer;
            offset = address - it->second.begin;
            return true;
        };
        const auto recordBytes = work.indirectCount == 0 ? 0 : static_cast<std::uint64_t>(work.indirectCount - 1) * work.indirectStride + 20u;
        if (work.indirectCount == 0) return true;
        if (!locate(work.indirectArguments, recordBytes, recordBuffer, recordOffset) || (work.indirectCountAddress != 0 && !locate(work.indirectCountAddress, 4, countBuffer, countOffset))) {
            failure = "indirect draw records outside imported memory";
            return false;
        }
        static int traced = 0;
        if (Native::Logs() && (traced < 24 || (work.programAddresses[0] == 0x128822fe00ull && traced < 200))) {
            ++traced;
            std::array<std::uint32_t, 5> record{};
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(work.indirectArguments), 20)) std::memcpy(record.data(), reinterpret_cast<const void*>(work.indirectArguments), 20);
            std::fprintf(stderr, "[native] indirect draw vs 0x%llx: records 0x%llx x%u stride %u count at 0x%llx; index 0x%llx size %u x%u B; record now {%u %u %u %d %u}\n", static_cast<unsigned long long>(work.programAddresses[0]), static_cast<unsigned long long>(work.indirectArguments), work.indirectCount, work.indirectStride, static_cast<unsigned long long>(work.indirectCountAddress), static_cast<unsigned long long>(work.indexAddress), work.count, work.indexSize, record[0], record[1], record[2], static_cast<std::int32_t>(record[3]), record[4]);
        }
        if ((work.indirectCount > 1 && !s.context.multiDrawIndirect) || (countBuffer != VK_NULL_HANDLE && !s.context.drawIndirectCount)) {
            failure = "indirect draw needs multiDrawIndirect / drawIndirectCount";
            return false;
        }
    }
    std::uint32_t meshGroups = 0;
    if (graphics.stages.mesh) {
        const auto& mesh = *graphics.stages.mesh;
        const auto inputSize = mesh.inputPrimitive == 1 ? 1u : mesh.inputPrimitive == 2 ? 2u : 3u;
        if (work.count < inputSize || mesh.primitivesPerGroup == 0) return true;
        const auto step = mesh.inputPrimitive == 5 || mesh.inputPrimitive == 6 ? 1u : inputSize;
        const auto primitives = (work.count - inputSize) / step + 1u;
        meshGroups = (primitives - 1u) / mesh.primitivesPerGroup + 1u;
        const std::array<std::uint32_t, ShaderRecompiler::MeshDrawPushBytes / 4> words{work.count, work.firstVertex, work.firstInstance, work.indexed ? work.indexSize : 0u, 0u, 0u};
        std::memcpy(block.data() + ShaderRecompiler::MeshDrawPushOffsetBytes, words.data(), sizeof(words));
    }
    const bool targetsChanged = views != s.lastViews || graphics.renderExtent.width != s.lastExtent.width || graphics.renderExtent.height != s.lastExtent.height;
    const bool uploadsChanged = s.images->Recordings() != s.barrierRecordings;
    const bool samplesWritten = std::any_of(work.sampled.begin(), work.sampled.end(), [&](std::uint64_t address) { return s.writtenSinceBarrier.contains(address); });
    const bool barrierNeeded = s.graphicsBarrier || targetsChanged || uploadsChanged || samplesWritten || DrawBarriersRequested();
    if (barrierNeeded) {
        ++s.barrierCounts[s.graphicsBarrier ? 0 : targetsChanged ? 1 : uploadsChanged ? 2 : 3];
        s.endPass();
        s.writtenSinceBarrier.clear();
    }
    ++s.barrierCounts[4];
    for (const auto& color : graphics.colors) s.writtenSinceBarrier.insert(color.address);
    if (graphics.depth) s.writtenSinceBarrier.insert(graphics.depth->address);
    s.lastViews = views;
    s.lastExtent = graphics.renderExtent;
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (barrierNeeded) {
        vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        s.graphicsBarrier = false;
        s.barrierRecordings = s.images->Recordings();
    }
    for (const auto& stage : work.stages) {
        if (s.writesGuestMemory(*stage.shader)) {
            s.graphicsBarrier = true;
            static std::set<std::uint64_t> reported;
            if (Native::Logs() && reported.insert(stage.programAddress).second) std::fprintf(stderr, "[native] draw program 0x%llx (stage %d) stores to memory: barrier before the next draw\n", static_cast<unsigned long long>(stage.programAddress), static_cast<int>(stage.stage));
        }
    }
    std::int32_t query = -1;
    {
        static const std::uint32_t wanted = [] { const char* text = std::getenv("APS5_NATIVE_DRAW_STATS"); return text != nullptr ? static_cast<std::uint32_t>(std::atoi(text)) : 0u; }();
        // APS5_NATIVE_DRAW_STATS_AT=<s>: start measuring then, so the queries land on the frame of
        // interest instead of the first draws of the run.
        static const double statsAt = [] { const char* text = std::getenv("APS5_NATIVE_DRAW_STATS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto statsStarted = std::chrono::steady_clock::now();
        const bool statsDue = std::chrono::duration<double>(std::chrono::steady_clock::now() - statsStarted).count() >= statsAt;
        if (s.statistics == VK_NULL_HANDLE && wanted != 0 && statsDue) {
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            info.queryCount = 64;
            info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT | VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT | VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT;
            if (s.context.Function<PFN_vkCreateQueryPool>("vkCreateQueryPool")(s.vk, &info, nullptr, &s.statistics) == VK_SUCCESS) s.statisticsLeft = wanted;
        }
        static const std::uint64_t statisticsProgram = [] { const char* text = std::getenv("APS5_NATIVE_DRAW_STATS_PS"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        if (statsDue && s.statisticsLeft != 0 && s.statisticsUsed < 64 && (statisticsProgram == 0 || work.programAddresses[1] == statisticsProgram)) {
            s.endPass();
            --s.statisticsLeft;
            query = static_cast<std::int32_t>(s.statisticsUsed++);
            char name[256];
            std::snprintf(name, sizeof(name), "ps 0x%llx target 0x%llx count %u heap %d depth 0x%llx test %d write %d op %d clear %g bounds %d [%g %g] stencil %d (front op %d ref %u mask %x) cull %d front %d", static_cast<unsigned long long>(work.programAddresses[1]), static_cast<unsigned long long>(graphics.colors.empty() ? 0ull : graphics.colors.front().address), work.count, heap,
                static_cast<unsigned long long>(graphics.depth ? graphics.depth->address : 0ull), graphics.depthTest, graphics.depthWrite, static_cast<int>(graphics.depthCompare), graphics.depth ? graphics.depth->clearDepth : -1.0f, graphics.depthBoundsTest, graphics.minDepthBounds, graphics.maxDepthBounds,
                graphics.stencilTest, static_cast<int>(graphics.stencilFront.compareOp), graphics.stencilFront.reference, graphics.stencilFront.compareMask, static_cast<int>(graphics.cullMode), static_cast<int>(graphics.frontFace));
            s.statisticsNames.push_back(name);
            s.context.Function<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool")(s.commands, s.statistics, static_cast<std::uint32_t>(query), 1);
        }
    }
    char timedName[96] = "";
    if (State::GpuTimes() || s.tracing) std::snprintf(timedName, sizeof(timedName), "draw vs 0x%llx ps 0x%llx %ux%u", static_cast<unsigned long long>(work.programAddresses[0]), static_cast<unsigned long long>(work.programAddresses[1]), graphics.renderExtent.width, graphics.renderExtent.height);
    if (s.tracing) {
        char line[256];
        std::snprintf(line, sizeof(line), "%s target 0x%llx VkFormat %d count %u x%u heap %d depth %d", timedName, static_cast<unsigned long long>(graphics.colors.empty() ? 0ull : graphics.colors.front().address), static_cast<int>(graphics.colors.empty() ? VK_FORMAT_UNDEFINED : graphics.colors.front().format), work.count, work.instances, static_cast<int>(heap), static_cast<int>(graphics.depth.has_value()));
        s.frameTrace.emplace_back(line);
    }
    if (s.context.pipelineExecutableInfo) {
        static std::set<VkPipeline> reported;
        if (const auto handle = pipeline->BoundHandle(); reported.insert(handle).second) {
            std::fprintf(stderr, "[pipeline-stats] native graphics pipeline vs 0x%llx ps 0x%llx %ux%u%s\n", static_cast<unsigned long long>(work.programAddresses[0]), static_cast<unsigned long long>(work.programAddresses[1]), graphics.renderExtent.width, graphics.renderExtent.height, handle == pipeline->BoundHandle() ? "" : "");
            Graphics::LogPipelineStatistics_nid_no_patch(s.context, handle);
        }
    }
    auto passKey = pipeline->PassKey(*framebuffer, graphics.renderExtent, graphics);
    if (s.passOpen && !passKey.empty() && passKey == s.passKey) {
        pipeline->Continue(s.commands, graphics);
    } else {
        s.endPass();
        s.passTimed = State::GpuTimes() ? s.beginTimed(std::string("pass from ") + timedName) : -1;
        pipeline->Begin(s.commands, *framebuffer, graphics.renderExtent, graphics);
        ++s.barrierCounts[5];
        s.passOpen = true;
        s.passKey = std::move(passKey);
    }
    // Debug experiment: APS5_DEBUG_CLEAR_TARGET=<address> clears attachment 0 to a bright red inside
    // the pass of every draw whose first color target is that address and that writes at least
    // APS5_DEBUG_CLEAR_TARGETS colors, to show whether attachment 0 reaches the image the dump
    // reads. Not a fix.
    {
        static const std::vector<std::uint64_t> clearTargets = AddressList("APS5_DEBUG_CLEAR_TARGET");
        static const std::size_t clearColors = [] { const char* text = std::getenv("APS5_DEBUG_CLEAR_TARGETS"); return text != nullptr ? static_cast<std::size_t>(std::strtoul(text, nullptr, 0)) : 4u; }();
        static const double clearAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto clearStarted = std::chrono::steady_clock::now();
        if (!clearTargets.empty() && !graphics.colors.empty() && graphics.colors.size() >= clearColors &&
            std::find(clearTargets.begin(), clearTargets.end(), graphics.colors.front().address) != clearTargets.end() &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - clearStarted).count() >= clearAfter) {
            VkClearAttachment attachment{};
            attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            attachment.colorAttachment = 0;
            attachment.clearValue.color.float32[0] = 0.5f;
            attachment.clearValue.color.float32[3] = 1.0f;
            VkClearRect rect{};
            rect.rect.extent = graphics.renderExtent;
            rect.layerCount = 1;
            s.context.Function<PFN_vkCmdClearAttachments>("vkCmdClearAttachments")(s.commands, 1, &attachment, 1, &rect);
            static int cleared = 0;
            if (cleared++ < 8)
                std::fprintf(stderr, "[native] debug clear of attachment 0 (0x%llx, %zu colors, %ux%u) in the pass of ps 0x%llx\n", static_cast<unsigned long long>(graphics.colors.front().address),
                    graphics.colors.size(), graphics.renderExtent.width, graphics.renderExtent.height, static_cast<unsigned long long>(work.programAddresses[1]));
        }
    }
    if (query >= 0) s.context.Function<PFN_vkCmdBeginQuery>("vkCmdBeginQuery")(s.commands, s.statistics, static_cast<std::uint32_t>(query), 0);
    if (!writes.empty()) vkCmdPushDescriptorSetKHR(s.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->Layout(), 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    if (heap) {
        const auto set = s.images->HeapSet();
        vkCmdBindDescriptorSets(s.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->Layout(), ShaderRecompiler::RuntimeAbi::NativeHeap::Set, 1, &set, 0, nullptr);
    }
    VkShaderStageFlags pushStages = 0;
    for (const auto& stage : work.stages) pushStages |= Graphics::VulkanStage(stage.stage);
    pipeline->PushConstants(s.commands, pushStages, block);
    if (Native::Logs()) {
        char entry[128];
        std::snprintf(entry, sizeof(entry), "draw vs 0x%llx ps 0x%llx count %u x%u%s", static_cast<unsigned long long>(work.programAddresses[0]), static_cast<unsigned long long>(work.programAddresses[1]), work.count, work.instances, graphics.stages.mesh ? " mesh" : "");
        s.note(entry);
    }
    if (graphics.stages.mesh) {
        s.context.Function<PFN_vkCmdDrawMeshTasksEXT>("vkCmdDrawMeshTasksEXT")(s.commands, meshGroups, work.instances, 1);
    } else if (work.indexed) {
        if (!vertexBuffers.empty()) s.context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers")(s.commands, 0, static_cast<std::uint32_t>(vertexBuffers.size()), vertexBuffers.data(), vertexOffsets.data());
        s.context.Function<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer")(s.commands, indexBuffer, indexOffset, work.indexSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        if (countBuffer != VK_NULL_HANDLE) s.context.Function<PFN_vkCmdDrawIndexedIndirectCount>("vkCmdDrawIndexedIndirectCount")(s.commands, recordBuffer, recordOffset, countBuffer, countOffset, work.indirectCount, work.indirectStride);
        else if (recordBuffer != VK_NULL_HANDLE) s.context.Function<PFN_vkCmdDrawIndexedIndirect>("vkCmdDrawIndexedIndirect")(s.commands, recordBuffer, recordOffset, work.indirectCount, work.indirectStride);
        else s.context.Function<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed")(s.commands, work.count, work.instances, 0, static_cast<std::int32_t>(work.firstVertex), work.firstInstance);
    } else {
        if (!vertexBuffers.empty()) s.context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers")(s.commands, 0, static_cast<std::uint32_t>(vertexBuffers.size()), vertexBuffers.data(), vertexOffsets.data());
        s.context.Function<PFN_vkCmdDraw>("vkCmdDraw")(s.commands, work.count, work.instances, work.firstVertex, work.firstInstance);
    }
    if (query >= 0) s.context.Function<PFN_vkCmdEndQuery>("vkCmdEndQuery")(s.commands, s.statistics, static_cast<std::uint32_t>(query));
    if (query >= 0 || s.passKey.empty()) s.endPass();
    s.noteRecorded();
    phases.Mark(3);
    return true;
}

namespace {

std::uint32_t TexelBytes(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_S8_UINT: return 1;
        case VK_FORMAT_D16_UNORM: case VK_FORMAT_D16_UNORM_S8_UINT: return 2;
        case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT: return 4;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT: return 2;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32G32_SFLOAT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK: return 8;
        case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK: case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK: return 16;
        default: return 0;
    }
}

}

// Copies the listed targets into host buffers and writes them next to the frame trace. Used both
// by the flip-time dump and by the mid-frame dump APS5_DEBUG_DUMP_AT_PS asks for.
template <typename StateType>
void DumpTargetSurfaces(StateType& s, const std::filesystem::path& root, const std::vector<NativeImages::DrawnTarget>& targets, const std::string& prefix) {
        std::filesystem::create_directories(root);
        struct Pending {
            NativeImages::DrawnTarget target;
            Import buffer;
            void* mapping = nullptr;
            std::size_t bytes = 0;
        };
        std::vector<Pending> pending;
        s.begin();
        s.endPass();
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        for (const auto& target : targets) {
            const auto texel = TexelBytes(target.format);
            if (texel == 0) {
                std::fprintf(stderr, "[native] target dump: 0x%llx VkFormat %d not dumped\n", static_cast<unsigned long long>(target.address), static_cast<int>(target.format));
                continue;
            }
            Pending entry;
            entry.target = target;
            const bool blocks = target.format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && target.format <= VK_FORMAT_BC7_SRGB_BLOCK;
            entry.bytes = blocks ? static_cast<std::size_t>((target.width + 3u) / 4u) * ((target.height + 3u) / 4u) * texel : static_cast<std::size_t>(target.width) * target.height * texel;
            try {
                entry.buffer = s.hostBuffer(entry.bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, entry.mapping);
            } catch (const std::exception& error) {
                // A diagnostic: a target that cannot be staged is left out, the game keeps running.
                std::fprintf(stderr, "[native] target dump: 0x%llx (%zu bytes) not dumped: %s\n", static_cast<unsigned long long>(target.address), entry.bytes, error.what());
                continue;
            }
            VkBufferImageCopy region{};
            region.imageSubresource = {static_cast<VkImageAspectFlags>(target.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
            region.imageExtent = {target.width, target.height, 1};
            s.context.template Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(s.commands, target.image, VK_IMAGE_LAYOUT_GENERAL, entry.buffer.buffer, 1, &region);
            pending.push_back(std::move(entry));
        }
        s.flushLocked();
        for (const auto& entry : pending) {
            char name[192];
            {
                std::lock_guard censusLock(TargetCensusMutex());
                for (const auto& [key, value] : TargetCensus())
                    std::fprintf(stderr, "[native] target census 0x%llx image %p: %llu draws, first ps 0x%llx\n", static_cast<unsigned long long>(key.first), reinterpret_cast<const void*>(key.second),
                        static_cast<unsigned long long>(value.first), static_cast<unsigned long long>(value.second));
            }
            std::fprintf(stderr, "[native] dump surface 0x%llx image %p %ux%u VkFormat %d drawn %llu\n", static_cast<unsigned long long>(entry.target.address), static_cast<const void*>(entry.target.image),
                entry.target.width, entry.target.height, static_cast<int>(entry.target.format), static_cast<unsigned long long>(entry.target.drawn));
            std::snprintf(name, sizeof(name), "%s%s_%012llx_%ux%u_f%d_d%llu.bin", prefix.c_str(), entry.target.depth ? "depth" : "target", static_cast<unsigned long long>(entry.target.address), entry.target.width, entry.target.height, static_cast<int>(entry.target.format), static_cast<unsigned long long>(entry.target.drawn));
            std::ofstream file(root / name, std::ios::binary);
            file.write(static_cast<const char*>(entry.mapping), static_cast<std::streamsize>(entry.bytes));
            if (entry.target.guestBytes != 0 && GuestMemory::Accessible(reinterpret_cast<const void*>(entry.target.address), entry.target.guestBytes)) {
                std::snprintf(name, sizeof(name), "%sguest_%012llx_%ux%u_fmt%u_tile%u.bin", prefix.c_str(), static_cast<unsigned long long>(entry.target.address), entry.target.width, entry.target.height, entry.target.guestFormat, entry.target.tileMode);
                std::ofstream(root / name, std::ios::binary).write(reinterpret_cast<const char*>(entry.target.address), static_cast<std::streamsize>(entry.target.guestBytes));
            }
            s.context.template Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(s.vk, entry.buffer.buffer, nullptr);
            s.context.template Function<PFN_vkFreeMemory>("vkFreeMemory")(s.vk, entry.buffer.memory, nullptr);
        }
        std::fprintf(stderr, "[native] target dump: %zu surfaces written to %s (%s)\n", pending.size(), root.string().c_str(), prefix.c_str());
    }

void NativeDevice::NoteFlip() {
    FrameTargetDraws().store(0);
    {
        static const double countAfter = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto countStarted = std::chrono::steady_clock::now();
        FrameCountGate().store(std::chrono::duration<double>(std::chrono::steady_clock::now() - countStarted).count() >= countAfter);
    }
    {
        std::lock_guard lock(state->workMutex);
        state->frameScanPending = true;
    }
    static const double dumpAt = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : -1.0; }();
    static const std::vector<std::uint64_t> named = [] {
        std::vector<std::uint64_t> addresses;
        if (const char* text = std::getenv("APS5_NATIVE_DUMP_ADDRESSES")) {
            while (*text) {
                char* end = nullptr;
                const auto address = std::strtoull(text, &end, 0);
                if (end == text) break;
                addresses.push_back(address);
                text = *end == ',' ? end + 1 : end;
            }
        }
        return addresses;
    }();
    static const auto started = std::chrono::steady_clock::now();
    static std::uint64_t frameStart = 0;
    static bool dumped = false;
    static int namedEvents = 0;
    auto& s = *state;
    std::lock_guard lock(s.workMutex);
    const auto previous = frameStart;
    frameStart = s.images->DrawSerial();
    const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR");
    const std::filesystem::path root = directory != nullptr ? directory : "native-targets";
    const auto dumpSurfaces = [&](const std::vector<NativeImages::DrawnTarget>& targets, const std::string& prefix) { DumpTargetSurfaces(s, root, targets, prefix); };
    if (!named.empty() && namedEvents < 4) {
        bool hit = dumpAt >= 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= dumpAt + 10.0 * namedEvents;
        // With a dump time the named surfaces are dumped by time alone (every 10 s from it).
        if (dumpAt < 0) {
            for (const auto& target : s.images->DrawnTargets(previous)) hit = hit || std::find(named.begin(), named.end(), target.address) != named.end();
        }
        if (hit) {
            std::vector<NativeImages::DrawnTarget> surfaces;
            for (const auto address : named) {
                const auto found = s.images->SurfacesAt(address);
                surfaces.insert(surfaces.end(), found.begin(), found.end());
            }
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "named%d_", namedEvents++);
            dumpSurfaces(surfaces, prefix);
        }
    }
    // Debug aid: APS5_DEBUG_ZERO_SURFACES=<seconds> reads every storage surface back once at that
    // time and lists the ones whose level 0, layer 0 is entirely zero (compute output that never
    // landed, or inputs nothing produced).
    {
        static const double surveyAt = [] { const char* text = std::getenv("APS5_DEBUG_ZERO_SURFACES"); return text != nullptr ? std::atof(text) : -1.0; }();
        static bool surveyed = false;
        if (surveyAt >= 0 && !surveyed && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= surveyAt) {
            surveyed = true;
            std::size_t zero = 0, total = 0;
            for (const auto& target : s.images->StorageSurfaces()) {
                const auto texel = TexelBytes(target.format);
                const auto bytes = static_cast<VkDeviceSize>(target.width) * target.height * texel;
                if (texel == 0 || bytes > (64u << 20)) continue;
                void* mapping = nullptr;
                Import buffer;
                try {
                    buffer = s.hostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, mapping);
                } catch (const std::exception&) {
                    continue;
                }
                s.begin();
                s.endPass();
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {target.width, target.height, 1};
                s.context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(s.commands, target.image, VK_IMAGE_LAYOUT_GENERAL, buffer.buffer, 1, &region);
                s.flushLocked();
                const auto* words = static_cast<const std::uint8_t*>(mapping);
                bool any = false;
                for (VkDeviceSize i = 0; i < bytes && !any; ++i) any = words[i] != 0;
                ++total;
                if (!any) {
                    ++zero;
                    std::fprintf(stderr, "[native] zero surface: 0x%llx %ux%u guest format %u VkFormat %d tile %u drawn %llu\n", static_cast<unsigned long long>(target.address), target.width, target.height, target.guestFormat, static_cast<int>(target.format), target.tileMode, static_cast<unsigned long long>(target.drawn));
                }
                s.context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(s.vk, buffer.buffer, nullptr);
                s.context.Function<PFN_vkFreeMemory>("vkFreeMemory")(s.vk, buffer.memory, nullptr);
            }
            std::fprintf(stderr, "[native] zero surface survey: %zu of %zu storage surfaces are zero\n", zero, total);
        }
    }
    if (!s.requestedDumps.empty()) {
        static int requestedEvents = 0;
        std::vector<NativeImages::DrawnTarget> surfaces;
        for (const auto address : s.requestedDumps) {
            const auto found = s.images->SurfacesAt(address);
            surfaces.insert(surfaces.end(), found.begin(), found.end());
        }
        s.requestedDumps.clear();
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "traced%d_", requestedEvents++);
        dumpSurfaces(surfaces, prefix);
    }
    if (dumpAt < 0 || dumped || std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() < dumpAt) return;
    if (!s.tracing) {
        s.tracing = true;
        s.frameTrace.clear();
        return;
    }
    s.tracing = false;
    dumped = true;
    dumpSurfaces(s.images->DrawnTargets(previous), "");
    std::ofstream trace(root / "frame-trace.txt");
    for (const auto& line : s.frameTrace) trace << line << '\n';
}

void NativeDevice::RequestSurfaceDump(std::uint64_t address) {
    std::lock_guard lock(state->workMutex);
    state->requestedDumps.push_back(address);
}

bool NativeDevice::WritesGuestMemory(const ShaderRecompiler::RecompileResult& shader) {
    std::lock_guard lock(state->workMutex);
    return state->writesGuestMemory(shader);
}

void NativeDevice::MarkTextureMemoryWritten(std::uint64_t begin, std::uint64_t end) {
    std::lock_guard lock(state->workMutex);
    state->images->MarkGpuWritten(begin, end);
}

std::string NativeDevice::TextureOverlap(std::uint64_t begin, std::uint64_t end) {
    std::lock_guard lock(state->workMutex);
    return state->images->DescribeUploadedOverlap(begin, end);
}

bool NativeDevice::TextureRegistered(std::span<const std::uint32_t, 8> words) {
    std::lock_guard lock(state->workMutex);
    return state->images->Registered(words);
}

void NativeDevice::NoteSubmission() {
    // GT7 writes a frame's CPU data before submitting the frame, which follows the previous flip:
    // texture memory is checked for CPU writes once per frame, at its first submission.
    std::lock_guard lock(state->workMutex);
    if (!std::exchange(state->frameScanPending, false)) return;
    state->images->NewBatch();
}

void NativeDevice::RequireBarrier() {
    std::lock_guard lock(state->workMutex);
    ++state->syncPackets;
}

ShaderRecompiler::SpirvTarget NativeDevice::GraphicsTarget() const {
    auto target = state->driverDevice->Target();
    target.gpuResolvedResources = true;
    target.registrationModules = true;
    target.nativeImageTable = 1u;
    target.nativeSamplerTable = 1u;
    return target;
}

bool NativeDevice::DisplayImage(std::uint64_t address, VkImage& image, VkExtent2D& extent, VkFormat& format) {
    std::lock_guard lock(state->workMutex);
    return state->images->Display(address, image, extent, format);
}

void NativeDevice::Flush() {
    std::lock_guard lock(state->workMutex);
    state->flushLocked();
}

void NativeDevice::Submit() {
    std::lock_guard lock(state->workMutex);
    state->flushLocked(false);
}

void NativeDevice::AfterBatch(std::function<void()> action) {
    {
        std::lock_guard lock(state->workMutex);
        if (state->recording) {
            state->afterBatch.push_back(std::move(action));
            return;
        }
        std::lock_guard completion(state->completionMutex);
        if (!state->completionQueue.empty()) {
            state->slots[state->completionQueue.back()].afterBatch.push_back(std::move(action));
            return;
        }
    }
    action();
}

void NativeDevice::AfterBatch(std::function<void()> action, std::uint64_t serial) {
    {
        std::lock_guard lock(state->workMutex);
        if (serial > state->completedSerial.load(std::memory_order_acquire)) {
            if (state->recording && state->openSerial == serial) {
                state->afterBatch.push_back(std::move(action));
                return;
            }
            std::lock_guard completion(state->completionMutex);
            for (const auto index : state->completionQueue) {
                if (state->slots[index].serial < serial) continue;
                state->slots[index].afterBatch.push_back(std::move(action));
                return;
            }
            if (state->recording) {
                state->afterBatch.push_back(std::move(action));
                return;
            }
        }
    }
    action();
}

std::uint64_t NativeDevice::RecordingSerial() const {
    std::lock_guard lock(state->workMutex);
    return state->openSerial;
}

bool NativeDevice::Pending() const {
    std::lock_guard lock(state->workMutex);
    if (state->recording) return true;
    std::lock_guard completion(state->completionMutex);
    return !state->completionQueue.empty();
}

std::shared_ptr<Graphics::PipelineJournalReplay> NativeDevice::ReplayPipelineJournal(bool prepareAll) {
    Graphics::PipelineJournalHooks hooks;
    hooks.nativeGraphics = [this, prepareAll](const Graphics::GraphicsPipelineRecipe& recipe, std::span<const Graphics::CompiledShader> shaders) {
        return ReplayGraphicsPipeline(recipe, shaders, prepareAll);
    };
    hooks.prepareAll = prepareAll;
    return state->driverDevice->ReplayPipelineJournal(std::move(hooks));
}

bool NativeDevice::ReplayGraphicsPipeline(const Graphics::GraphicsPipelineRecipe& recipe, std::span<const Graphics::CompiledShader> shaders, bool prepareAll) {
    auto& s = *state;
    if (recipe.splitSets || !recipe.fragmentLayoutKey.empty()) throw std::runtime_error("native pipeline journal requires a single stage-buffer set");
    for (const auto& shader : shaders) {
        for (const auto& binding : shader.program->bindings) {
            if (binding.descriptorSet == ShaderRecompiler::RuntimeAbi::NativeHeap::Set && (binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers)) continue;
            if (binding.descriptorSet != 0 || binding.count != 1) throw std::runtime_error("native pipeline journal has an unsupported buffer set or count");
            if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer && binding.role != ShaderRecompiler::DescriptorRole::ShaderData && binding.role != ShaderRecompiler::DescriptorRole::Gds && binding.role != ShaderRecompiler::DescriptorRole::FlattenedSrt) throw std::runtime_error("native pipeline journal has an unsupported buffer role");
            bool found = false;
            for (std::size_t index = 0; index + 1 < s.graphicsLayoutKey.size(); index += 2) {
                if (s.graphicsLayoutKey[index] == binding.binding && (s.graphicsLayoutKey[index + 1] & Graphics::VulkanStage(shader.stage)) != 0) found = true;
            }
            if (!found) throw std::runtime_error("native pipeline journal binding is outside the canonical layout");
        }
    }
    const Graphics::PipelineSetLayouts layouts{s.graphicsLayout, VK_NULL_HANDLE, false, s.graphicsLayoutKey, {}, s.images->HeapLayout(), s.emptyLayout};
    bool created = false;
    const auto pipeline = Graphics::CachedPipeline(s.context, recipe.state, recipe.vertexInput, layouts, shaders, recipe.attachmentLayout, &created);
    if (prepareAll) pipeline->PrepareOptimization();
    return created;
}

bool NativeDevice::PrebuildGraphicsLibrary(const ShaderRecompiler::RecompileResult& shader, ShaderRecompiler::ShaderStage stage, std::uint32_t wave, std::uint32_t pushOffset, const std::optional<ShaderRecompiler::MeshConfiguration>& mesh) {
    auto& s = *state;
    std::unique_lock lock(s.workMutex);
    const auto& key = s.graphicsLayoutKey;
    const auto layout = s.graphicsLayout;
    for (const auto& binding : shader.bindings) {
        if (binding.descriptorSet == ShaderRecompiler::RuntimeAbi::NativeHeap::Set) continue;
        if (binding.descriptorSet != 0) throw std::runtime_error("native graphics preparation requires stage buffers in set 0");
        if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer && binding.role != ShaderRecompiler::DescriptorRole::ShaderData && binding.role != ShaderRecompiler::DescriptorRole::Gds && binding.role != ShaderRecompiler::DescriptorRole::FlattenedSrt) throw std::runtime_error("unsupported native graphics preparation binding role");
    }
    const Graphics::PipelineSetLayouts layouts{layout, VK_NULL_HANDLE, false, key, {}, s.images->HeapLayout(), s.emptyLayout};
    lock.unlock();
    const bool fragment = stage == ShaderRecompiler::ShaderStage::Fragment;
    Graphics::ShaderStages stages{Graphics::ShaderPath::Vertex, 0u, fragment ? 32u : wave, fragment ? wave : 32u, mesh, {}};
    const std::array<Graphics::CompiledShader, 1> shaders{{{stage, &shader, pushOffset}}};
    return s.driverDevice->PrebuildGraphicsLibraries(stages, shaders, &layouts);
}

bool NativeDevice::PrebuildCompute(const ShaderRecompiler::RecompileResult& shader, std::string& failure) {
    std::unique_lock lock(state->workMutex);
    return state->pipelineFor(shader, shader.PipelineVariantId(), failure, lock) != nullptr;
}

ShaderRecompiler::SpirvTarget NativeDevice::ComputeTarget(std::uint32_t waveSize) const {
    auto target = state->driverDevice->ComputeTarget(waveSize);
    target.gpuResolvedResources = true;
    target.registrationModules = true;
    target.nativeImageTable = 1u;
    target.nativeSamplerTable = 1u;
    return target;
}

void NoteGpuMappingsChanged() {
    // The game maps memory in bursts while it loads: each change is noted, and the imports and page
    // table are brought up to date once, before the next GPU work that can see them.
    MappingGeneration().fetch_add(1, std::memory_order_acq_rel);
}

}
