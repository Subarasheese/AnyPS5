#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeImages.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "RuntimeAbi.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"

#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "BdaAbi.hpp"
#include <spirv/unified1/spirv.hpp>
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
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
#include <tuple>
#include <sys/ioctl.h>
#include <unistd.h>

namespace AgcDriver::Native {

namespace {

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
        std::fprintf(stderr, "[native] device-local LDS scratch: %llu bytes for %ux%ux%u groups (stride %u dwords)\n",
            static_cast<unsigned long long>(size), groups[0], groups[1], groups[2], dwords);
        workgroupScratch = made;
        return made;
    }
    // Graphics: one push-descriptor layout per set of stage buffer bindings, and what recorded
    // draws reference until their batch completed.
    std::map<std::vector<std::uint32_t>, VkDescriptorSetLayout> graphicsLayouts;
    std::vector<std::shared_ptr<void>> kept;
    std::vector<std::function<void()>> afterBatch;
    // What the open batch records, for a batch that does not complete (the last 48 entries).
    std::vector<std::string> batchLog;
    // APS5_NATIVE_GPU_TIMES=1: GPU time per program (timestamps around each draw and dispatch).
    VkQueryPool timestamps = VK_NULL_HANDLE;
    std::vector<std::string> timedNames;
    std::map<std::string, std::pair<double, std::uint64_t>> gpuTimes;
    std::chrono::steady_clock::time_point timesReported = std::chrono::steady_clock::now();
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
        std::fprintf(stderr, "[native] GPU time over 5 s: %.0f ms; top:\n", total);
        for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 12); ++i) std::fprintf(stderr, "[native]   %8.1f ms  %s\n", sorted[i].first, sorted[i].second.c_str());
        gpuTimes.clear();
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

    Import hostBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, void*& mapping) {
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
        if (allocation.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk, &allocation, nullptr, &made.memory) != VK_SUCCESS) throw std::runtime_error("native device: buffer memory");
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
        vkAllocateCommandBuffers(vk, &bufferInfo, &commands);
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(vk, &fenceInfo, nullptr, &fence);
        ring = hostBuffer(ringBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ringMapping);
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
                BackgroundSlot slot;
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
        pipelineReady.wait(lock, [&] { return !building.contains(key); });
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
        info.layout = made.layout;
        const auto started = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[native] building compute pipeline: %zu SPIR-V words, %zu bindings\n", shader.spirv.size(), shader.bindings.size());
        const auto result = vkCreateComputePipelines(vk, VK_NULL_HANDLE, 1, &info, nullptr, &made.pipeline);
        std::fprintf(stderr, "[native] compute pipeline built in %.0f ms (%d)\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(), static_cast<int>(result));
        vkDestroyShaderModule(vk, module, nullptr);
        if (result != VK_SUCCESS) {
            failure = "vkCreateComputePipelines " + std::to_string(static_cast<int>(result));
            return std::nullopt;
        }
        return made;
    }

    void flushLocked() {
        auto& s = *this;
    if (!s.recording) return;
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(s.commands);
        s.recording = false;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &s.commands;
        {
            // The driver device's queue is shared with its presentation: submissions hold the GPU lock.
            std::lock_guard queue(GuestMemory::GpuMutex());
            if (vkQueueSubmit(s.context.queue, 1, &submit, s.fence) != VK_SUCCESS) throw std::runtime_error("native device: vkQueueSubmit");
        }
        const auto waitStart = std::chrono::steady_clock::now();
        const auto waited = vkWaitForFences(s.vk, 1, &s.fence, VK_TRUE, 10'000'000'000ull);
        {
            // APS5_NATIVE_STATS: batches and the time the executor waits for the GPU on them.
            static std::uint64_t flushes = 0, recorded = 0;
            static double waitedMs = 0.0, worstMs = 0.0;
            static auto reported = std::chrono::steady_clock::now();
            const auto now = std::chrono::steady_clock::now();
            const auto ms = std::chrono::duration<double, std::milli>(now - waitStart).count();
            ++flushes;
            recorded += s.batchLog.size();
            waitedMs += ms;
            worstMs = std::max(worstMs, ms);
            if (now - reported > std::chrono::seconds(5)) {
                std::fprintf(stderr, "[native] batches: %llu in 5 s, GPU wait %.0f ms total (worst %.1f ms), %.1f entries per batch\n", static_cast<unsigned long long>(flushes), waitedMs, worstMs, flushes ? static_cast<double>(recorded) / flushes : 0.0);
                flushes = recorded = 0;
                waitedMs = worstMs = 0.0;
                reported = now;
            }
        }
        // Read the first-claim host-visible diagnostics before any failure
        // throws. On an incomplete/lost batch this is a best-effort snapshot;
        // leave it untouched rather than racing an outstanding GPU writer.
        for (const auto word : {0u, ShaderRecompiler::BdaAbi::NativeDiagnosticWord}) {
            auto* faultRecord = reinterpret_cast<ShaderRecompiler::BdaAbi::Fault*>(static_cast<std::uint32_t*>(s.faultMapping) + word);
            if (faultRecord->state == ShaderRecompiler::BdaAbi::FaultState::Ready) {
                std::fprintf(stderr, "[native] GPU address fault: reason %u address 0x%llx bytes %u pc 0x%x\n", static_cast<unsigned>(faultRecord->reason), static_cast<unsigned long long>(faultRecord->address), faultRecord->bytes, faultRecord->instruction);
                if (waited == VK_SUCCESS) std::memset(faultRecord, 0, sizeof(*faultRecord));
            }
        }
        if (waited != VK_SUCCESS) {
            std::fprintf(stderr, "[native] batch failed (%d); its last work:\n", static_cast<int>(waited));
            for (const auto& entry : s.batchLog) std::fprintf(stderr, "[native]   %s\n", entry.c_str());
            throw std::runtime_error("native device: batch did not complete (" + std::to_string(static_cast<int>(waited)) + ")");
        }
        // A timeout does not mean the queue retired: never reset its fence,
        // release kept resources, or destroy retired imports on that path.
        vkResetFences(s.vk, 1, &s.fence);
        if (waited == VK_SUCCESS) s.images->Completed();
        if (waited == VK_SUCCESS) {
            auto actions = std::move(s.afterBatch);
            s.afterBatch.clear();
            for (auto& action : actions) action();
        }
        if (waited == VK_SUCCESS) s.collectTimes();
        else s.timedNames.clear();
        if (waited == VK_SUCCESS && s.statisticsUsed != 0) {
            std::vector<std::uint64_t> values(s.statisticsUsed * 6u);
            s.context.Function<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(s.vk, s.statistics, 0, s.statisticsUsed, values.size() * 8u, values.data(), 48, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            for (std::uint32_t q = 0; q < s.statisticsUsed; ++q) std::fprintf(stderr, "[native] draw stats %s: vertices %llu, VS %llu, clipping in %llu out %llu, FS %llu, TES %llu\n", s.statisticsNames[q].c_str(), static_cast<unsigned long long>(values[q * 6]), static_cast<unsigned long long>(values[q * 6 + 1]), static_cast<unsigned long long>(values[q * 6 + 2]), static_cast<unsigned long long>(values[q * 6 + 3]), static_cast<unsigned long long>(values[q * 6 + 4]), static_cast<unsigned long long>(values[q * 6 + 5]));
            s.statisticsUsed = 0;
            s.statisticsNames.clear();
        }
        s.kept.clear();
        {
            // Nothing recorded before this point is in flight any more.
            std::unique_lock imports(s.importsMutex);
            for (auto& import : s.retired) s.destroy(import);
            s.retired.clear();
        }
        s.batchLog.clear();

    }

    void begin() {
        if (recording) return;
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
        std::fprintf(stderr, "[native] allocation probe at %s: %d (thread %ld)\n", where, static_cast<int>(result), static_cast<long>(gettid()));
    }

    void destroy(Import& import) const {
        if (import.buffer != VK_NULL_HANDLE) vkDestroyBuffer(vk, import.buffer, nullptr);
        if (import.memory != VK_NULL_HANDLE) vkFreeMemory(vk, import.memory, nullptr);
        import = {};
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
            if (allocation.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk, &allocation, nullptr, &tableMemory) != VK_SUCCESS) throw std::runtime_error("native device: page table memory");
            vkBindBufferMemory(vk, table, tableMemory, 0);
            vkMapMemory(vk, tableMemory, 0, VK_WHOLE_SIZE, 0, &tableMapping);
            VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addressInfo.buffer = table;
            tableAddress = vkGetBufferDeviceAddress(vk, &addressInfo);
        }
        auto* words = static_cast<std::uint32_t*>(tableMapping);
        std::memset(words, 0, needed);
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
            if (reported++ < 4) std::fprintf(stderr, "[native] page table: %zu 64 KiB pages shared by two imports (the later one maps them)\n", shared);
        }
        tableUsedBytes = needed;
        ++tableGeneration;
    }
};

NativeDevice& NativeDevice::Create(std::shared_ptr<VulkanDevice> driverDevice) {
    std::lock_guard lock(CreateMutex());
    if (auto* existing = device.load()) return *existing;
    auto* made = new NativeDevice(std::move(driverDevice));
    device.store(made);
    made->SyncGpuMemory();
    const auto stats = made->Stats();
    std::fprintf(stderr, "[native] device ready: %zu GPU ranges imported (%.1f MiB), %zu failed\n", stats.imports, stats.importedBytes / 1048576.0, stats.failedImports);
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
    state->images = std::make_unique<NativeImages>(state->context, [this](std::uint64_t address, std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset) {
        std::shared_lock lock(state->importsMutex);
        auto it = state->imports.upper_bound(address);
        if (it == state->imports.begin()) return false;
        --it;
        if (address + bytes > it->second.end) return false;
        buffer = it->second.buffer;
        offset = address - it->second.begin;
        return true;
    });
    VkDescriptorSetLayoutCreateInfo emptyInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    if (vkCreateDescriptorSetLayout(state->vk, &emptyInfo, nullptr, &state->emptyLayout) != VK_SUCCESS) throw std::runtime_error("native device: empty set layout");
    const VkDescriptorSetLayoutBinding workgroupBinding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo workgroupInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    workgroupInfo.bindingCount = 1;
    workgroupInfo.pBindings = &workgroupBinding;
    if (vkCreateDescriptorSetLayout(state->vk, &workgroupInfo, nullptr, &state->workgroupLayout) != VK_SUCCESS)
        throw std::runtime_error("native LDS descriptor layout");
    std::fprintf(stderr, "[native] native device on %s, host import alignment %llu\n", state->driverDevice->DeviceName().c_str(), static_cast<unsigned long long>(state->importAlignment));
}

NativeDevice::~NativeDevice() {
    for (auto& [begin, import] : state->imports) state->destroy(import);
    for (auto& import : state->retired) state->destroy(import);
}

void NativeDevice::SyncGpuMemory() {
    // The registry's GPU ranges now, as import chunks; the lease is held only for the comparison.
    struct Chunk {
        std::uint64_t begin, bytes;
        bool writable;
    };
    std::vector<Chunk> wanted;
    {
        const auto lease = GuestAllocations::GuestAllocationsAcquireAll_nid_postfix();
        for (const auto& range : lease) {
            if (range == nullptr || !range->gpu || !range->readable || range->bytes == 0) continue;
            // Committed runs only: reserved pages of heaps the game commits on demand cannot be pinned.
            for (const auto& [runBegin, runEnd] : GuestMemory::CommittedRanges(range->address, range->bytes)) {
                const auto begin = runBegin & ~(state->importAlignment - 1);
                const auto end = runEnd & ~(state->importAlignment - 1);
                for (auto at = begin; at < end; at += State::ChunkBytes) wanted.push_back({at, std::min(State::ChunkBytes, end - at), range->writable});
            }
        }
    }
    std::sort(wanted.begin(), wanted.end(), [](const Chunk& a, const Chunk& b) { return a.begin < b.begin; });
    std::unique_lock lock(state->importsMutex);
    bool changed = false;
    for (auto it = state->imports.begin(); it != state->imports.end();) {
        const auto found = std::find_if(wanted.begin(), wanted.end(), [&](const Chunk& chunk) { return chunk.begin == it->second.begin && chunk.begin + chunk.bytes == it->second.end; });
        if (found != wanted.end()) {
            ++it;
            continue;
        }
        // Unmapped: recorded work may still read the import, so it is freed at the next idle point.
        state->importedBytes -= it->second.end - it->second.begin;
        state->retired.push_back(it->second);
        it = state->imports.erase(it);
        changed = true;
    }
    for (const auto& chunk : wanted) {
        if (const auto it = state->imports.find(chunk.begin); it != state->imports.end() && it->second.end == chunk.begin + chunk.bytes) continue;
        if (state->failed.count({chunk.begin, chunk.bytes}) != 0) continue;
        Import made;
        if (!state->importChunk(chunk.begin, chunk.bytes, chunk.writable, made)) {
            ++state->failedImports;
            state->probe("failed import");
            state->failed.insert({chunk.begin, chunk.bytes});
            std::fprintf(stderr, "[native] GPU memory 0x%llx+0x%llx could not be imported: %s\n", static_cast<unsigned long long>(chunk.begin), static_cast<unsigned long long>(chunk.bytes), state->lastFailure);
            continue;
        }
        state->imports[chunk.begin] = made;
        state->importedBytes += chunk.bytes;
        changed = true;
    }
    if (changed || state->table == VK_NULL_HANDLE) state->rebuildTable();
}

std::uint64_t NativeDevice::DeviceAddress(std::uint64_t address, std::uint64_t bytes) const {
    std::shared_lock lock(state->importsMutex);
    auto it = state->imports.upper_bound(address);
    if (it == state->imports.begin()) return 0;
    --it;
    if (address + bytes > it->second.end) return 0;
    return it->second.address + (address - it->second.begin);
}

NativeDevice::Statistics NativeDevice::Stats() const {
    std::shared_lock lock(state->importsMutex);
    return {state->imports.size(), state->importedBytes, state->failedImports, state->tableGeneration};
}

bool NativeDevice::DmaGds(std::span<const std::uint32_t> packet) {
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
    s.prepareGds();
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
    // DMA completion makes memory visible to subsequent CPU packet effects as
    // well as shaders; no GDS data is staged through host memory.
    s.flushLocked();
    return true;
}

bool NativeDevice::Dispatch(const ComputeWork& work, std::string& failure) {
    auto& s = *state;
    std::unique_lock lock(s.workMutex);
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
    VkDeviceSize dataOffset = 0;
    if (!pushed && dataBytes != 0) {
        const auto alignment = std::max<VkDeviceSize>(s.context.limits.minStorageBufferOffsetAlignment, 16);
        if (s.recording && ((s.ringOffset + alignment - 1) & ~(alignment - 1)) + dataBytes > s.ringBytes) s.flushLocked();
    }
    s.begin();
    {
        const auto uploads = s.beginTimed("image creation and uploads");
        s.images->Prepare(s.commands);
        s.endTimed(uploads);
    }
    if (!pushed && dataBytes != 0) {
        const auto alignment = std::max<VkDeviceSize>(s.context.limits.minStorageBufferOffsetAlignment, 16);
        dataOffset = (s.ringOffset + alignment - 1) & ~(alignment - 1);
        std::memcpy(static_cast<std::byte*>(s.ringMapping) + dataOffset, work.shaderData.data(), dataBytes);
        s.writeInvocationAddresses(static_cast<std::byte*>(s.ringMapping) + dataOffset, dataBytes, work.programAddress);
        s.ringOffset = dataOffset + dataBytes;
    }
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
        }
        vkCmdPushConstants(s.commands, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, push.data());
    }
    {
        char entry[96];
        std::snprintf(entry, sizeof(entry), "dispatch 0x%llx %ux%ux%u", static_cast<unsigned long long>(work.programAddress), work.groups[0], work.groups[1], work.groups[2]);
        s.note(entry);
    }
    char timedName[64];
    std::snprintf(timedName, sizeof(timedName), "dispatch 0x%llx", static_cast<unsigned long long>(work.programAddress));
    const auto timed = s.beginTimed(timedName);
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
    s.endTimed(timed);
    return true;
}

bool NativeDevice::Draw(const DrawWork& work, std::string& failure) {
    auto& s = *state;
    std::unique_lock lock(s.workMutex);
    s.prepareCompute();
    const auto& graphics = *work.state;
    if (graphics.colors.empty() && !graphics.depth) {
        failure = "draw without color or depth targets";
        return false;
    }
    // The stages' buffer bindings, one push-descriptor set (every stage's bindings have their own
    // numbers, RuntimeAbi::BindingNumber), and the shader data blocks that do not fit push constants.
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<std::uint32_t> layoutKey;
    bool heap = false;
    std::size_t ringBytes = 0;
    for (const auto& stage : work.stages) {
        const auto stageFlag = Graphics::VulkanStage(stage.stage);
        if (stage.shader->pushDataStartDword == ~0u) ringBytes += (stage.shaderData.size() * 4u + 255u) & ~std::size_t{255};
        for (const auto& binding : stage.shader->bindings) {
            if (binding.descriptorSet == ShaderRecompiler::RuntimeAbi::NativeHeap::Set && (binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers)) {
                heap = true;
                continue;
            }
            if (binding.role != ShaderRecompiler::DescriptorRole::BdaPagetable && binding.role != ShaderRecompiler::DescriptorRole::FaultBuffer && binding.role != ShaderRecompiler::DescriptorRole::ShaderData && binding.role != ShaderRecompiler::DescriptorRole::Gds) {
                failure = "graphics binding role " + std::to_string(static_cast<int>(binding.role)) + " is not native yet";
                return false;
            }
            if (binding.descriptorSet != 0) {
                failure = "graphics stage buffers outside set 0";
                return false;
            }
            const auto found = std::find_if(bindings.begin(), bindings.end(), [&](const VkDescriptorSetLayoutBinding& b) { return b.binding == binding.binding; });
            if (found != bindings.end()) found->stageFlags |= stageFlag;
            else bindings.push_back({binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, static_cast<VkShaderStageFlags>(stageFlag), nullptr});
        }
    }
    std::sort(bindings.begin(), bindings.end(), [](const auto& a, const auto& b) { return a.binding < b.binding; });
    for (const auto& binding : bindings) layoutKey.insert(layoutKey.end(), {binding.binding, binding.stageFlags});
    layoutKey.push_back(heap ? 0x4e48u : 0u);
    auto& setLayout = s.graphicsLayouts[layoutKey];
    if (setLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(s.vk, &info, nullptr, &setLayout) != VK_SUCCESS) {
            failure = "graphics descriptor set layout";
            return false;
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
            auto it = s.imports.upper_bound(address);
            static int reported = 0;
            if (reported < 24) {
                ++reported;
                std::fprintf(stderr, "[native] vertex attribute %u: V# %08x %08x %08x %08x address 0x%llx %s; viewport %.0f,%.0f %.0fx%.0f scissor %d,%d %ux%u\n", attribute.location, fields[0], fields[1], fields[2], fields[3], static_cast<unsigned long long>(address), Graphics::NullVertexDescriptor(attribute) ? "null" : (it == s.imports.begin() || address >= std::prev(it)->second.end) ? "NOT IMPORTED" : "imported", graphics.viewport.x, graphics.viewport.y, graphics.viewport.width, graphics.viewport.height, graphics.scissor.offset.x, graphics.scissor.offset.y, graphics.scissor.extent.width, graphics.scissor.extent.height);
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
                        static_cast<void>(Graphics::BuildCachedPipeline(context, state, vertexInput, copied, compiled, VK_IMAGE_LAYOUT_GENERAL));
                    } catch (const std::exception& error) {
                        reason = std::string("graphics pipeline: ") + error.what();
                    }
                    std::lock_guard lock(s.workMutex);
                    s.buildingGraphics.erase(hash);
                    if (!reason.empty()) s.failedGraphics.emplace(hash, reason);
                    s.pipelineReady.notify_all();
                }).detach();
            }
            s.pipelineReady.wait(lock, [&] { return !s.buildingGraphics.contains(hash); });
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
    if (s.recording && ringBytes != 0 && s.ringOffset + ringBytes + 256 > s.ringBytes) s.flushLocked();
    s.begin();
    {
        const auto uploads = s.beginTimed("image creation and uploads");
        s.images->Prepare(s.commands);
        s.endTimed(uploads);
    }
    std::vector<VkImageView> views;
    try {
        for (const auto& color : graphics.colors) {
            VkImage image = VK_NULL_HANDLE;
            views.push_back(s.images->TargetView(s.commands, color, image));
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
        if (seen.size() < 48 && seen.insert({color.address, graphics.renderExtent.width, graphics.renderExtent.height, static_cast<int>(color.format), heap, work.stages.size()}).second) {
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
            std::fprintf(stderr, "[native] draw into 0x%llx (%zu targets) %ux%u VkFormat %d blend %u mask 0x%x heap %d count %u:%s\n", static_cast<unsigned long long>(color.address), graphics.colors.size(), graphics.renderExtent.width, graphics.renderExtent.height, static_cast<int>(color.format), graphics.blends.empty() ? 0u : graphics.blends.front().blendEnable, graphics.blends.empty() ? 0u : graphics.blends.front().colorWriteMask, heap, work.count, stages.c_str());
        }
    }
    const std::vector<std::shared_ptr<Graphics::StorageTexture>> owners(graphics.colors.size());
    auto framebuffer = pipeline->AcquireFramebuffer(views, owners, graphics.renderExtent);
    s.kept.push_back(framebuffer);
    s.kept.push_back(pipeline);
    // Shader data blocks: push constants (pushDataStartDword) or the ring.
    auto block = Graphics::AssemblePushConstants(compiled);
    std::vector<VkDescriptorBufferInfo> infos;
    infos.reserve(bindings.size());
    std::vector<VkWriteDescriptorSet> writes;
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
        } else if (dataBytes != 0) {
            dataOffset = (s.ringOffset + alignment - 1) & ~(alignment - 1);
            std::memcpy(static_cast<std::byte*>(s.ringMapping) + dataOffset, stage.shaderData.data(), dataBytes);
            s.writeInvocationAddresses(static_cast<std::byte*>(s.ringMapping) + dataOffset, dataBytes, stage.programAddress);
            s.ringOffset = dataOffset + dataBytes;
        }
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
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.srcAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(s.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    std::int32_t query = -1;
    {
        static const std::uint32_t wanted = [] { const char* text = std::getenv("APS5_NATIVE_DRAW_STATS"); return text != nullptr ? static_cast<std::uint32_t>(std::atoi(text)) : 0u; }();
        if (s.statistics == VK_NULL_HANDLE && wanted != 0) {
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            info.queryCount = 64;
            info.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT | VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT | VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT;
            if (s.context.Function<PFN_vkCreateQueryPool>("vkCreateQueryPool")(s.vk, &info, nullptr, &s.statistics) == VK_SUCCESS) s.statisticsLeft = wanted;
        }
        if (s.statisticsLeft != 0 && s.statisticsUsed < 64) {
            --s.statisticsLeft;
            query = static_cast<std::int32_t>(s.statisticsUsed++);
            char name[96];
            std::snprintf(name, sizeof(name), "0x%llx count %u heap %d depth %d", static_cast<unsigned long long>(graphics.colors.empty() ? 0ull : graphics.colors.front().address), work.count, heap, graphics.depth.has_value());
            s.statisticsNames.push_back(name);
            s.context.Function<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool")(s.commands, s.statistics, static_cast<std::uint32_t>(query), 1);
        }
    }
    char timedName[96];
    std::snprintf(timedName, sizeof(timedName), "draw vs 0x%llx ps 0x%llx %ux%u", static_cast<unsigned long long>(work.programAddresses[0]), static_cast<unsigned long long>(work.programAddresses[1]), graphics.renderExtent.width, graphics.renderExtent.height);
    const auto timed = s.beginTimed(timedName);
    pipeline->Begin(s.commands, *framebuffer, graphics.renderExtent, graphics);
    if (query >= 0) s.context.Function<PFN_vkCmdBeginQuery>("vkCmdBeginQuery")(s.commands, s.statistics, static_cast<std::uint32_t>(query), 0);
    if (!writes.empty()) vkCmdPushDescriptorSetKHR(s.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->Layout(), 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    if (heap) {
        const auto set = s.images->HeapSet();
        vkCmdBindDescriptorSets(s.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->Layout(), ShaderRecompiler::RuntimeAbi::NativeHeap::Set, 1, &set, 0, nullptr);
    }
    VkShaderStageFlags pushStages = 0;
    for (const auto& stage : work.stages) pushStages |= Graphics::VulkanStage(stage.stage);
    pipeline->PushConstants(s.commands, pushStages, block);
    {
        char entry[128];
        std::snprintf(entry, sizeof(entry), "draw vs 0x%llx ps 0x%llx count %u x%u%s", static_cast<unsigned long long>(work.programAddresses[0]), static_cast<unsigned long long>(work.programAddresses[1]), work.count, work.instances, graphics.stages.mesh ? " mesh" : "");
        s.note(entry);
    }
    if (graphics.stages.mesh) {
        s.context.Function<PFN_vkCmdDrawMeshTasksEXT>("vkCmdDrawMeshTasksEXT")(s.commands, meshGroups, work.instances, 1);
    } else if (work.indexed) {
        if (!vertexBuffers.empty()) s.context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers")(s.commands, 0, static_cast<std::uint32_t>(vertexBuffers.size()), vertexBuffers.data(), vertexOffsets.data());
        s.context.Function<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer")(s.commands, indexBuffer, indexOffset, work.indexSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        s.context.Function<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed")(s.commands, work.count, work.instances, 0, static_cast<std::int32_t>(work.firstVertex), work.firstInstance);
    } else {
        if (!vertexBuffers.empty()) s.context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers")(s.commands, 0, static_cast<std::uint32_t>(vertexBuffers.size()), vertexBuffers.data(), vertexOffsets.data());
        s.context.Function<PFN_vkCmdDraw>("vkCmdDraw")(s.commands, work.count, work.instances, work.firstVertex, work.firstInstance);
    }
    if (query >= 0) s.context.Function<PFN_vkCmdEndQuery>("vkCmdEndQuery")(s.commands, s.statistics, static_cast<std::uint32_t>(query));
    Graphics::EndRenderPass(s.context, s.commands);
    s.endTimed(timed);
    return true;
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

void NativeDevice::AfterBatch(std::function<void()> action) {
    {
        std::lock_guard lock(state->workMutex);
        if (state->recording) {
            state->afterBatch.push_back(std::move(action));
            return;
        }
    }
    action();
}

bool NativeDevice::Pending() const {
    std::lock_guard lock(state->workMutex);
    return state->recording;
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
    if (auto* current = NativeDevice::Get()) {
        try {
            current->SyncGpuMemory();
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[native] GPU memory import failed: %s\n", error.what());
        }
    }
}

}
