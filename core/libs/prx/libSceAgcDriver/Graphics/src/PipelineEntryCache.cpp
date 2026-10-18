#include "prx/libSceAgcDriver/Graphics/include/PipelineEntryCache.hpp"
#include "prx/libSceAgcDriver/Native/include/CompileStatus.hpp"
#include "ShaderCacheDirectory.hpp"
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

constexpr std::size_t VulkanHeaderBytes = 32;

struct DeviceEntries {
    std::vector<std::byte> header;
    std::filesystem::path directory;
};

const DeviceEntries* EntriesFor(const Context& context) {
    static std::mutex mutex;
    static std::map<VkDevice, DeviceEntries> devices;
    std::lock_guard lock(mutex);
    if (const auto found = devices.find(context.device); found != devices.end()) return found->second.directory.empty() ? nullptr : &found->second;
    auto& entries = devices[context.device];
    if (std::getenv("APS5_NO_PIPELINE_ENTRY_CACHE") != nullptr) return nullptr;
    const auto root = ShaderRecompiler::ShaderCacheDirectory();
    if (root.empty()) return nullptr;
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache empty = VK_NULL_HANDLE;
    if (context.Function<PFN_vkCreatePipelineCache>("vkCreatePipelineCache")(context.device, &info, nullptr, &empty) != VK_SUCCESS) return nullptr;
    const auto getData = context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData");
    std::size_t size = 0;
    std::vector<std::byte> data;
    if (getData(context.device, empty, &size, nullptr) == VK_SUCCESS && size >= VulkanHeaderBytes) {
        data.resize(size);
        if (getData(context.device, empty, &size, data.data()) != VK_SUCCESS) data.clear();
    }
    context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache")(context.device, empty, nullptr);
    if (data.size() < VulkanHeaderBytes) return nullptr;
    entries.header.assign(data.begin(), data.begin() + VulkanHeaderBytes);
    char name[64];
    std::snprintf(name, sizeof(name), "pipelines-%016llx", static_cast<unsigned long long>(ShaderRecompiler::HashBytes(entries.header)));
    entries.directory = root / name;
    return &entries;
}

}

VkResult CreateWithEntryCache(const Context& context, std::span<const std::byte> key, const std::function<VkResult(VkPipelineCache)>& create) {
    const auto* entries = EntriesFor(context);
    if (entries == nullptr) return create(context.pipelineCache);
    char name[48];
    std::snprintf(name, sizeof(name), "%016llx%016llx.bin", static_cast<unsigned long long>(ShaderRecompiler::HashBytes(key)), static_cast<unsigned long long>(ShaderRecompiler::HashBytes(key, 0x9e3779b97f4a7c15ull)));
    const auto path = entries->directory / std::string(name, 2) / name;
    std::vector<std::byte> stored;
    if (!ShaderRecompiler::ReadWholeFile(path, stored) || stored.size() < VulkanHeaderBytes || std::memcmp(stored.data(), entries->header.data(), VulkanHeaderBytes) != 0) stored.clear();
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = stored.size();
    info.pInitialData = stored.empty() ? nullptr : stored.data();
    std::optional<Native::CompileJob> job;
    if (stored.empty()) {
        job.emplace();
        job->Compiling();
    }
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (context.Function<PFN_vkCreatePipelineCache>("vkCreatePipelineCache")(context.device, &info, nullptr, &cache) != VK_SUCCESS) return create(context.pipelineCache);
    const auto result = create(cache);
    if (result == VK_SUCCESS) {
        const auto getData = context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData");
        std::size_t size = 0;
        if (getData(context.device, cache, &size, nullptr) == VK_SUCCESS && size > VulkanHeaderBytes && size != stored.size()) {
            std::vector<std::byte> data(size);
            if (getData(context.device, cache, &size, data.data()) == VK_SUCCESS) {
                data.resize(size);
                if (!ShaderRecompiler::WriteFileAtomically(path, data)) std::fprintf(stderr, "[pipeline-cache] cannot write %s\n", path.string().c_str());
            }
        }
    }
    context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache")(context.device, cache, nullptr);
    return result;
}

}
