#include "prx/libSceAgcDriver/Native/include/NativeImages.hpp"

#include "prx/libSceAgcDriver/Graphics/include/GuestSamplerResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <optional>
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "RuntimeAbi.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Native {

namespace Heap = ShaderRecompiler::RuntimeAbi::NativeHeap;
using Graphics::GuestTextureResource;
using Graphics::TextureDimension;

namespace {

// Descriptors noted by the game's initializer hooks, taken by the next Prepare.
struct Noted {
    std::mutex mutex;
    std::vector<std::array<std::uint32_t, 8>> textures;
    std::set<std::uint64_t> targets;
    std::size_t takenTargets = 0;
};

Noted& Pending() {
    static Noted noted;
    return noted;
}

constexpr std::uint32_t ImageSlots = 32768;
constexpr std::uint32_t ImageMisses = 4096;
constexpr std::uint32_t SamplerSlots = 8192;
constexpr std::uint32_t SamplerMisses = 1024;
constexpr std::uint32_t SamplerBinding = ShaderRecompiler::RuntimeAbi::FirstImageBinding + ShaderRecompiler::RuntimeAbi::ImageBindingCount;

enum class Numeric { Float, Uint, Sint };

Numeric NumericOf(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32:
        case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R64_UINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_S8_UINT:
            return Numeric::Uint;
        case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_B8G8R8A8_SINT:
            return Numeric::Sint;
        default:
            return Numeric::Float;
    }
}

// The dimension slots of a binding kind (DescriptorBindingForImage): base + this.
enum Slot : std::uint32_t { Slot1D = 0, Slot1DArray = 1, Slot2D = 2, Slot2DArray = 3, Slot3D = 6 };
constexpr std::uint32_t SampledBase[3] = {1u, 8u, 15u};
constexpr std::uint32_t StorageFloatBase = ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
constexpr std::uint32_t StorageUintBase = StorageFloatBase + 7u;
constexpr std::uint32_t AtomicUintBase = StorageUintBase + 7u;

VkComponentSwizzle Channel(std::uint32_t guestFormat, std::uint8_t selector) {
    switch (selector) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return Graphics::TextureComponentChannel(guestFormat, VK_COMPONENT_SWIZZLE_R);
        case 5: return Graphics::TextureComponentChannel(guestFormat, VK_COMPONENT_SWIZZLE_G);
        case 6: return Graphics::TextureComponentChannel(guestFormat, VK_COMPONENT_SWIZZLE_B);
        case 7: return Graphics::TextureComponentChannel(guestFormat, VK_COMPONENT_SWIZZLE_A);
        default: return VK_COMPONENT_SWIZZLE_ZERO;
    }
}

// The memory one surface occupies; T#s that differ only in their view (mips, layers, swizzle, LOD
// clamp) share it.
struct SurfaceKey {
    std::uint64_t address;
    std::uint32_t layout, width, height, layers, mips, tile, dimension;   // layout: element bytes, | 0x100 block compressed
    auto operator<=>(const SurfaceKey&) const = default;
};

struct Surface {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    GuestTextureResource resource{};
    Graphics::SurfaceGeometry geometry;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkFormat storageFormat = VK_FORMAT_UNDEFINED;
    bool uploaded = false;
    bool target = false;
    std::uint64_t drawn = 0;          // serial of the last draw into it (render targets)
    std::map<std::array<std::uint32_t, 6>, VkImageView> attachments;
    std::uint64_t fingerprint = 0;   // of the game's bytes at the last upload
    std::uint32_t storage = 0;       // first storage heap index (one per level), 0: none
};

// A cheap fingerprint of a surface's bytes in the game's memory: 64 spread 8-byte samples plus the
// first and last 64 bytes. GT7 initializes many textures every frame over unchanged memory; only
// new bytes are a new upload.
std::uint64_t Fingerprint(std::uint64_t address, std::uint64_t bytes) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t at) {
        std::uint64_t value;
        std::memcpy(&value, reinterpret_cast<const void*>(at), sizeof(value));
        hash = (hash ^ value) * 0x100000001b3ull;
    };
    if (bytes < 8) return hash;
    for (std::uint64_t at = 0; at < std::min<std::uint64_t>(bytes, 64) - 7; at += 8) mix(address + at);
    for (std::uint64_t at = bytes - std::min<std::uint64_t>(bytes, 64); at + 8 <= bytes; at += 8) mix(address + at);
    const auto stride = std::max<std::uint64_t>((bytes / 64) & ~std::uint64_t{7}, 8);
    for (std::uint64_t at = 0; at + 8 <= bytes; at += stride) mix(address + at);
    return hash;
}

struct WordsHash {
    std::size_t operator()(const std::array<std::uint32_t, 8>& words) const noexcept { return Heap::Hash(words.data(), 8); }
};

}

struct NativeImages::State {
    Graphics::Context context;
    Locate locate;
    VkDevice vk = VK_NULL_HANDLE;

    std::unique_ptr<Graphics::Buffer> imageTable, samplerTable;
    // Recorded shader data keeps its table address until the batch retires.
    std::vector<std::unique_ptr<Graphics::Buffer>> retiredImageTables, retiredSamplerTables;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;

    std::map<SurfaceKey, Surface> surfaces;
    std::unordered_map<std::array<std::uint32_t, 8>, std::uint32_t, WordsHash> images;   // T# -> sampled heap index
    // Depth buffers: one depth/stencil image per depth surface; T#s GT7 makes over their memory
    // sample them (depth aspect).
    struct Depth {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        VkImageView attachment = VK_NULL_HANDLE;
        std::uint32_t layers = 1;
    };
    std::map<std::uint64_t, Depth> depths;
    struct Registered {
        std::uint32_t index;
        GuestTextureResource resource;
    };
    std::unordered_map<std::array<std::uint32_t, 8>, Registered, WordsHash> registered;
    std::map<std::array<std::uint32_t, 4>, std::uint32_t> samplers;
    std::vector<std::array<std::uint32_t, 8>> textureQueue;
    std::vector<std::array<std::uint32_t, 4>> samplerQueue;
    std::set<std::uint64_t> targets;
    std::set<std::array<std::uint32_t, 8>> failedImages;
    std::set<std::array<std::uint32_t, 4>> failedSamplers;
    std::uint32_t nextSampled = 1, nextStorage = 1, nextSampler = 2;
    std::vector<VkImageView> views;
    std::vector<VkSampler> samplerObjects;
    std::vector<std::shared_ptr<Graphics::DeviceBuffer>> staging;
    std::vector<VkImage> nullImages;
    // The null view of every image binding (index 0's), written at a new heap index before its real
    // views so no binding a shader may select there is left unwritten.
    std::map<std::uint32_t, std::pair<VkImageView, bool>> nullViews;
    std::vector<VkDeviceMemory> nullMemory;
    std::size_t uploads = 0, missedImages = 0, missedSamplers = 0;

    template<typename T> T fn(const char* name) const { return context.Function<T>(name); }

    // ---- tables --------------------------------------------------------------------------------

    std::unique_ptr<Graphics::Buffer> makeTable(std::uint32_t slots, std::uint32_t entryBytes, std::uint32_t misses, std::uint32_t missBytes) {
        const auto missOffset = Heap::HeaderBytes + slots * entryBytes;
        auto table = std::make_unique<Graphics::Buffer>(context, missOffset + misses * missBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        auto bytes = table->Bytes();
        std::memset(bytes.data(), 0, bytes.size());
        auto* header = reinterpret_cast<std::uint32_t*>(bytes.data());
        header[0] = slots - 1u;
        header[1] = 0;
        header[2] = misses;
        header[3] = missOffset;
        return table;
    }

    // Inserts words -> payload; the payload first and the words last, so a shader probing
    // concurrently sees either the empty slot or the whole entry.
    template<std::size_t Count, typename Entry>
    bool insert(Graphics::Buffer& table, const std::array<std::uint32_t, Count>& words, const Entry& entry) {
        auto bytes = table.Bytes();
        const auto mask = reinterpret_cast<const std::uint32_t*>(bytes.data())[0];
        auto slot = Heap::Hash(words.data(), Count) & mask;
        for (std::uint32_t probe = 0; probe < Heap::ProbeLimit; ++probe, slot = (slot + 1u) & mask) {
            auto* stored = reinterpret_cast<Entry*>(bytes.data() + Heap::HeaderBytes + static_cast<std::size_t>(slot) * sizeof(Entry));
            if (stored->words[0] != 0 || stored->words[1] != 0) continue;
            Entry copy = entry;
            copy.words[0] = copy.words[1] = 0;
            std::memcpy(stored, &copy, sizeof(Entry));
            std::atomic_thread_fence(std::memory_order_release);
            std::memcpy(stored->words, words.data(), 8);
            return true;
        }
        return false;
    }

    // Keep every entry inside the shader's bounded probe window. A collision
    // cluster is not a full heap: rehash into a larger table rather than silently
    // marking an unreachable descriptor as registered.
    template<std::size_t Count, typename Entry>
    void insertGrowing(std::unique_ptr<Graphics::Buffer>& table,
                       std::vector<std::unique_ptr<Graphics::Buffer>>& retired,
                       const std::array<std::uint32_t, Count>& words, const Entry& entry) {
        if (insert(*table, words, entry)) return;
        const auto oldBytes = table->Bytes();
        const auto* header = reinterpret_cast<const std::uint32_t*>(oldBytes.data());
        const auto oldSlots = header[0] + 1u;
        const auto* oldEntries = reinterpret_cast<const Entry*>(oldBytes.data() + Heap::HeaderBytes);
        for (auto slots = oldSlots * 2u; slots <= (1u << 22u); slots *= 2u) {
            auto grown = makeTable(slots, sizeof(Entry), header[2], Count * sizeof(std::uint32_t));
            bool fits = true;
            for (std::uint32_t i = 0; i < oldSlots; ++i) {
                if (oldEntries[i].words[0] == 0 && oldEntries[i].words[1] == 0) continue;
                std::array<std::uint32_t, Count> key;
                std::memcpy(key.data(), oldEntries[i].words, sizeof(key));
                if (!insert(*grown, key, oldEntries[i])) { fits = false; break; }
            }
            if (!fits || !insert(*grown, words, entry)) continue;
            retired.push_back(std::move(table));
            table = std::move(grown);
            std::fprintf(stderr, "[native] descriptor table grew from %u to %u slots\n", oldSlots, slots);
            return;
        }
        throw std::runtime_error("native descriptor table exceeds its collision budget");
    }

    template<std::size_t Count>
    void takeMisses(Graphics::Buffer& table, std::vector<std::array<std::uint32_t, Count>>& out, std::size_t& counter) {
        auto bytes = table.Bytes();
        auto* header = reinterpret_cast<std::uint32_t*>(bytes.data());
        const auto count = std::min(header[1], header[2]);
        std::set<std::array<std::uint32_t, Count>> unique;
        for (std::uint32_t i = 0; i < count; ++i) {
            std::array<std::uint32_t, Count> words;
            std::memcpy(words.data(), bytes.data() + header[3] + static_cast<std::size_t>(i) * Count * 4u, Count * 4u);
            unique.insert(words);
        }
        header[1] = 0;
        for (const auto& words : unique) out.push_back(words), ++counter;
    }

    // ---- heap ----------------------------------------------------------------------------------

    void makeHeap() {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<VkDescriptorBindingFlags> flags;
        for (std::uint32_t kind = ShaderRecompiler::RuntimeAbi::FirstImageBinding; kind <= SamplerBinding; ++kind) {
            const bool sampler = kind == SamplerBinding;
            const bool storage = !sampler && kind >= StorageFloatBase;
            bindings.push_back({kind, sampler ? VK_DESCRIPTOR_TYPE_SAMPLER : storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampler ? Heap::SamplerCapacity : Heap::ImageCapacity, VK_SHADER_STAGE_ALL, nullptr});
            flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
        }
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flagInfo.bindingCount = static_cast<std::uint32_t>(flags.size());
        flagInfo.pBindingFlags = flags.data();
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &flagInfo};
        layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(vk, &layoutInfo, nullptr, &layout) != VK_SUCCESS) throw std::runtime_error("native images: heap layout");
        const std::uint32_t imageKinds = ShaderRecompiler::RuntimeAbi::ImageBindingCount / 2u;
        const std::array<VkDescriptorPoolSize, 3> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, imageKinds * Heap::ImageCapacity}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, imageKinds * Heap::ImageCapacity}, {VK_DESCRIPTOR_TYPE_SAMPLER, Heap::SamplerCapacity}}};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        poolInfo.pPoolSizes = sizes.data();
        if (fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(vk, &poolInfo, nullptr, &pool) != VK_SUCCESS) throw std::runtime_error("native images: heap pool");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &layout;
        if (fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(vk, &allocation, &set) != VK_SUCCESS) throw std::runtime_error("native images: heap set");
    }

    void clearIndex(std::uint32_t index, bool storage) {
        for (const auto& [binding, entry] : nullViews) {
            if (entry.second == storage) write(binding, index, entry.first, storage);
        }
    }

    void write(std::uint32_t binding, std::uint32_t index, VkImageView view, bool storage) {
        VkDescriptorImageInfo info{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = index;
        write.descriptorCount = 1;
        write.descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &info;
        fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(vk, 1, &write, 0, nullptr);
    }

    void writeSampler(std::uint32_t index, VkSampler sampler) {
        VkDescriptorImageInfo info{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = SamplerBinding;
        write.dstArrayElement = index;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &info;
        fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(vk, 1, &write, 0, nullptr);
    }

    VkImageView view(VkImage image, VkImageViewType type, VkFormat format, VkComponentMapping components, VkImageSubresourceRange range) {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image;
        info.viewType = type;
        info.format = format;
        info.components = components;
        info.subresourceRange = range;
        VkImageView made = VK_NULL_HANDLE;
        if (fn<PFN_vkCreateImageView>("vkCreateImageView")(vk, &info, nullptr, &made) != VK_SUCCESS) throw std::runtime_error("native images: vkCreateImageView");
        views.push_back(made);
        return made;
    }

    std::pair<VkImage, VkDeviceMemory> image(const VkImageCreateInfo& info) {
        VkImage made = VK_NULL_HANDLE;
        if (fn<PFN_vkCreateImage>("vkCreateImage")(vk, &info, nullptr, &made) != VK_SUCCESS) throw std::runtime_error("native images: vkCreateImage");
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(vk, made, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.DeviceMemoryType(requirements.memoryTypeBits, requirements.size, "native image");
        // NVIDIA's driver fails the process's first large device-local allocation in this process
        // (VK_ERROR_OUT_OF_DEVICE_MEMORY with the heap nearly empty) until a small allocation of the
        // plain system-memory type has been made; the same request then succeeds.
        VkDeviceMemory memory = VK_NULL_HANDLE;
        auto result = fn<PFN_vkAllocateMemory>("vkAllocateMemory")(vk, &allocation, nullptr, &memory);
        if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            for (std::uint32_t type = 0; type < context.memory.memoryTypeCount && result != VK_SUCCESS; ++type) {
                if (context.memory.memoryTypes[type].propertyFlags != 0) continue;
                VkMemoryAllocateInfo small{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                small.allocationSize = 65536;
                small.memoryTypeIndex = type;
                VkDeviceMemory scratch = VK_NULL_HANDLE;
                if (fn<PFN_vkAllocateMemory>("vkAllocateMemory")(vk, &small, nullptr, &scratch) == VK_SUCCESS) fn<PFN_vkFreeMemory>("vkFreeMemory")(vk, scratch, nullptr);
                result = fn<PFN_vkAllocateMemory>("vkAllocateMemory")(vk, &allocation, nullptr, &memory);
            }
        }
        if (result != VK_SUCCESS) {
            fn<PFN_vkDestroyImage>("vkDestroyImage")(vk, made, nullptr);
            throw std::runtime_error("native images: image memory (" + std::to_string(static_cast<int>(result)) + ", " + std::to_string(requirements.size) + " bytes, type " + std::to_string(allocation.memoryTypeIndex) + ")");
        }
        fn<PFN_vkBindImageMemory>("vkBindImageMemory")(vk, made, memory, 0);
        return {made, memory};
    }

    void toGeneral(VkCommandBuffer commands, VkImage target, VkImageLayout from, VkAccessFlags source, VkAccessFlags destination, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT, VkImageLayout to = VK_IMAGE_LAYOUT_GENERAL) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = source;
        barrier.dstAccessMask = destination;
        barrier.oldLayout = from;
        barrier.newLayout = to;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = target;
        barrier.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    // Index 0 of every binding: a 1x1 zero image of the binding's numeric class and dimension, so a
    // lookup miss samples zero (as an unbound PS5 descriptor reads) instead of faulting.
    void makeNullDescriptors(VkCommandBuffer commands) {
        const std::array<VkFormat, 3> formats{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT};
        for (std::uint32_t numeric = 0; numeric < 3; ++numeric) {
            for (const auto type : {VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D}) {
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = type;
                info.format = formats[numeric];
                info.extent = {1, 1, 1};
                info.mipLevels = 1;
                info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                const auto [made, memory] = image(info);
                nullImages.push_back(made);
                nullMemory.push_back(memory);
                toGeneral(commands, made, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                const VkClearColorValue zero{};
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, made, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
                const VkComponentMapping identity{};
                const auto slots = type == VK_IMAGE_TYPE_1D ? std::vector<std::pair<std::uint32_t, VkImageViewType>>{{Slot1D, VK_IMAGE_VIEW_TYPE_1D}, {Slot1DArray, VK_IMAGE_VIEW_TYPE_1D_ARRAY}}
                                 : type == VK_IMAGE_TYPE_2D ? std::vector<std::pair<std::uint32_t, VkImageViewType>>{{Slot2D, VK_IMAGE_VIEW_TYPE_2D}, {Slot2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY}}
                                                            : std::vector<std::pair<std::uint32_t, VkImageViewType>>{{Slot3D, VK_IMAGE_VIEW_TYPE_3D}};
                for (const auto& [slot, viewType] : slots) {
                    const auto made = view(nullImages.back(), viewType, formats[numeric], identity, range);
                    const auto record = [&](std::uint32_t binding, bool storage) {
                        write(binding, 0, made, storage);
                        nullViews[binding] = {made, storage};
                    };
                    record(SampledBase[numeric] + slot, false);
                    if (numeric == 0) record(ShaderRecompiler::RuntimeAbi::FirstComparisonImageBinding + slot, false), record(StorageFloatBase + slot, true);
                    if (numeric == 1) record(StorageUintBase + slot, true), record(AtomicUintBase + slot, true);
                }
            }
        }
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        VkSampler sampler = VK_NULL_HANDLE;
        fn<PFN_vkCreateSampler>("vkCreateSampler")(vk, &samplerInfo, nullptr, &sampler);
        samplerObjects.push_back(sampler);
        writeSampler(0, sampler);
        writeSampler(1, sampler);
    }

    // ---- images --------------------------------------------------------------------------------

    Surface& surfaceFor(const GuestTextureResource& resource, VkCommandBuffer commands) {
        const auto geometry = Graphics::DescribeSurface(resource);
        const SurfaceKey key{resource.baseAddress, Graphics::BytesPerElement(resource.format) | (Graphics::IsBlockCompressed(resource.format) ? 0x100u : 0u), resource.width, resource.height, geometry.imageLayers * geometry.imageDepth, resource.mipCount, static_cast<std::uint32_t>(resource.tileMode), static_cast<std::uint32_t>(resource.dimension)};
        if (const auto found = surfaces.find(key); found != surfaces.end()) return found->second;
        Surface surface;
        surface.resource = resource;
        surface.geometry = geometry;
        surface.format = Graphics::SampledTextureFormat(context, resource.format);
        const bool compressed = Graphics::IsBlockCompressed(resource.format);
        if (!compressed && Graphics::StorageFormatAvailable(context, resource.format)) surface.storageFormat = Graphics::StorageFormatForGuest(context, resource.format);
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        // Views of other formats of the same element size (a render target sampled as sRGB, a texture
        // written as uint) share the surface: mutable format, usages any compatible format supports.
        info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (resource.dimension == TextureDimension::kCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u) | (compressed ? 0u : VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
        info.imageType = resource.dimension == TextureDimension::k1D || resource.dimension == TextureDimension::k1DArray ? VK_IMAGE_TYPE_1D : resource.dimension == TextureDimension::k3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        info.format = surface.format;
        info.extent = {resource.width, resource.height, geometry.imageDepth};
        info.mipLevels = resource.mipCount;
        info.arrayLayers = geometry.imageLayers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        const auto elementBytes = Graphics::BytesPerElement(resource.format);
        const bool attachable = !compressed && resource.dimension != TextureDimension::k3D && (elementBytes == 1 || elementBytes == 2 || elementBytes == 4 || elementBytes == 8 || elementBytes == 16);
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (surface.storageFormat != VK_FORMAT_UNDEFINED ? VK_IMAGE_USAGE_STORAGE_BIT : 0u) | (attachable ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0u);
        const auto [made, memory] = image(info);
        surface.image = made;
        surface.memory = memory;
        toGeneral(commands, made, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        // Storage views, one per level, at consecutive heap indices.
        if (surface.storageFormat != VK_FORMAT_UNDEFINED && nextStorage + resource.mipCount <= Heap::ImageCapacity) {
            surface.storage = nextStorage;
            nextStorage += resource.mipCount;
            for (std::uint32_t level = 0; level < resource.mipCount; ++level) clearIndex(surface.storage + level, true);
            const auto numeric = NumericOf(surface.storageFormat);
            for (std::uint32_t level = 0; level < resource.mipCount; ++level) {
                for (const auto& [slot, type] : slotsOf(resource.dimension, geometry.imageLayers)) {
                    const auto made = view(surface.image, type, surface.storageFormat, {}, {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_3D ? 1u : geometry.imageLayers});
                    if (numeric == Numeric::Float) write(StorageFloatBase + slot, surface.storage + level, made, true);
                    if (numeric == Numeric::Uint) write(StorageUintBase + slot, surface.storage + level, made, true), write(AtomicUintBase + slot, surface.storage + level, made, true);
                }
            }
        }
        return surfaces.emplace(key, surface).first->second;
    }

    static std::vector<std::pair<std::uint32_t, VkImageViewType>> slotsOf(TextureDimension dimension, std::uint32_t layers) {
        switch (dimension) {
            case TextureDimension::k1D: return {{Slot1D, VK_IMAGE_VIEW_TYPE_1D}, {Slot1DArray, VK_IMAGE_VIEW_TYPE_1D_ARRAY}};
            case TextureDimension::k1DArray: return {{Slot1DArray, VK_IMAGE_VIEW_TYPE_1D_ARRAY}};
            case TextureDimension::k2D: return {{Slot2D, VK_IMAGE_VIEW_TYPE_2D}, {Slot2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY}};
            case TextureDimension::k2DArray:
                if (layers == 1) return {{Slot2D, VK_IMAGE_VIEW_TYPE_2D}, {Slot2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY}};
                return {{Slot2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY}};
            case TextureDimension::kCube: return {{Slot2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY}};
            case TextureDimension::k3D: return {{Slot3D, VK_IMAGE_VIEW_TYPE_3D}};
        }
        return {};
    }

    // Detiles the surface from the game's memory: tiled bytes -> linear staging -> image.
    void upload(VkCommandBuffer commands, Surface& surface) {
        surface.uploaded = true;
        const auto& resource = surface.resource;
        const auto& geometry = surface.geometry;
        VkBuffer source = VK_NULL_HANDLE;
        VkDeviceSize sourceOffset = 0;
        if (geometry.guestBytes == 0 || !locate(resource.baseAddress, geometry.guestBytes, source, sourceOffset)) return;
        surface.fingerprint = Fingerprint(resource.baseAddress, geometry.guestBytes);
        const auto elementBytes = Graphics::BytesPerElement(resource.format);
        auto linear = std::make_shared<Graphics::DeviceBuffer>(context, static_cast<std::size_t>(geometry.sliceLinearBytes * geometry.layers), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        staging.push_back(linear);
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        std::vector<VkBufferImageCopy> regions;
        for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) {
            for (std::uint32_t level = 0; level < geometry.mips.size(); ++level) {
                if (!geometry.HasLayer(level, layer)) continue;
                const auto& mip = geometry.mips[level];
                context.detiler->Dispatch(commands, resource.tileMode, elementBytes, source, sourceOffset + geometry.GuestLayerOffset(layer) + mip.tiledOffset, linear->Handle(), geometry.LinearLayerOffset(layer) + mip.linearOffset, mip, false, layer, geometry.thick, {.pipeBankXor = resource.pipeBankXor});
                VkBufferImageCopy region{};
                region.bufferOffset = geometry.LinearLayerOffset(layer) + mip.linearOffset;
                region.bufferRowLength = mip.pitchBytes / elementBytes * Graphics::BlockWidth(resource.format);
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, geometry.CopyLayer(layer), 1};
                region.imageOffset = {0, 0, geometry.CopyDepth(layer)};
                region.imageExtent = {std::max(resource.width >> level, 1u), std::max(resource.height >> level, 1u), 1u};
                regions.push_back(region);
            }
        }
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), surface.image, VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        ++uploads;
    }

    // A T#'s sampled views and table entry; the surface is made (and uploaded, unless it is a render
    // target's memory) on first use.
    void registerImage(VkCommandBuffer commands, const std::array<std::uint32_t, 8>& words, bool initialized) {
        if (words[0] == 0 && words[1] == 0) return;
        if ((words[0] | (words[1] & 0xffu)) == 0) return;   // null base: an unbound descriptor reads the null image
        if (const auto found = images.find(words); found != images.end()) {
            // The game initialized the texture again: new content in the same memory, unless the
            // GPU owns that memory.
            if (initialized) {
                const auto resource = Graphics::DecodeTextureResource(words);
                auto& surface = surfaceFor(resource, commands);
                if (!surface.target && targets.count(resource.baseAddress) == 0 && (surface.fingerprint == 0 || Fingerprint(resource.baseAddress, surface.geometry.guestBytes) != surface.fingerprint)) upload(commands, surface);
            }
            return;
        }
        if (failedImages.count(words) != 0 || nextSampled >= Heap::ImageCapacity) return;
        try {
            const auto resource = Graphics::DecodeTextureResource(words);
            if (resource.width == 0 || resource.height == 0 || resource.mipCount == 0) throw std::runtime_error("empty surface");
            auto& surface = surfaceFor(resource, commands);
            surface.target = surface.target || targets.count(resource.baseAddress) != 0;
            if (!surface.uploaded && !surface.target) upload(commands, surface);
            const auto index = nextSampled++;
            clearIndex(index, false);
            const auto numeric = NumericOf(surface.format);
            const VkComponentMapping components = Graphics::IsConvertedTextureFormat(resource.format) ? VkComponentMapping{} : VkComponentMapping{Channel(resource.format, resource.dstSelX), Channel(resource.format, resource.dstSelY), Channel(resource.format, resource.dstSelZ), Channel(resource.format, resource.dstSelW)};
            const auto levels = std::min(resource.lastLevel, resource.mipCount - 1u) - std::min(resource.baseLevel, resource.mipCount - 1u) + 1u;
            const auto baseLayer = std::min(resource.baseArray, surface.geometry.imageLayers - 1u);
            const auto layers = surface.geometry.imageLayers - baseLayer;
            for (const auto& [slot, type] : slotsOf(resource.dimension, layers)) {
                const bool single = type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_3D;
                const auto made = view(surface.image, type, surface.format, components, {VK_IMAGE_ASPECT_COLOR_BIT, std::min(resource.baseLevel, resource.mipCount - 1u), levels, baseLayer, single ? 1u : layers});
                write(SampledBase[static_cast<std::uint32_t>(numeric)] + slot, index, made, false);
            }
            Heap::ImageEntry entry{};
            std::memcpy(entry.words, words.data(), sizeof(entry.words));
            entry.sampled = index;
            entry.storage = surface.storage == 0 ? 0u : surface.storage + std::min(resource.baseLevel, resource.mipCount - 1u);
            if (const auto depth = depths.find(resource.baseAddress); depth != depths.end()) pointAtDepth(index, resource, depth->second);
            insertGrowing(imageTable, retiredImageTables, words, entry);
            images.emplace(words, index);
            registered[words] = {index, resource};
        } catch (const std::exception& error) {
            failedImages.insert(words);
            if (failedImages.size() <= 32) std::fprintf(stderr, "[native] T# %08x %08x %08x %08x %08x %08x %08x %08x not native: %s\n", words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7], error.what());
        }
    }

    std::uint64_t drawSerial = 0;

    // A color target's attachment view: the surface of its memory (shared with the textures GT7
    // makes over the same memory), never uploaded from memory again.
    VkImageView targetView(VkCommandBuffer commands, const Graphics::ColorTarget& color, VkImage& imageOut) {
        GuestTextureResource resource{};
        resource.baseAddress = color.surfaceAddress != 0 ? color.surfaceAddress : color.address;
        const auto extent = color.surfaceExtent.width != 0 ? color.surfaceExtent : color.extent;
        resource.width = extent.width;
        resource.height = extent.height;
        resource.depthOrLastArray = color.arrayLastSlice;
        resource.mipCount = std::max(color.mipCount, 1u);
        resource.lastLevel = resource.mipCount - 1u;
        resource.tileMode = TileOf(color.tileMode);
        resource.dimension = color.depth > 1 ? TextureDimension::k3D : color.arrayLastSlice > 0 ? TextureDimension::k2DArray : TextureDimension::k2D;
        if (color.depth > 1) resource.depthOrLastArray = color.depth - 1u;
        const auto guest = color.guestFormat != 0 ? std::optional<std::uint32_t>(color.guestFormat) : Graphics::FindGuestTextureFormat(color.format, color.elementBytes);
        if (!guest) throw std::runtime_error("color target format " + std::to_string(static_cast<int>(color.format)) + " has no guest texture format");
        resource.format = *guest;
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        resource.pipeBankXor = color.pipeBankXor;
        if (targets.insert(resource.baseAddress).second && targets.size() <= 64) std::fprintf(stderr, "[native] render target 0x%llx %ux%u VkFormat %d tile %u mips %u\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<int>(color.format), static_cast<unsigned>(color.tileMode), resource.mipCount);
        auto& surface = surfaceFor(resource, commands);
        if (!surface.uploaded && !surface.target) {
            // A target over memory no texture filled starts cleared, as GT7's fresh render targets do.
            // Debug aid: APS5_NATIVE_TARGET_CLEAR=1 starts targets magenta (shows what draws cover).
            static const bool debugClear = std::getenv("APS5_NATIVE_TARGET_CLEAR") != nullptr;
            VkClearColorValue zero{};
            if (debugClear) zero.float32[0] = zero.float32[2] = zero.float32[3] = 1.0f;
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, surface.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
            Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        }
        surface.target = true;
        surface.uploaded = true;
        surface.drawn = ++drawSerial;
        imageOut = surface.image;
        const std::array<std::uint32_t, 6> key{static_cast<std::uint32_t>(color.format), color.mip, color.arraySlice, std::max(color.layers, 1u), color.depthSlice, 0};
        if (const auto found = surface.attachments.find(key); found != surface.attachments.end()) return found->second;
        const bool volume = resource.dimension == TextureDimension::k3D;
        const auto layers = std::max(color.layers, 1u);
        const auto made = view(surface.image, layers > 1 || volume ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, color.format, {}, {VK_IMAGE_ASPECT_COLOR_BIT, color.mip, 1, volume ? 0u : color.arraySlice, volume ? 1u : layers});
        surface.attachments.emplace(key, made);
        return made;
    }

    // A T#'s sampled slots made to read a depth buffer (the depth plane as R).
    void pointAtDepth(std::uint32_t index, const GuestTextureResource& resource, Depth& depth) {
        const auto baseLayer = resource.baseArray;
        if (baseLayer >= depth.layers) throw std::runtime_error("native depth sampled view starts outside its array");
        const auto sampled = view(depth.image, VK_IMAGE_VIEW_TYPE_2D, depth.format, {}, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, baseLayer, 1});
        write(SampledBase[0] + Slot2D, index, sampled, false);
        write(ShaderRecompiler::RuntimeAbi::FirstComparisonImageBinding + Slot2D, index, sampled, false);
        const auto lastLayer = std::min(resource.depthOrLastArray, depth.layers - 1u);
        if (lastLayer < baseLayer) throw std::runtime_error("native depth sampled view has a reversed slice range");
        const auto arrayView = view(depth.image, VK_IMAGE_VIEW_TYPE_2D_ARRAY, depth.format, {}, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, baseLayer, lastLayer - baseLayer + 1u});
        write(SampledBase[0] + Slot2DArray, index, arrayView, false);
        write(ShaderRecompiler::RuntimeAbi::FirstComparisonImageBinding + Slot2DArray, index, arrayView, false);
    }

    VkImageView depthView(VkCommandBuffer commands, const Graphics::DepthTarget& target) {
        if (target.lastSlice < target.firstSlice) throw std::runtime_error("native depth attachment has a reversed slice range");
        const auto layers = target.lastSlice - target.firstSlice + 1u;
        const auto address = target.address != 0 ? target.address : target.stencilAddress;
        auto found = depths.find(address);
        if (found != depths.end() && (found->second.format != target.format || found->second.extent.width != target.extent.width || found->second.extent.height != target.extent.height || found->second.layers != layers)) {
            // GT7 reused the memory for a depth buffer of another shape: a new image (the old one
            // is kept; recorded work may still use it).
            depths.erase(found);
            found = depths.end();
        }
        if (found != depths.end()) return found->second.attachment;
        if (target.samples != 1) throw std::runtime_error("multisampled depth is not native yet");
        Depth depth;
        depth.format = target.format;
        depth.extent = target.extent;
        depth.layers = layers;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = target.format;
        info.extent = {target.extent.width, target.extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        const auto [made, memory] = image(info);
        depth.image = made;
        depth.memory = memory;
        const bool stencil = target.format == VK_FORMAT_D32_SFLOAT_S8_UINT || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        toGeneral(commands, made, VK_IMAGE_LAYOUT_UNDEFINED, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, aspects);
        // A new depth buffer starts at GT7's clear values (what its HTILE says before the first clear).
        const VkClearDepthStencilValue clear{target.clearDepth, target.clearStencil};
        const VkImageSubresourceRange range{aspects, 0, 1, 0, layers};
        fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, made, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
        depth.attachment = view(made, layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, target.format, {}, range);
        auto& stored = depths.emplace(address, depth).first->second;
        // T#s GT7 already made over this memory now read the depth buffer.
        for (const auto& [words, record] : registered) {
            if (record.resource.baseAddress == address) pointAtDepth(record.index, record.resource, stored);
        }
        return stored.attachment;
    }

    static Graphics::TextureTileMode TileOf(Graphics::ColorTileMode mode) {
        using C = Graphics::ColorTileMode;
        using T = Graphics::TextureTileMode;
        switch (mode) {
            case C::Linear: return T::kLinear;
            case C::Standard4KB: return T::kStandard4KB;
            case C::Standard64KB: return T::kStandard64KB;
            case C::S4KBX: return T::kS4KBX;
            case C::D4KBX: return T::kD4KBX;
            case C::D64KBX: return T::kD64KBX;
            case C::RenderTarget: return T::kR64KBX;
        }
        return T::kR64KBX;
    }

    void registerSampler(const std::array<std::uint32_t, 4>& words) {
        static int traced = 0;
        if (traced < 12) {
            ++traced;
            std::fprintf(stderr, "[native] sampler miss %08x %08x %08x %08x (%s)\n", words[0], words[1], words[2], words[3], samplers.count(words) != 0 ? "registered" : "new");
        }
        if (samplers.count(words) != 0 || failedSamplers.count(words) != 0 || nextSampler + 2u > Heap::SamplerCapacity) return;
        try {
            // Texel-coordinate S#s (FORCE_UNNORMALIZED): the translated shader normalizes their
            // coordinates itself, so the sampler is an ordinary normalized one.
            const auto decoded = Graphics::DecodeSamplerResource(words, (words[0] & 0x8000u) != 0u, false, false, true);
            const auto make = [&](bool point) {
                VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
                VkSamplerReductionModeCreateInfo reduction{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
                reduction.reductionMode = decoded.reductionMode;
                if (decoded.reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE) info.pNext = &reduction;
                info.magFilter = point ? VK_FILTER_NEAREST : decoded.magFilter;
                info.minFilter = point ? VK_FILTER_NEAREST : decoded.minFilter;
                info.mipmapMode = point ? VK_SAMPLER_MIPMAP_MODE_NEAREST : decoded.mipmapMode;
                info.addressModeU = decoded.addressModeU;
                info.addressModeV = decoded.addressModeV;
                info.addressModeW = decoded.addressModeW;
                info.mipLodBias = std::clamp(decoded.lodBias, -context.limits.maxSamplerLodBias, context.limits.maxSamplerLodBias);
                info.anisotropyEnable = !point && decoded.anisotropyEnable && context.samplerAnisotropy ? VK_TRUE : VK_FALSE;
                info.maxAnisotropy = std::min(decoded.maxAnisotropy, context.limits.maxSamplerAnisotropy);
                info.compareEnable = decoded.compareEnable ? VK_TRUE : VK_FALSE;
                info.compareOp = decoded.compareOp;
                info.minLod = decoded.minLod;
                info.maxLod = decoded.maxLod;
                info.borderColor = decoded.borderColor;
                info.unnormalizedCoordinates = VK_FALSE;
                VkSampler made = VK_NULL_HANDLE;
                if (fn<PFN_vkCreateSampler>("vkCreateSampler")(vk, &info, nullptr, &made) != VK_SUCCESS) throw std::runtime_error("vkCreateSampler");
                samplerObjects.push_back(made);
                return made;
            };
            const auto index = nextSampler;
            nextSampler += 2;
            writeSampler(index, make(false));
            writeSampler(index + 1, make(true));
            Heap::SamplerEntry entry{};
            std::memcpy(entry.words, words.data(), sizeof(entry.words));
            entry.sampler = index;
            insertGrowing(samplerTable, retiredSamplerTables, words, entry);
            samplers.emplace(words, index);
        } catch (const std::exception& error) {
            failedSamplers.insert(words);
            std::fprintf(stderr, "[native] S# %08x %08x %08x %08x not native: %s\n", words[0], words[1], words[2], words[3], error.what());
        }
    }
};

NativeImages::NativeImages(const Graphics::Context& context, Locate locate) : state(std::make_unique<State>()) {
    if (context.detiler == nullptr) throw std::runtime_error("native images: the device has no texture detiler");
    state->context = context;
    state->locate = std::move(locate);
    state->vk = context.device;
    state->imageTable = state->makeTable(ImageSlots, sizeof(Heap::ImageEntry), ImageMisses, 32u);
    state->samplerTable = state->makeTable(SamplerSlots, sizeof(Heap::SamplerEntry), SamplerMisses, 16u);
    state->makeHeap();
}

NativeImages::~NativeImages() = default;

std::uint64_t NativeImages::ImageTable() const { return state->imageTable->DeviceAddress(); }
std::uint64_t NativeImages::SamplerTable() const { return state->samplerTable->DeviceAddress(); }
VkDescriptorSetLayout NativeImages::HeapLayout() const { return state->layout; }
VkDescriptorSet NativeImages::HeapSet() const { return state->set; }

VkImageView NativeImages::DepthView(VkCommandBuffer commands, const Graphics::DepthTarget& depth) {
    return state->depthView(commands, depth);
}

VkImageView NativeImages::TargetView(VkCommandBuffer commands, const Graphics::ColorTarget& color, VkImage& image) {
    return state->targetView(commands, color, image);
}

bool NativeImages::Display(std::uint64_t address, VkImage& image, VkExtent2D& extent, VkFormat& format) const {
    const Surface* best = nullptr;
    for (auto it = state->surfaces.lower_bound(SurfaceKey{address, 0, 0, 0, 0, 0, 0, 0}); it != state->surfaces.end() && it->first.address == address; ++it) {
        if (it->second.drawn != 0 && (best == nullptr || it->second.drawn > best->drawn)) best = &it->second;
    }
    if (best == nullptr) return false;
    image = best->image;
    extent = {best->resource.width, best->resource.height};
    format = best->format;
    return true;
}

void NativeImages::Prepare(VkCommandBuffer commands) {
    auto& s = *state;
    if (s.nullImages.empty()) s.makeNullDescriptors(commands);
    {
        auto& noted = Pending();
        std::lock_guard lock(noted.mutex);
        for (auto& words : noted.textures) s.textureQueue.push_back(words);
        noted.textures.clear();
        s.targets.insert(noted.targets.begin(), noted.targets.end());
    }
    if (s.textureQueue.empty() && s.samplerQueue.empty()) return;
    s.context.detiler->BeginBatch();
    auto textures = std::move(s.textureQueue);
    s.textureQueue.clear();
    for (const auto& words : textures) s.registerImage(commands, words, true);
    auto samplers = std::move(s.samplerQueue);
    s.samplerQueue.clear();
    for (const auto& words : samplers) s.registerSampler(words);
}

void NativeImages::Completed() {
    auto& s = *state;
    s.staging.clear();
    std::vector<std::array<std::uint32_t, 8>> missed;
    s.takeMisses(*s.imageTable, missed, s.missedImages);
    for (auto& table : s.retiredImageTables) s.takeMisses(*table, missed, s.missedImages);
    s.retiredImageTables.clear();
    // A missed T# is registered as a view of an existing (or new) surface; it is not new content.
    for (const auto& words : missed) {
        if (s.images.count(words) == 0) s.textureQueue.push_back(words);
    }
    s.takeMisses(*s.samplerTable, s.samplerQueue, s.missedSamplers);
    for (auto& table : s.retiredSamplerTables) s.takeMisses(*table, s.samplerQueue, s.missedSamplers);
    s.retiredSamplerTables.clear();
    static int batches = 0;
    if (++batches % 600 == 1) {
        const auto* image = reinterpret_cast<const std::uint32_t*>(s.imageTable->Bytes().data());
        const auto* sampler = reinterpret_cast<const std::uint32_t*>(s.samplerTable->Bytes().data());
        std::fprintf(stderr, "[native] tables after batch %d: images mask %x misses %u/%u, samplers mask %x misses %u/%u; totals %zu/%zu\n", batches, image[0], image[1], image[2], sampler[0], sampler[1], sampler[2], s.missedImages, s.missedSamplers);
    }
    static std::size_t reported = 0;
    if (s.images.size() / 100 != reported) {
        reported = s.images.size() / 100;
        std::fprintf(stderr, "[native] images: %zu T#s over %zu surfaces (%zu uploads), %zu samplers; misses %zu images, %zu samplers; %zu failed\n", s.images.size(), s.surfaces.size(), s.uploads, s.samplers.size(), s.missedImages, s.missedSamplers, s.failedImages.size());
    }
}

void NoteTextureDescriptor(std::span<const std::uint32_t, 8> words) {
    auto& noted = Pending();
    std::array<std::uint32_t, 8> copy;
    std::memcpy(copy.data(), words.data(), sizeof(copy));
    std::lock_guard lock(noted.mutex);
    noted.textures.push_back(copy);
}

void NoteRenderTarget(std::uint64_t address, std::uint64_t) {
    auto& noted = Pending();
    std::lock_guard lock(noted.mutex);
    noted.targets.insert(address);
}

}
