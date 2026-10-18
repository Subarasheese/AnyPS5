#include "prx/libSceAgcDriver/Native/include/NativeImages.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <chrono>

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

#include <algorithm>
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
    std::vector<std::pair<std::uint64_t, std::uint64_t>> written;
    std::vector<std::array<std::uint32_t, 8>> sampled;
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
    bool stale = false;
    bool target = false;
    std::uint64_t drawn = 0;          // serial of the last draw into it (render targets)
    std::map<std::array<std::uint32_t, 6>, VkImageView> attachments;
    std::uint64_t fingerprint = 0;   // of the game's bytes at the last upload
    std::uint64_t generation = 0;
    std::uint64_t contents = 0;
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

std::uint64_t ContentHash(std::uint64_t address, std::uint64_t bytes) {
    std::uint64_t hash = 0x9e3779b97f4a7c15ull ^ bytes;
    const auto* data = reinterpret_cast<const std::byte*>(address);
    std::uint64_t at = 0;
    for (; at + 8 <= bytes; at += 8) {
        std::uint64_t value;
        std::memcpy(&value, data + at, sizeof(value));
        hash = (hash ^ value) * 0xff51afd7ed558ccdull;
        hash ^= hash >> 32u;
    }
    for (; at < bytes; ++at) hash = (hash ^ static_cast<std::uint8_t>(data[at])) * 0x100000001b3ull;
    return hash;
}

TextureDimension SurfaceDimension(TextureDimension dimension) {
    return dimension == TextureDimension::k2D || dimension == TextureDimension::kCube ? TextureDimension::k2DArray : dimension;
}

bool CubeCompatible(const GuestTextureResource& resource, const Graphics::SurfaceGeometry& geometry) {
    return SurfaceDimension(resource.dimension) == TextureDimension::k2DArray && resource.width == resource.height && geometry.imageLayers >= 6 && geometry.imageLayers % 6 == 0;
}

struct WordsHash {
    std::size_t operator()(const std::array<std::uint32_t, 8>& words) const noexcept { return Heap::Hash(words.data(), 8); }
};

}

struct NativeImages::State {
    std::uint64_t recordings = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writtenQueue;
    std::vector<std::array<std::uint32_t, 8>> sampledQueue;
    std::vector<Surface*> staleSampled;
    std::function<void()> beforeRecording;
    void recording() {
        if (beforeRecording) beforeRecording();
    }
    bool checkWrites = true;
    std::chrono::steady_clock::time_point checkedWrites{};
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
        std::uint32_t storage;
        GuestTextureResource resource;
        Surface* surface = nullptr;
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
        constexpr VkMemoryPropertyFlags vram = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const bool writableVram = std::any_of(context.memory.memoryTypes, context.memory.memoryTypes + context.memory.memoryTypeCount, [&](const VkMemoryType& type) { return (type.propertyFlags & vram) == vram; });
        const auto bytesNeeded = missOffset + misses * missBytes;
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        std::unique_ptr<Graphics::Buffer> table;
        if (writableVram) {
            try {
                table = std::make_unique<Graphics::Buffer>(context, bytesNeeded, usage, vram);
            } catch (const std::exception&) {
            }
        }
        if (table == nullptr) table = std::make_unique<Graphics::Buffer>(context, bytesNeeded, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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
            if (Native::Logs()) std::fprintf(stderr, "[native] descriptor table grew from %u to %u slots\n", oldSlots, slots);
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
        recording();
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
        recording();
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

    Surface& surfaceFor(const GuestTextureResource& resource, VkCommandBuffer commands, bool forTarget = false) {
        const auto geometry = Graphics::DescribeSurface(resource);
        const SurfaceKey key{resource.baseAddress, Graphics::BytesPerElement(resource.format) | (Graphics::IsBlockCompressed(resource.format) ? 0x100u : 0u), resource.width, resource.height, geometry.imageLayers * geometry.imageDepth, resource.mipCount, static_cast<std::uint32_t>(resource.tileMode), static_cast<std::uint32_t>(SurfaceDimension(resource.dimension))};
        if (const auto found = surfaces.find(key); found != surfaces.end()) return found->second;
        for (auto it = surfaces.lower_bound(SurfaceKey{resource.baseAddress, 0, 0, 0, 0, 0, 0, 0}); it != surfaces.end() && it->first.address == resource.baseAddress; ++it) {
            const auto& other = it->first;
            if (!forTarget && !it->second.target) continue;
            if (other.layout != key.layout || other.width != key.width || other.height != key.height || other.layers != key.layers || other.dimension != key.dimension || key.mips > other.mips) continue;
            return it->second;
        }
        for (auto it = surfaces.lower_bound(SurfaceKey{resource.baseAddress, 0, 0, 0, 0, 0, 0, 0}); it != surfaces.end() && it->first.address == resource.baseAddress; ++it) {
            const auto& other = it->first;
            if (other.layout != key.layout || other.width != key.width || other.height != key.height) continue;
            if (!forTarget && !it->second.target && !it->second.uploaded) continue;
            char detail[256];
            std::snprintf(detail, sizeof(detail), "0x%llx %ux%u: new %s layers %u mips %u tile %u dimension %u; existing %s layers %u mips %u tile %u dimension %u",
                static_cast<unsigned long long>(resource.baseAddress), key.width, key.height, forTarget ? "render target" : "texture", key.layers, key.mips, key.tile, key.dimension,
                it->second.target ? "render target" : "texture", other.layers, other.mips, other.tile, other.dimension);
            Native::ReportGraphicsIssue("same memory split into separate images", detail);
        }
        if (Native::Logs()) {
            static int reported = 0;
            const auto end = resource.baseAddress + geometry.guestBytes;
            for (const auto& [other, existing] : surfaces) {
                if (reported >= 400) break;
                const auto otherEnd = existing.resource.baseAddress + existing.geometry.guestBytes;
                if (existing.resource.baseAddress >= end || otherEnd <= resource.baseAddress) continue;
                ++reported;
                std::fprintf(stderr, "[native] surface overlap: new 0x%llx %ux%u mips %u layers %u fmt %u dim %d bytes 0x%llx; existing 0x%llx %ux%u mips %u layers %u fmt %u dim %d bytes 0x%llx%s\n",
                    static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.mipCount, geometry.imageLayers, resource.format, static_cast<int>(resource.dimension), static_cast<unsigned long long>(geometry.guestBytes),
                    static_cast<unsigned long long>(existing.resource.baseAddress), existing.resource.width, existing.resource.height, existing.resource.mipCount, existing.geometry.imageLayers, existing.resource.format, static_cast<int>(existing.resource.dimension), static_cast<unsigned long long>(existing.geometry.guestBytes), existing.target ? " target" : "");
            }
        }
        Surface surface;
        surface.resource = resource;
        surface.geometry = geometry;
        surface.format = Graphics::SampledTextureFormat(context, resource.format);
        const bool compressed = Graphics::IsBlockCompressed(resource.format);
        if (!compressed && Graphics::StorageFormatAvailable(context, resource.format)) surface.storageFormat = Graphics::StorageFormatForGuest(context, resource.format);
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        // Views of other formats of the same element size (a render target sampled as sRGB, a texture
        // written as uint) share the surface: mutable format, usages any compatible format supports.
        info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (CubeCompatible(resource, geometry) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u) | (compressed ? 0u : VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
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

    bool gather(VkCommandBuffer commands, std::uint64_t address, std::uint64_t bytes, VkBuffer& source, VkDeviceSize& sourceOffset) {
        recording();
        std::vector<std::pair<VkBuffer, VkBufferCopy>> pieces;
        for (std::uint64_t at = address; at < address + bytes;) {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
            const auto available = locate(at, buffer, offset);
            if (available == 0) {
                char detail[128];
                std::snprintf(detail, sizeof(detail), "texture 0x%llx (%llu bytes) at 0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(at));
                Native::ReportGraphicsIssue("texture memory not visible to the GPU (texture left empty)", detail);
                return false;
            }
            const auto take = std::min<std::uint64_t>(available, address + bytes - at);
            pieces.push_back({buffer, VkBufferCopy{offset, at - address, take}});
            at += take;
        }
        auto gathered = std::make_shared<Graphics::DeviceBuffer>(context, static_cast<std::size_t>(bytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        staging.push_back(gathered);
        Graphics::RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        for (const auto& [buffer, copy] : pieces) fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, buffer, gathered->Handle(), 1, &copy);
        source = gathered->Handle();
        sourceOffset = 0;
        if (Native::Logs()) std::fprintf(stderr, "[native] texture 0x%llx (%llu bytes) gathered from %zu GPU memory imports\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), pieces.size());
        return true;
    }

    // Detiles the surface from the game's memory: tiled bytes -> linear staging -> image.
    void upload(VkCommandBuffer commands, Surface& surface, bool gpuWritten = false) {
        surface.uploaded = true;
        NotePoolUse("texture", surface.resource.baseAddress);
        const auto& resource = surface.resource;
        const auto& geometry = surface.geometry;
        VkBuffer source = VK_NULL_HANDLE;
        VkDeviceSize sourceOffset = 0;
        if (geometry.guestBytes == 0) return;
        if (locate(resource.baseAddress, source, sourceOffset) < geometry.guestBytes && !gather(commands, resource.baseAddress, geometry.guestBytes, source, sourceOffset)) {
            surface.stale = true;
            return;
        }
        surface.stale = false;
        recording();
        surface.generation = GuestMemory::WriteWatched() ? GuestMemory::CollectWrites(resource.baseAddress, geometry.guestBytes) : 0;
        surface.contents = surface.generation != 0 && !gpuWritten ? ContentHash(resource.baseAddress, geometry.guestBytes) : 0;
        surface.fingerprint = gpuWritten ? 0 : Fingerprint(resource.baseAddress, geometry.guestBytes);
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
                if (!surface.target && targets.count(resource.baseAddress) == 0 && (surface.fingerprint == 0 || Fingerprint(resource.baseAddress, surface.geometry.guestBytes) != surface.fingerprint)) surface.stale = true;
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
            const auto requested = Graphics::IsConvertedTextureFormat(resource.format) ? surface.format : Graphics::SampledTextureFormat(context, resource.format);
            const auto viewFormat = requested != VK_FORMAT_UNDEFINED && Graphics::BytesPerElement(resource.format) == Graphics::BytesPerElement(surface.resource.format) && !Graphics::IsBlockCompressed(resource.format) ? requested : surface.format;
            const auto numeric = NumericOf(viewFormat);
            const VkComponentMapping components = Graphics::IsConvertedTextureFormat(resource.format) ? VkComponentMapping{} : VkComponentMapping{Channel(resource.format, resource.dstSelX), Channel(resource.format, resource.dstSelY), Channel(resource.format, resource.dstSelZ), Channel(resource.format, resource.dstSelW)};
            const auto levels = std::min(resource.lastLevel, resource.mipCount - 1u) - std::min(resource.baseLevel, resource.mipCount - 1u) + 1u;
            const auto baseLayer = std::min(resource.baseArray, surface.geometry.imageLayers - 1u);
            const auto layers = surface.geometry.imageLayers - baseLayer;
            for (const auto& [slot, type] : slotsOf(resource.dimension, layers)) {
                const bool single = type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_3D;
                const auto made = view(surface.image, type, viewFormat, components, {VK_IMAGE_ASPECT_COLOR_BIT, std::min(resource.baseLevel, resource.mipCount - 1u), levels, baseLayer, single ? 1u : layers});
                write(SampledBase[static_cast<std::uint32_t>(numeric)] + slot, index, made, false);
            }
            Heap::ImageEntry entry{};
            std::memcpy(entry.words, words.data(), sizeof(entry.words));
            entry.sampled = index;
            entry.storage = surface.storage == 0 ? 0u : surface.storage + std::min(resource.baseLevel, resource.mipCount - 1u);
            if (const auto depth = depths.find(resource.baseAddress); depth != depths.end()) pointAtDepth(index, resource, depth->second);
            insertGrowing(imageTable, retiredImageTables, words, entry);
            images.emplace(words, index);
            registered[words] = {index, entry.storage, resource, &surface};
        } catch (const std::exception& error) {
            failedImages.insert(words);
            Native::ReportGraphicsIssue("texture not created", error.what());
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
        if (targets.insert(resource.baseAddress).second && targets.size() <= 64 && Native::Logs()) std::fprintf(stderr, "[native] render target 0x%llx %ux%u VkFormat %d tile %u mips %u\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<int>(color.format), static_cast<unsigned>(color.tileMode), resource.mipCount);
        auto& surface = surfaceFor(resource, commands, true);
        if (!surface.uploaded && !surface.target) {
            // A target over memory no texture filled starts cleared, as GT7's fresh render targets do.
            // Debug aid: APS5_NATIVE_TARGET_CLEAR=1 starts targets magenta (shows what draws cover).
            static const bool debugClear = std::getenv("APS5_NATIVE_TARGET_CLEAR") != nullptr;
            VkClearColorValue zero{};
            if (debugClear) zero.float32[0] = zero.float32[2] = zero.float32[3] = 1.0f;
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            ++recordings;
            recording();
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
        ++recordings;
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
            if (Native::Logs()) std::fprintf(stderr, "[native] sampler miss %08x %08x %08x %08x (%s)\n", words[0], words[1], words[2], words[3], samplers.count(words) != 0 ? "registered" : "new");
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
            Native::ReportGraphicsIssue("sampler not created", error.what());
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

std::vector<NativeImages::DrawnTarget> NativeImages::DrawnTargets(std::uint64_t since) const {
    std::vector<DrawnTarget> drawn;
    for (const auto& [key, surface] : state->surfaces) {
        if (!surface.target || surface.drawn <= since || surface.image == VK_NULL_HANDLE) continue;
        drawn.push_back({surface.resource.baseAddress, surface.image, surface.format, surface.resource.width, surface.resource.height, surface.drawn});
    }
    return drawn;
}

std::vector<NativeImages::DrawnTarget> NativeImages::SurfacesAt(std::uint64_t address) const {
    std::vector<DrawnTarget> found;
    for (auto it = state->surfaces.lower_bound(SurfaceKey{address, 0, 0, 0, 0, 0, 0, 0}); it != state->surfaces.end() && it->first.address == address; ++it) {
        if (it->second.image != VK_NULL_HANDLE) found.push_back({address, it->second.image, it->second.format, it->second.resource.width, it->second.resource.height, it->second.drawn, it->second.geometry.guestBytes, it->second.resource.format, static_cast<std::uint32_t>(it->second.resource.tileMode)});
    }
    return found;
}

std::vector<std::shared_ptr<void>> NativeImages::TakeBatchResources() {
    auto& s = *state;
    std::vector<std::shared_ptr<void>> held(s.staging.begin(), s.staging.end());
    s.staging.clear();
    for (auto& table : s.retiredImageTables) held.push_back(std::shared_ptr<Graphics::Buffer>(std::move(table)));
    for (auto& table : s.retiredSamplerTables) held.push_back(std::shared_ptr<Graphics::Buffer>(std::move(table)));
    s.retiredImageTables.clear();
    s.retiredSamplerTables.clear();
    return held;
}

void NativeImages::BeforeRecording(std::function<void()> close) {
    state->beforeRecording = std::move(close);
}

void NativeImages::MarkGpuWritten(std::uint64_t begin, std::uint64_t end) {
    for (auto& [key, surface] : state->surfaces) {
        if (!surface.uploaded || surface.target) continue;
        const auto first = surface.resource.baseAddress;
        if (first >= end || first + surface.geometry.guestBytes <= begin) continue;
        surface.stale = true;
    }
}

std::string NativeImages::DescribeUploadedOverlap(std::uint64_t begin, std::uint64_t end) const {
    for (const auto& [key, surface] : state->surfaces) {
        if (!surface.uploaded || surface.target) continue;
        const auto first = surface.resource.baseAddress;
        const auto last = first + surface.geometry.guestBytes;
        if (first >= end || last <= begin) continue;
        char text[96];
        std::snprintf(text, sizeof(text), "0x%llx %ux%u format %u", static_cast<unsigned long long>(first), surface.resource.width, surface.resource.height, surface.resource.format);
        return text;
    }
    return {};
}

bool NativeImages::Registered(std::span<const std::uint32_t, 8> words) const {
    std::array<std::uint32_t, 8> copy;
    std::memcpy(copy.data(), words.data(), sizeof(copy));
    return state->images.count(copy) != 0;
}

bool NativeImages::ImageIndices(std::span<const std::uint32_t, 8> words, std::uint32_t& sampled, std::uint32_t& storage) {
    auto& s = *state;
    std::array<std::uint32_t, 8> copy;
    std::memcpy(copy.data(), words.data(), sizeof(copy));
    sampled = storage = 0;
    if (const auto found = s.registered.find(copy); found != s.registered.end()) {
        if (found->second.surface != nullptr && found->second.surface->stale) s.staleSampled.push_back(found->second.surface);
        sampled = found->second.index;
        storage = found->second.storage;
        return true;
    }
    if ((copy[0] | (copy[1] & 0xffu)) == 0 || s.failedImages.count(copy) != 0 || s.nextSampled >= Heap::ImageCapacity) return true;
    s.sampledQueue.push_back(copy);
    return false;
}

bool NativeImages::SamplerIndex(std::span<const std::uint32_t, 4> words, std::uint32_t& sampler) {
    auto& s = *state;
    std::array<std::uint32_t, 4> copy;
    std::memcpy(copy.data(), words.data(), sizeof(copy));
    sampler = 0;
    if (const auto found = s.samplers.find(copy); found != s.samplers.end()) {
        sampler = found->second;
        return true;
    }
    if (s.failedSamplers.count(copy) != 0 || s.nextSampler + 2u > Heap::SamplerCapacity) return true;
    s.samplerQueue.push_back(copy);
    return false;
}

std::uint64_t NativeImages::DrawSerial() const {
    return state->drawSerial;
}

void NativeImages::NewBatch() {
    state->checkWrites = true;
}

void NativeImages::Prepare(VkCommandBuffer commands) {
    auto& s = *state;
    if (s.checkWrites && GuestMemory::WriteWatched()) {
        s.checkWrites = false;
        struct ScanTime {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~ScanTime() {
                if (!Native::Logs()) return;
                static double ms = 0.0;
                static std::uint64_t scans = 0;
                static auto reported = std::chrono::steady_clock::now();
                const auto now = std::chrono::steady_clock::now();
                ms += std::chrono::duration<double, std::milli>(now - start).count();
                ++scans;
                if (now - reported < std::chrono::seconds(5)) return;
                std::fprintf(stderr, "[native] texture write scans: %llu in 5 s, %.0f ms\n", static_cast<unsigned long long>(scans), ms);
                ms = 0.0;
                scans = 0;
                reported = now;
            }
        } scanTime;
        {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
            for (const auto& [key, surface] : s.surfaces) {
                if (!surface.uploaded || surface.target || surface.generation == 0) continue;
                spans.emplace_back(surface.resource.baseAddress, surface.resource.baseAddress + surface.geometry.guestBytes);
            }
            std::sort(spans.begin(), spans.end());
            std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
            for (const auto& span : spans) {
                if (!merged.empty() && span.first <= merged.back().second + (16ull << 20u)) merged.back().second = std::max(merged.back().second, span.second);
                else merged.push_back(span);
            }
            for (const auto& [begin, end] : merged) static_cast<void>(GuestMemory::CollectWrites(begin, static_cast<std::size_t>(end - begin)));
            for (auto& [key, surface] : s.surfaces) {
                if (!surface.uploaded || surface.target || surface.generation == 0 || s.targets.count(surface.resource.baseAddress) != 0) continue;
                const auto address = surface.resource.baseAddress;
                const auto bytes = surface.geometry.guestBytes;
                if (GuestMemory::UnchangedSince(address, bytes, surface.generation)) continue;
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) {
                    surface.generation = 0;
                    continue;
                }
                // New content is uploaded when work samples the surface (a written ring of
                // transient memory holds many surfaces nothing samples again).
                surface.stale = true;
                if (Native::Logs()) {
                    static std::map<std::uint64_t, std::uint64_t> refreshed;
                    static std::map<std::uint64_t, std::string> shapes;
                    char shape[64];
                    std::snprintf(shape, sizeof(shape), "%ux%u fmt %u %llu B", surface.resource.width, surface.resource.height, surface.resource.format, static_cast<unsigned long long>(bytes));
                    shapes[address] = shape;
                    static auto reported = std::chrono::steady_clock::now();
                    ++refreshed[address];
                    if (std::chrono::steady_clock::now() - reported > std::chrono::seconds(5)) {
                        reported = std::chrono::steady_clock::now();
                        std::vector<std::pair<std::uint64_t, std::uint64_t>> top(refreshed.begin(), refreshed.end());
                        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                        std::string line;
                        for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 8); ++i) {
                            char entry[160];
                            std::snprintf(entry, sizeof(entry), " 0x%llx x%llu (%s)", static_cast<unsigned long long>(top[i].first), static_cast<unsigned long long>(top[i].second), shapes[top[i].first].c_str());
                            line += entry;
                        }
                        std::fprintf(stderr, "[native] written textures refreshed in 5 s: %zu surfaces;%s\n", refreshed.size(), line.c_str());
                        refreshed.clear();
                    }
                }
            }
        }
    }
    if (s.nullImages.empty()) {
        ++s.recordings;
        s.makeNullDescriptors(commands);
    }
    {
        auto& noted = Pending();
        std::lock_guard lock(noted.mutex);
        for (auto& words : noted.textures) s.textureQueue.push_back(words);
        noted.textures.clear();
        s.targets.insert(noted.targets.begin(), noted.targets.end());
        s.writtenQueue.insert(s.writtenQueue.end(), noted.written.begin(), noted.written.end());
        noted.written.clear();
        for (const auto& words : noted.sampled) {
            if (s.images.count(words) == 0) {
                s.sampledQueue.push_back(words);
                continue;
            }
            const auto found = s.registered.find(words);
            if (found != s.registered.end() && found->second.surface != nullptr && found->second.surface->stale) s.staleSampled.push_back(found->second.surface);
        }
        noted.sampled.clear();
    }
    if (!s.staleSampled.empty()) {
        ++s.recordings;
        s.context.detiler->BeginBatch();
        auto dirty = std::move(s.staleSampled);
        s.staleSampled.clear();
        for (auto* surface : dirty) {
            if (!surface->stale) continue;
            surface->stale = false;
            s.upload(commands, *surface, true);
        }
    }
    if (!s.sampledQueue.empty()) {
        ++s.recordings;
        s.context.detiler->BeginBatch();
        auto sampled = std::move(s.sampledQueue);
        s.sampledQueue.clear();
        for (const auto& words : sampled) s.registerImage(commands, words, false);
    }
    if (!s.textureQueue.empty() || !s.samplerQueue.empty()) {
        ++s.recordings;
        s.context.detiler->BeginBatch();
        auto textures = std::move(s.textureQueue);
        s.textureQueue.clear();
        for (const auto& words : textures) s.registerImage(commands, words, true);
        auto samplers = std::move(s.samplerQueue);
        s.samplerQueue.clear();
        for (const auto& words : samplers) s.registerSampler(words);
    }
    if (!s.writtenQueue.empty()) {
        auto written = std::move(s.writtenQueue);
        s.writtenQueue.clear();
        for (const auto& range : written) {
            bool uploaded = false, pending = false;
            for (const auto& [key, surface] : s.surfaces) {
                const auto first = surface.resource.baseAddress;
                if (range.first >= first + surface.geometry.guestBytes || first >= range.first + range.second) continue;
                (surface.uploaded ? uploaded : pending) = true;
            }
            char detail[64];
            std::snprintf(detail, sizeof(detail), "%llu bytes", static_cast<unsigned long long>(range.second));
            if (!uploaded && pending) Native::ReportGraphicsIssue("CPU write notified for a texture not uploaded yet", detail);
            else if (!uploaded) Native::ReportGraphicsIssue("CPU write notified over no known texture", detail);
        }
        bool begun = false;
        for (auto& [key, surface] : s.surfaces) {
            if (!surface.uploaded || surface.target || s.targets.count(surface.resource.baseAddress) != 0) continue;
            const auto begin = surface.resource.baseAddress;
            const auto end = begin + surface.geometry.guestBytes;
            const bool overlaps = std::any_of(written.begin(), written.end(), [&](const auto& range) { return range.first < end && begin < range.first + range.second; });
            if (!overlaps) continue;
            if (!begun) {
                s.context.detiler->BeginBatch();
                ++s.recordings;
                begun = true;
            }
            s.upload(commands, surface);
        }
    }
}

std::uint64_t NativeImages::Recordings() const {
    return state->recordings;
}

void NativeImages::Completed() {
    auto& s = *state;
    std::vector<std::array<std::uint32_t, 8>> missed;
    s.takeMisses(*s.imageTable, missed, s.missedImages);
    // A missed T# is registered as a view of an existing (or new) surface; it is not new content.
    for (const auto& words : missed) {
        if (s.images.count(words) == 0) s.textureQueue.push_back(words);
        try {
            const auto resource = Graphics::DecodeTextureResource(words);
            char detail[160];
            std::snprintf(detail, sizeof(detail), "0x%llx %ux%u format %u%s", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, s.targets.count(resource.baseAddress) != 0 ? " (render target memory)" : "");
            Native::ReportGraphicsIssue("texture sampled before it was registered, drawn without it in that batch", detail);
        } catch (const std::exception& error) {
            Native::ReportGraphicsIssue("texture sampled before it was registered, drawn without it in that batch", error.what());
        }
    }
    s.takeMisses(*s.samplerTable, s.samplerQueue, s.missedSamplers);
    static int batches = 0;
    if (++batches % 600 == 1) {
        const auto* image = reinterpret_cast<const std::uint32_t*>(s.imageTable->Bytes().data());
        const auto* sampler = reinterpret_cast<const std::uint32_t*>(s.samplerTable->Bytes().data());
        if (Native::Logs()) std::fprintf(stderr, "[native] tables after batch %d: images mask %x misses %u/%u, samplers mask %x misses %u/%u; totals %zu/%zu\n", batches, image[0], image[1], image[2], sampler[0], sampler[1], sampler[2], s.missedImages, s.missedSamplers);
    }
    static std::size_t reported = 0;
    if (s.images.size() / 100 != reported) {
        reported = s.images.size() / 100;
        if (Native::Logs()) std::fprintf(stderr, "[native] images: %zu T#s over %zu surfaces (%zu uploads), %zu samplers; misses %zu images, %zu samplers; %zu failed\n", s.images.size(), s.surfaces.size(), s.uploads, s.samplers.size(), s.missedImages, s.missedSamplers, s.failedImages.size());
    }
}

void NoteTextureDescriptor(std::span<const std::uint32_t, 8> words) {
    auto& noted = Pending();
    std::array<std::uint32_t, 8> copy;
    std::memcpy(copy.data(), words.data(), sizeof(copy));
    std::lock_guard lock(noted.mutex);
    noted.textures.push_back(copy);
}

void NoteSampledDescriptor(std::span<const std::uint32_t, 8> words) {
    auto& noted = Pending();
    std::array<std::uint32_t, 8> copy;
    std::memcpy(copy.data(), words.data(), sizeof(copy));
    std::lock_guard lock(noted.mutex);
    noted.sampled.push_back(copy);
}

void NoteGuestWrite(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    auto& noted = Pending();
    std::lock_guard lock(noted.mutex);
    noted.written.emplace_back(address, bytes);
}

void NoteRenderTarget(std::uint64_t address, std::uint64_t) {
    auto& noted = Pending();
    std::lock_guard lock(noted.mutex);
    noted.targets.insert(address);
}

}
