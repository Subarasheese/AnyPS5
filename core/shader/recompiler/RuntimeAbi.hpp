#ifndef CORE_SHADER_RECOMPILER_RUNTIMEABI_HPP
#define CORE_SHADER_RECOMPILER_RUNTIMEABI_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <stdexcept>

namespace ShaderRecompiler::RuntimeAbi {

inline constexpr std::uint32_t Version = 11u;
inline constexpr std::uint32_t DescriptorSet = 0u;
inline constexpr std::uint32_t FragmentDescriptorSet = 1u;
inline constexpr std::uint32_t StageCount = 4u;
inline constexpr std::uint32_t PushConstantDwords = 32u;
inline constexpr std::uint32_t FirstImageBinding = 1u;
inline constexpr std::uint32_t FirstComparisonImageBinding = 22u;
inline constexpr std::uint32_t FirstStorageImageBinding = 29u;
inline constexpr std::uint32_t ImageBindingCount = 56u;

enum class Binding : std::uint32_t {
    Buffers = 0u,
    Samplers = 57u,
    Gds = 58u,
    BdaPagetable = 59u,
    FaultBuffer = 60u,
    FlattenedSrt = 61u,
    ShaderData = 62u,
    Count = 63u
};

enum class Stage : std::uint32_t { Main, Fragment, TessellationControl, TessellationEvaluation };

inline constexpr std::uint32_t UserDataCapacity = 128u;
inline constexpr std::uint32_t BufferCapacity = 128u;
inline constexpr std::uint32_t ImageCapacity = 256u;
inline constexpr std::uint32_t SampledHeapCapacity = 64u;
inline constexpr std::uint32_t BindlessTableSlots = 16u;
// BASE_LEVEL and LAST_LEVEL in the descriptor are four-bit fields.
inline constexpr std::uint32_t StorageMipSlots = 16u;
inline constexpr std::uint32_t StorageHeapCapacity = 64u;
inline constexpr std::uint32_t SamplerHeapCapacity = 32u;
inline constexpr std::uint32_t StateExportMappings = 0u;
inline constexpr std::uint32_t StateDwords = 2u;
inline constexpr std::uint32_t StateVertexSelectors = 0u;
inline constexpr std::uint32_t VertexSelectorsPerDword = 2u;
inline constexpr std::uint32_t ImageModeMask = 0xffu;
inline constexpr std::uint32_t CompareStateShift = 9u;
inline constexpr std::uint32_t CompareStateMask = 0x1fffu;

struct ResourceMetadata {
    std::uint32_t binding;
    std::uint32_t firstElement;
    std::uint32_t elementCount;
    std::uint32_t flags;
    std::array<std::uint32_t, 8> descriptor;
};

struct ShaderData {
    std::uint32_t version;
    std::uint32_t imageCount;
    std::uint32_t samplerCount;
    std::uint32_t reserved;
    std::array<std::uint32_t, UserDataCapacity> userData;
    std::array<std::uint32_t, BufferCapacity / 4u> bufferOffsets;
    std::array<std::uint32_t, 4> dispatchThreadLimit;
    std::array<ResourceMetadata, ImageCapacity> images;
    std::array<ResourceMetadata, SamplerHeapCapacity> samplers;
    std::array<std::uint32_t, 8> exportMappings;
};

inline constexpr std::uint32_t UserDataDword = offsetof(ShaderData, userData) / sizeof(std::uint32_t);
inline constexpr std::uint32_t BufferOffsetsDword = offsetof(ShaderData, bufferOffsets) / sizeof(std::uint32_t);
inline constexpr std::uint32_t DispatchThreadLimitDword = offsetof(ShaderData, dispatchThreadLimit) / sizeof(std::uint32_t);
inline constexpr std::uint32_t ExportMappingsDword = offsetof(ShaderData, exportMappings) / sizeof(std::uint32_t);
inline constexpr std::uint32_t ShaderDataDwords = sizeof(ShaderData) / sizeof(std::uint32_t);

inline std::uint32_t HeapCapacity(Binding binding) {
    const auto value = static_cast<std::uint32_t>(binding);
    if (value >= FirstImageBinding && value < FirstStorageImageBinding) return SampledHeapCapacity;
    if (value >= FirstStorageImageBinding && value < static_cast<std::uint32_t>(Binding::Samplers)) return StorageHeapCapacity;
    if (binding == Binding::Samplers) return SamplerHeapCapacity;
    throw std::runtime_error("Shader runtime ABI: binding is not a typed heap");
}

static_assert(std::is_standard_layout_v<ResourceMetadata> && std::is_trivially_copyable_v<ResourceMetadata> && sizeof(ResourceMetadata) == 48u);
static_assert(std::is_standard_layout_v<ShaderData> && std::is_trivially_copyable_v<ShaderData> && sizeof(ShaderData) == 14528u);
static_assert(UserDataDword == 4u && BufferOffsetsDword == 132u && DispatchThreadLimitDword == 164u);
static_assert(ExportMappingsDword == 3624u);
static_assert(offsetof(ResourceMetadata, descriptor) == 16u && offsetof(ShaderData, images) == 672u && offsetof(ShaderData, samplers) == 12960u);

inline void RequireVersion(std::uint32_t version) {
    if (version != Version) throw std::runtime_error("Shader runtime ABI: incompatible version");
}

inline std::uint32_t BindingNumber(Stage stage, Binding binding) {
    const auto group = static_cast<std::uint32_t>(stage);
    const auto index = static_cast<std::uint32_t>(binding);
    const auto count = static_cast<std::uint32_t>(Binding::Count);
    if (group >= StageCount || index >= count) throw std::runtime_error("Shader runtime ABI: invalid stage or binding");
    return group * count + index;
}

static_assert(FirstImageBinding + ImageBindingCount == static_cast<std::uint32_t>(Binding::Samplers));

// GPU-resolved resources (SpirvTarget::gpuResolvedResources, the native renderer): a program finds
// each image and sampler at run time from its own T#/S# words in a global open-addressing table
// (SpirvTarget::nativeImageTable / nativeSamplerTable, device addresses), and binds them from
// global heaps in descriptor set NativeHeap::Set, at the heap kinds' compute-stage binding numbers.
// The native renderer's page table (GPU-resolved programs read guest memory through it): header
// {Magic, level-1 count, 0, 0}; level 1 (u32 per 16 MiB of guest address space: level-2 block + 1,
// 0 unmapped) from word HeaderWords; level-2 blocks after it (aligned to 4 words), PagesPerBlock
// entries of {device address of the 64 KiB page (lo, hi), valid bytes of the page as
// begin | end << 16... see Entry}. Two dependent loads per translation.
namespace NativePageTable {
inline constexpr std::uint32_t Magic = 0x4e505431u;   // "NPT1"
inline constexpr std::uint32_t HeaderWords = 4;
inline constexpr std::uint32_t PageShift = 16;
inline constexpr std::uint32_t BlockShift = 24;
inline constexpr std::uint32_t PagesPerBlock = 1u << (BlockShift - PageShift);
struct Entry {
    std::uint32_t deviceLow, deviceHigh;   // device address of the page's first byte (0: unmapped)
    std::uint32_t validBegin;              // first mapped byte of the page
    std::uint32_t validBytes;              // mapped bytes from the page's start to its import's end (capped), | permissions in the top 2 bits
};
static_assert(sizeof(Entry) == 16);
inline constexpr std::uint32_t PermissionShift = 30;
inline constexpr std::uint32_t BytesMask = (1u << PermissionShift) - 1u;
inline std::uint32_t Level2Word(std::uint32_t level1Count) { return (HeaderWords + level1Count + 3u) & ~3u; }
}

namespace NativeHeap {
// GPU-resolved shader data ends in two runtime BDA pointers (image,sampler).
// Neither pointer is part of the translated module or its persistent cache key.
inline constexpr std::uint32_t TableAddressDwords = 4u;
inline constexpr std::uint32_t ProgramAddressDwords = 2u;
inline constexpr std::uint32_t InvocationAddressDwords = TableAddressDwords + ProgramAddressDwords;

inline constexpr std::uint32_t Set = 2;
inline constexpr std::uint32_t ImageCapacity = 16384;
inline constexpr std::uint32_t SamplerCapacity = 4096;
inline constexpr std::uint32_t ProbeLimit = 16;
// Table layout: u32 slot mask (slot count - 1, a power of two minus one) at +0; the miss region's
// u32 count (atomically bumped per miss), capacity (entries) and byte offset from the table at
// +4/+8/+12, each miss entry the looked-up words; entries from +16.
inline constexpr std::uint32_t HeaderBytes = 16;
struct ImageEntry {
    std::uint32_t words[8];   // the T#; words[0] == words[1] == 0: empty slot
    std::uint32_t sampled;    // heap index of the sampled view (0: the null image)
    std::uint32_t storage;    // heap index of mip 0's storage view; mip m at storage + m
    std::uint32_t classWord;  // reserved for runtime image modes
    std::uint32_t reserved;
};
struct SamplerEntry {
    std::uint32_t words[4];   // the S#; all zero: empty slot
    std::uint32_t sampler;    // heap index of the sampler; sampler + 1 the point-filtered variant
    std::uint32_t reserved[3];
};
static_assert(sizeof(ImageEntry) == 48 && sizeof(SamplerEntry) == 32);
inline std::uint32_t Hash(const std::uint32_t* words, std::uint32_t count) {
    std::uint32_t hash = 0x811c9dc5u;
    for (std::uint32_t index = 0; index < count; ++index) {
        hash ^= words[index];
        hash *= 0x01000193u;
    }
    return hash;
}
}

}

#endif
