#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_NATIVEDEVICE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_NATIVEDEVICE_HPP

// The native GT7 renderer's GPU: the driver's Vulkan device (its feature set is the one translated
// shaders are compiled for), used as a plain device, with the game's GPU memory imported in place.
//
// Every range GT7 maps for GPU access is imported once, as it is mapped, as VkBuffers over the
// game's own pages (VK_EXT_external_memory_host). Buffers the game describes are then plain device
// addresses: a translated shader turns a guest address into a device address through the page
// table (ShaderRecompiler::BdaAbi layout), which changes only when GT7 maps or unmaps GPU memory.

#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <array>
#include <functional>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver {
class VulkanDevice;
namespace Graphics {
struct State;
struct GraphicsPipelineRecipe;
struct CompiledShader;
class PipelineJournalReplay;
}
}

namespace AgcDriver::Native {

class NativeDevice {
public:
    struct Statistics {
        std::size_t imports = 0;
        std::uint64_t importedBytes = 0;
        std::size_t failedImports = 0;
        std::uint64_t pageTableGeneration = 0;
    };

    // Creates the native device over the driver's Vulkan device (once) and imports the GPU memory
    // already mapped.
    static NativeDevice& Create(std::shared_ptr<VulkanDevice> device);
    // The device, or null before the driver's device exists.
    static NativeDevice* Get();

    ~NativeDevice();
    NativeDevice(const NativeDevice&) = delete;
    NativeDevice& operator=(const NativeDevice&) = delete;

    // Brings the imports in line with the GPU mappings of the allocation registry.
    void SyncGpuMemory();
    // Brings imports up to date when GPU mappings changed since the last sync.
    void SyncChangedMappings() const;
    // The device address of guest address `address`, 0 when no import covers [address, +bytes).
    std::uint64_t DeviceAddress(std::uint64_t address, std::uint64_t bytes = 1) const;
    // The device address of readable guest memory at `address` and the bytes mapped contiguously from it.
    std::uint64_t DeviceSpan(std::uint64_t address, std::uint64_t& bytes) const;
    Statistics Stats() const;

    // The recompiler target for this device's shaders, with resources resolved on the GPU
    // (SpirvTarget::gpuResolvedResources). `waveSize` 32 asks for a 32-lane subgroup.
    ShaderRecompiler::SpirvTarget ComputeTarget(std::uint32_t waveSize) const;

    // One compute dispatch of a shader compiled for ComputeTarget: `shaderData` is its shader data
    // block (user data, partial-group limits) for this dispatch; `pipelineKey` names the compiled
    // shader (same key, same SPIR-V). Recorded into the open batch; false (with `failure`) when
    // the shader needs a resource this device does not provide yet.
    struct ComputeWork {
        const ShaderRecompiler::RecompileResult* shader = nullptr;
        std::uint64_t pipelineKey = 0;
        std::array<std::uint32_t, 3> groups{};
        std::span<const std::uint32_t> shaderData;
        std::uint64_t programAddress = 0;   // GT7's program (diagnostics)
        std::uint64_t indirectArguments = 0;   // DISPATCH_INDIRECT: the group counts' address in GT7's memory (read by the GPU)
        std::span<const std::uint32_t> flattenedSrt;   // The descriptor words walked at the call site (DescriptorRole::FlattenedSrt).
        std::span<const std::uint32_t> slotWords;   // The T#s (8 words) then S#s (4) of the shader's image and sampler slots.
    };
    bool Dispatch(const ComputeWork& work, std::string& failure);
    std::shared_ptr<Graphics::PipelineJournalReplay> ReplayPipelineJournal(bool prepareAll = false);
    bool PrebuildGraphicsLibrary(const ShaderRecompiler::RecompileResult& shader, ShaderRecompiler::ShaderStage stage, std::uint32_t wave, std::uint32_t pushOffset, const std::optional<ShaderRecompiler::MeshConfiguration>& mesh);
    bool PrebuildCompute(const ShaderRecompiler::RecompileResult& shader, std::string& failure);
    // DMA_DATA to/from the persistent device-local GDS; false for other DMA.
    bool DmaGds(std::span<const std::uint32_t> packet, bool wait = true);

    // One draw of GT7's: its decoded pipeline state, the compiled stages (compiled for
    // GraphicsTarget) with each stage's shader data block, and the draw's parameters. Recorded into
    // the open batch; false (with `failure`) when something it needs is not native yet.
    struct DrawStage {
        const ShaderRecompiler::RecompileResult* shader = nullptr;
        std::uint64_t programAddress = 0; // Runtime base for S_GETPC/inline constants.
        ShaderRecompiler::ShaderStage stage{};
        std::uint32_t pushOffset = 0;
        std::vector<std::uint32_t> shaderData;
        std::vector<std::uint32_t> flattenedSrt;
        std::vector<std::uint32_t> slotWords;
        // A vertex stage's fixed-function attributes with this draw's vertex buffer descriptors.
        std::vector<ShaderRecompiler::VertexAttribute> attributes;
    };
    struct DrawWork {
        const Graphics::State* state = nullptr;
        std::vector<DrawStage> stages;
        bool indexed = false;
        std::uint64_t indexAddress = 0;
        std::uint32_t indexSize = 2;
        std::uint32_t count = 0;
        std::uint32_t instances = 1;
        std::uint32_t firstVertex = 0;
        std::uint32_t firstInstance = 0;
        // Indirect draws: `indirectCount` records of `indirectStride` bytes at `indirectArguments`
        // (GT7's memory, written by the GPU), or as many as the dword at `indirectCountAddress`
        // says, at most indirectCount.
        std::uint64_t indirectArguments = 0;
        std::uint32_t indirectCount = 0;
        std::uint32_t indirectStride = 0;
        std::uint64_t indirectCountAddress = 0;
        std::array<std::uint64_t, 2> programAddresses{};   // vertex and pixel programs (diagnostics)
        std::vector<std::uint64_t> sampled;
    };
    bool Draw(const DrawWork& work, std::string& failure);
    ShaderRecompiler::SpirvTarget GraphicsTarget() const;
    // The image GT7 last rendered at a display buffer's address (the presenter's source).
    bool DisplayImage(std::uint64_t address, VkImage& image, VkExtent2D& extent, VkFormat& format);
    // Submits the open batch and waits until the GPU finished it.
    void Flush();
    void Submit();
    bool TransferOnGpu(std::span<const std::uint32_t> packet);
    // Runs `action` once the open batch's work has completed (at its flush), or now when no batch
    // is open: GT7's end-of-pipe events (RELEASE_MEM label writes and interrupts).
    void AfterBatch(std::function<void()> action);
    // After the batch `serial` (RecordingSerial when work was recorded) completes; at once if it has.
    void AfterBatch(std::function<void()> action, std::uint64_t serial);
    // The batch work recorded now lands in (or a later one).
    std::uint64_t RecordingSerial() const;
    bool Pending() const;
    void RequireBarrier();
    void NoteFlip();
    void NoteSubmission();
    bool WritesGuestMemory(const ShaderRecompiler::RecompileResult& shader);
    std::string TextureOverlap(std::uint64_t begin, std::uint64_t end);
    void MarkTextureMemoryWritten(std::uint64_t begin, std::uint64_t end);
    bool TextureRegistered(std::span<const std::uint32_t, 8> words);
    void RequestSurfaceDump(std::uint64_t address);

private:
    explicit NativeDevice(std::shared_ptr<VulkanDevice> device);
    bool ReplayGraphicsPipeline(const Graphics::GraphicsPipelineRecipe& recipe, std::span<const Graphics::CompiledShader> shaders, bool prepareAll);
    struct State;
    std::unique_ptr<State> state;
};

// Called by the allocation registry's GPU-map observer: the native device re-syncs its imports.
void NoteGpuMappingsChanged();

}

#endif
