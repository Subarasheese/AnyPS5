#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_NATIVEIMAGES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_NATIVEIMAGES_HPP

// The native renderer's images and samplers. GT7 describes every texture it creates with a T# made
// by its statically linked texture initializer (0x27d120) and every render target with the
// render-target initializer (0x1638690); hooks on both hand those descriptions here, and each
// becomes a native VkImage, detiled once from the game's (imported) memory when it is a texture.
//
// Translated shaders find their images from their own T# words: a global hash table (device
// memory, RuntimeAbi::NativeHeap) maps the words to heap indices of descriptor set
// NativeHeap::Set, one bindless array per image binding kind. A T# or S# the table does not hold
// yet (a descriptor GT7 patched after initializing it, or a sampler, which has no initializer) is
// appended by the shader to the table's miss region and registered after the batch.

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"

namespace AgcDriver::Graphics {
struct ColorTarget;
struct DepthTarget;
}
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Native {

class NativeImages {
public:
    // The imported VkBuffer holding [address, address + bytes) and the offset of `address` in it.
    using Locate = std::function<std::uint64_t(std::uint64_t address, VkBuffer& buffer, VkDeviceSize& offset)>;

    NativeImages(const Graphics::Context& context, Locate locate);
    ~NativeImages();
    NativeImages(const NativeImages&) = delete;
    NativeImages& operator=(const NativeImages&) = delete;

    // Device addresses of the lookup tables (SpirvTarget::nativeImageTable / nativeSamplerTable).
    std::uint64_t ImageTable() const;
    std::uint64_t SamplerTable() const;
    VkDescriptorSetLayout HeapLayout() const;
    VkDescriptorSet HeapSet() const;

    // A color target's attachment view (and its image), recording the surface's creation if new.
    VkImageView TargetView(VkCommandBuffer commands, const Graphics::ColorTarget& color, VkImage& image);
    // A depth target's depth/stencil attachment view (GENERAL layout).
    VkImageView DepthView(VkCommandBuffer commands, const Graphics::DepthTarget& depth);
    // The image GT7 last rendered into at `address` (a display buffer), false when none.
    bool Display(std::uint64_t address, VkImage& image, VkExtent2D& extent, VkFormat& format) const;

    // On the recording thread, in the open batch before work that may use the resources: creates the
    // images and samplers noted since the last call and records their uploads.
    struct DrawnTarget {
        std::uint64_t address;
        VkImage image;
        VkFormat format;
        std::uint32_t width;
        std::uint32_t height;
        std::uint64_t drawn;
        std::uint64_t guestBytes = 0;
        std::uint32_t guestFormat = 0;
        std::uint32_t tileMode = 0;
        bool depth = false;
    };
    std::vector<DrawnTarget> DrawnTargets(std::uint64_t since) const;
    std::vector<DrawnTarget> SurfacesAt(std::uint64_t address) const;
    // Every surface with storage views (written by compute), for the zero-surface survey.
    std::vector<DrawnTarget> StorageSurfaces() const;
    std::string DescribeUploadedOverlap(std::uint64_t begin, std::uint64_t end) const;
    void MarkGpuWritten(std::uint64_t begin, std::uint64_t end);
    void BeforeRecording(std::function<void()> close);
    std::vector<std::shared_ptr<void>> TakeBatchResources();
    bool Registered(std::span<const std::uint32_t, 8> words) const;
    // The heap indices a draw's image and sampler slots hold (IrBindingLayout::nativeImageSlots):
    // false while the descriptor's resource does not exist yet, which queues it for the next Prepare.
    bool ImageIndices(std::span<const std::uint32_t, 8> words, std::uint32_t& sampled, std::uint32_t& storage);
    bool SamplerIndex(std::span<const std::uint32_t, 4> words, std::uint32_t& sampler);
    std::uint64_t DrawSerial() const;
    void NewBatch();
    void Prepare(VkCommandBuffer commands);
    std::uint64_t Recordings() const;
    // After the batch completed: frees its staging memory and takes the shaders' misses.
    void Completed();

private:
    struct State;
    std::unique_ptr<State> state;
};

// From the game's initializer hooks (any thread, before or after the native device exists).
void NoteTextureDescriptor(std::span<const std::uint32_t, 8> words);
void NoteRenderTarget(std::uint64_t address, std::uint64_t bytes);
void NoteGuestWrite(std::uint64_t address, std::uint64_t bytes);
void NoteSampledDescriptor(std::span<const std::uint32_t, 8> words);

}

#endif
