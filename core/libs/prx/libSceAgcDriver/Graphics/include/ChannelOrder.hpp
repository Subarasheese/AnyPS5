#pragma once
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
namespace AgcDriver::Graphics {
// Bit-preserving channel permutation. Both scratch buffers live in device memory.
class ChannelOrder {
  public:
    ChannelOrder(const Context &, std::uint32_t width, std::uint32_t height, std::uint32_t bytes, std::uint32_t depth = 1);
    ~ChannelOrder();
    void Copy(VkCommandBuffer, VkImage source, VkImage destination, std::uint8_t mapping, bool inverse, std::uint32_t mip = 0, std::uint32_t layer = 0, std::uint32_t copyDepth = 1) const;

  private:
    void Release();
    Context context;
    std::uint32_t width, height, bytes, depth;
    std::unique_ptr<DeviceBuffer> input, output;
    VkDescriptorSetLayout descriptors{};
    VkDescriptorPool pool{};
    VkDescriptorSet set{};
    VkPipelineLayout layout{};
    VkPipeline pipeline{};
};
} // namespace AgcDriver::Graphics
