#include "prx/libSceAgcDriver/Graphics/include/ChannelOrder.hpp"
#include "prx/libSceAgcDriver/Graphics/shaders/ChannelOrder_spv.h"
#include <array>
#include <cstdio>
namespace AgcDriver::Graphics {
ChannelOrder::ChannelOrder(const Context &c, std::uint32_t w, std::uint32_t h, std::uint32_t b, std::uint32_t d)
    : context(c), width(w), height(h), bytes(b), depth(d) {
    std::fprintf(stderr, "[channel-order] GPU attachment proxy %ux%ux%u, %u bytes/texel\n", w, h, d, b);
    VkShaderModule module{};
    try {
        const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        input = std::make_unique<DeviceBuffer>(c, std::size_t(w) * h * d * b, usage);
        output = std::make_unique<DeviceBuffer>(c, std::size_t(w) * h * d * b, usage);
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (unsigned i = 0; i < 2; i++)
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        di.bindingCount = 2;
        di.pBindings = bindings.data();
        Check(c.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(c.device, &di, nullptr, &descriptors),
              "channel descriptors");
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1;
        li.pSetLayouts = &descriptors;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &push;
        Check(c.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(c.device, &li, nullptr, &layout), "channel layout");
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = sizeof(CHANNEL_ORDER_SPV);
        mi.pCode = CHANNEL_ORDER_SPV;
        Check(c.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(c.device, &mi, nullptr, &module), "channel shader");
        VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pi.stage.module = module;
        pi.stage.pName = "main";
        pi.layout = layout;
        Check(c.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(c.device, c.pipelineCache, 1, &pi, nullptr, &pipeline),
              "channel pipeline");
        c.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(c.device, module, nullptr);
        module = {};
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &size;
        Check(c.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(c.device, &poolInfo, nullptr, &pool), "channel pool");
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &descriptors;
        Check(c.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(c.device, &ai, &set), "channel set");
        std::array<VkDescriptorBufferInfo, 2> infos{{{input->Handle(), 0, input->Size()}, {output->Handle(), 0, output->Size()}}};
        std::array<VkWriteDescriptorSet, 2> writes{};
        for (unsigned i = 0; i < 2; i++) {
            writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        c.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(c.device, 2, writes.data(), 0, nullptr);
    } catch (...) {
        if (module)
            c.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(c.device, module, nullptr);
        Release();
        throw;
    }
}
ChannelOrder::~ChannelOrder() { Release(); }
void ChannelOrder::Release() {
    if (pipeline)
        context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
    if (layout)
        context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
    if (pool)
        context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (descriptors)
        context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, descriptors, nullptr);
}
void ChannelOrder::Copy(VkCommandBuffer cmd, VkImage source, VkImage destination, std::uint8_t mapping, bool inverse, std::uint32_t mip, std::uint32_t layer, std::uint32_t copyDepth) const {
    const auto copyWidth = std::max(width >> mip, 1u), copyHeight = std::max(height >> mip, 1u);
    Require(copyDepth != 0 && copyDepth <= std::max(depth >> mip, 1u), "channel copy depth exceeds its buffer extent");
    const auto rows = static_cast<std::uint64_t>(copyHeight) * copyDepth;
    Require(rows <= context.limits.maxComputeWorkGroupCount[1], "channel copy volume exceeds compute dispatch height");
    const auto barrier = [&](VkPipelineStageFlags from, VkPipelineStageFlags to, VkAccessFlags reads, VkAccessFlags writes) {
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, reads, writes};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cmd, from, to, 0, 1, &m, 0, nullptr, 0, nullptr);
    };
    // Reuse is ordered after both preceding image writes and the last use of scratch.
    barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
    region.imageExtent = {copyWidth, copyHeight, copyDepth};
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(cmd, source, VK_IMAGE_LAYOUT_GENERAL, input->Handle(), 1,
                                                                           &region);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0,
                                                                             nullptr);
    const std::array<std::uint32_t, 4> push{copyWidth, static_cast<std::uint32_t>(rows), bytes, std::uint32_t(mapping) | (inverse ? 256u : 0u)};
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push.data());
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(cmd, (copyWidth + 63) / 64, static_cast<std::uint32_t>(rows), 1);
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(cmd, output->Handle(), destination, VK_IMAGE_LAYOUT_GENERAL, 1,
                                                                           &region);
}
} // namespace AgcDriver::Graphics
