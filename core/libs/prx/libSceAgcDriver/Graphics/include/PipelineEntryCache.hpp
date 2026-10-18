#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINEENTRYCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINEENTRYCACHE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstddef>
#include <functional>
#include <span>

namespace AgcDriver::Graphics {

VkResult CreateWithEntryCache(const Context& context, std::span<const std::byte> key, const std::function<VkResult(VkPipelineCache)>& create);

}

#endif
