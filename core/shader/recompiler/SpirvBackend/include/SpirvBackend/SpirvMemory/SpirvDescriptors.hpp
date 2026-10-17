#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVMEMORY_SPIRVDESCRIPTORS_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVMEMORY_SPIRVDESCRIPTORS_HPP

#include "SpirvBackend/SpirvEmitterState.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>
#include <span>

namespace ShaderRecompiler {

std::uint32_t RuntimeImageDword(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t member);
// GPU-resolved resources: the global table entry id (slot + 1, 0: miss) of an access's T#, and the
// heap index of its S#'s sampler (RuntimeAbi::NativeHeap).
std::uint32_t EmitNativeImageLookup(SpirvEmitterState& state, std::span<const std::uint32_t, 8> words);
std::uint32_t EmitNativeSamplerLookup(SpirvEmitterState& state, std::span<const std::uint32_t, 4> words);
std::uint32_t ResourceForDescriptor(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource);
std::uint32_t DescriptorElementPointer(SpirvEmitterState& state, std::uint32_t resultPtrType, std::uint32_t variableId, std::uint32_t arrayIndex, DescriptorBindingKind kind, std::uint32_t resource, const char* variableName);
std::uint32_t ImageScalarType(SpirvEmitterState& state, IrTextureNumericClass numericClass);
std::uint32_t ImageVectorType(SpirvEmitterState& state, IrTextureNumericClass numericClass, std::uint32_t components);
std::uint32_t ImageType(SpirvEmitterState& state, const ImageResource& image);
std::uint32_t ImageViewSizeType(SpirvEmitterState& state, RdnaImageDimension dimension);
std::uint32_t LoadSampledImageDescriptor(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t slotId);
std::uint32_t LoadSamplerDescriptor(SpirvEmitterState& state, std::uint32_t sampler);
std::uint32_t MakeSampledImage(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t sampler, std::uint32_t slotId);
std::uint32_t StorageImageDescriptorPointer(SpirvEmitterState& state, std::uint32_t resource);
void EmitStorageImageWrite(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t mipLod, std::uint32_t coord, std::uint32_t texel, std::uint32_t sample);
const RdnaImageDimensionInfo& RdnaImageDimensionInfoFor(RdnaImageDimension dimension);
[[noreturn]] void ExitDescriptorBindingFailure(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource, const char* reason);

}

#endif
