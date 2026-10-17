#include "SpirvBackend/SpirvMemory/SpirvDescriptors.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "PipelineSpecialization.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler
{
namespace {

constexpr RdnaImageDimensionInfo ImageDimensions[] = {
    {RdnaImageDimension::Dim1D, spv::Dim1D, 1, 1, 0, 0},
    {RdnaImageDimension::Dim2D, spv::Dim2D, 2, 2, 0, 0},
    {RdnaImageDimension::Dim3D, spv::Dim3D, 3, 3, 0, 0},
    //{RdnaImageDimension::DimCube, spv::DimCube, 3, 2, 0, 0},
    {RdnaImageDimension::Dim1DArray, spv::Dim1D, 2, 1, 1, 0},
    {RdnaImageDimension::Dim2DArray, spv::Dim2D, 3, 2, 1, 0},
    {RdnaImageDimension::Dim2DMsaa, spv::Dim2D, 2, 2, 0, 1},
    {RdnaImageDimension::Dim2DMsaaArray, spv::Dim2D, 3, 2, 1, 1},
};

}

[[noreturn]] void ExitDescriptorBindingFailure(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource, const char* reason) {
    throw std::runtime_error("shader binding resolution failed during SPIR-V emit: hash=" + std::to_string(state.program.Resources().shaderHash) + " stage=" + std::to_string(static_cast<std::uint32_t>(StageOf(state))) + " resource=" + std::to_string(resource) + " binding_kind=" + std::to_string(static_cast<std::uint32_t>(kind)) + " reason=" + reason);
}

std::uint32_t ResourceForDescriptor(const SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource) {
    for (const auto& descriptor : state.program.Metadata().bindings.descriptors) {
        if (descriptor.kind != kind) {
            continue;
        }
        const auto found = std::find(descriptor.resources.begin(), descriptor.resources.end(), resource);
        if (found == descriptor.resources.end()) {
            ExitDescriptorBindingFailure(state, kind, resource, "resource is absent from descriptor group");
        }
        return static_cast<std::uint32_t>(found - descriptor.resources.begin());
    }
    ExitDescriptorBindingFailure(state, kind, resource, "descriptor group was not allocated");
}

namespace {

std::uint32_t PhysicalLoadU32(SpirvEmitterState& state, std::uint32_t address) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, spv::MemoryAccessAlignedMask, 4u);
    return value;
}

// Heap addresses are invocation data, not constants captured when translating.
// Keeping them in the existing shader-data block works for both push data and
// storage-buffer data and permits the same module to survive another process.
std::uint32_t NativeTableAddress(SpirvEmitterState& state, bool sampler) {
    const auto words = state.program.Metadata().bindings.ShaderDataDwords();
    if (words < RuntimeAbi::NativeHeap::InvocationAddressDwords) FailEmit("native heap runtime addresses are missing");
    const auto first = words - RuntimeAbi::NativeHeap::InvocationAddressDwords + (sampler ? 2u : 0u);
    const auto low = Unary(state, spv::OpUConvert, TypeScalarU64(state), EmitShaderDataDwordLoad(state, first));
    const auto high = Unary(state, spv::OpUConvert, TypeScalarU64(state), EmitShaderDataDwordLoad(state, first + 1u));
    return Binary(state, spv::OpBitwiseOr, TypeScalarU64(state), low,
        Binary(state, spv::OpShiftLeftLogical, TypeScalarU64(state), high, BdaConstant(state, 32u)));
}

// (u32 words[count]) -> u32: slot + 1 of the table entry holding exactly these words, 0 when none
// does within the probe limit. Open addressing, linear probing, an all-zero first pair ends a run.
std::uint32_t DefineNativeLookup(SpirvEmitterState& state, const char* name, bool samplerTable, std::uint32_t count, std::uint32_t entryBytes) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto label = state.currentLabel;
    state.module.BeginHelperFunction();
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    if (count != 8u && count != 4u) FailEmit("native table lookups take four or eight words");
    const auto typeId = count == 8u ? state.module.Type(spv::OpTypeFunction, u32, u32, u32, u32, u32, u32, u32, u32, u32) : state.module.Type(spv::OpTypeFunction, u32, u32, u32, u32, u32);
    state.module.AddFunction(spv::OpFunction, u32, function, spv::FunctionControlDontInlineMask, typeId);
    std::vector<std::uint32_t> words(count);
    for (auto& word : words) {
        word = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionParameter, u32, word);
    }
    EmitLabel(state, state.module.AllocateId());
    const auto pointerU32 = TypePointer(state, spv::StorageClassFunction, u32);
    const auto slotVar = state.module.AllocateId();
    const auto probeVar = state.module.AllocateId();
    const auto resultVar = state.module.AllocateId();
    for (const auto variable : {slotVar, probeVar, resultVar}) state.module.AddFunction(spv::OpVariable, pointerU32, variable, spv::StorageClassFunction);
    const auto base = NativeTableAddress(state, samplerTable);
    const auto mask = PhysicalLoadU32(state, base);
    auto hash = constant(0x811c9dc5u);
    for (const auto word : words) hash = binary(spv::OpIMul, u32, binary(spv::OpBitwiseXor, u32, hash, word), constant(0x01000193u));
    state.module.AddFunction(spv::OpStore, slotVar, binary(spv::OpBitwiseAnd, u32, hash, mask));
    state.module.AddFunction(spv::OpStore, probeVar, constant(0u));
    state.module.AddFunction(spv::OpStore, resultVar, constant(0u));
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto load = [&](std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, variable);
        return value;
    };
    const auto probe = load(probeVar);
    const auto searching = binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThan, boolean, probe, constant(RuntimeAbi::NativeHeap::ProbeLimit)), binary(spv::OpIEqual, boolean, load(resultVar), constant(0u)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = load(slotVar);
    const auto entry = binary(spv::OpIAdd, u64, base, Unary(state, spv::OpUConvert, u64, binary(spv::OpIAdd, u32, constant(RuntimeAbi::NativeHeap::HeaderBytes), binary(spv::OpIMul, u32, slot, constant(entryBytes)))));
    auto match = ConstantBool(state, true);
    std::uint32_t first = 0, second = 0;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto stored = PhysicalLoadU32(state, binary(spv::OpIAdd, u64, entry, BdaConstant(state, static_cast<std::uint64_t>(index) * 4u)));
        if (index == 0) first = stored;
        if (index == 1) second = stored;
        match = binary(spv::OpLogicalAnd, boolean, match, binary(spv::OpIEqual, boolean, stored, words[index]));
    }
    const auto empty = binary(spv::OpLogicalAnd, boolean, binary(spv::OpIEqual, boolean, first, constant(0u)), binary(spv::OpIEqual, boolean, second, constant(0u)));
    state.module.AddFunction(spv::OpStore, resultVar, Select(state, u32, match, binary(spv::OpIAdd, u32, slot, constant(1u)), constant(0u)));
    state.module.AddFunction(spv::OpStore, probeVar, Select(state, u32, empty, constant(RuntimeAbi::NativeHeap::ProbeLimit), binary(spv::OpIAdd, u32, probe, constant(1u))));
    state.module.AddFunction(spv::OpStore, slotVar, binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, slot, constant(1u)), mask));
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    // A miss appends the words to the table's miss region (header words 1-3: count, capacity, byte
    // offset from the table), from which the native renderer registers the resource for later work.
    const auto result = load(resultVar);
    const auto missed = binary(spv::OpIEqual, boolean, result, constant(0u));
    const auto record = state.module.AllocateId();
    const auto recorded = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, recorded, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, missed, record, recorded);
    EmitLabel(state, record);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), counter, binary(spv::OpIAdd, u64, base, BdaConstant(state, 4u)));
    const auto capacity = PhysicalLoadU32(state, binary(spv::OpIAdd, u64, base, BdaConstant(state, 8u)));
    // Once feedback is full, extra increments cannot produce a record. Avoid
    // serializing every remaining pixel on the same global atomic counter.
    // Use an atomic read because other invocations are still reserving slots.
    // Racing reservations can overshoot capacity,as before;fits still gates
    // every write and the host consumes min(count,capacity).
    const auto observed = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, observed, counter, constant(spv::ScopeDevice), constant(spv::MemorySemanticsMaskNone));
    const auto feedbackDone = state.module.AllocateId();
    const auto reserve = state.module.AllocateId();
    const auto hasCapacity = binary(spv::OpULessThan, boolean, observed, capacity);
    state.module.AddFunction(spv::OpSelectionMerge, feedbackDone, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, hasCapacity, reserve, feedbackDone);
    EmitLabel(state, reserve);
    const auto index = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicIAdd, u32, index, counter, constant(spv::ScopeDevice), constant(spv::MemorySemanticsMaskNone), constant(1u));
    const auto fits = binary(spv::OpULessThan, boolean, index, capacity);
    const auto store = state.module.AllocateId();
    const auto stored = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, stored, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, fits, store, stored);
    EmitLabel(state, store);
    const auto regionOffset = PhysicalLoadU32(state, binary(spv::OpIAdd, u64, base, BdaConstant(state, 12u)));
    const auto slotOffset = binary(spv::OpIAdd, u32, regionOffset, binary(spv::OpIMul, u32, index, constant(count * 4u)));
    for (std::uint32_t word = 0; word < count; ++word) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, binary(spv::OpIAdd, u64, base, Unary(state, spv::OpUConvert, u64, binary(spv::OpIAdd, u32, slotOffset, constant(word * 4u)))));
        state.module.AddFunction(spv::OpStore, pointer, words[word], spv::MemoryAccessAlignedMask, 4u);
    }
    state.module.AddFunction(spv::OpBranch, stored);
    EmitLabel(state, stored);
    state.module.AddFunction(spv::OpBranch, feedbackDone);
    EmitLabel(state, feedbackDone);
    state.module.AddFunction(spv::OpBranch, recorded);
    EmitLabel(state, recorded);
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.EndHelperFunction();
    state.currentLabel = label;
    return function;
}

// A field of the current access's image table entry (0 for a miss).
std::uint32_t NativeImageField(SpirvEmitterState& state, std::uint32_t field) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto id = state.gpuImageEntry;
    const auto hit = Binary(state, spv::OpINotEqual, TypeBool(state), id, ConstantU32(state, 0u));
    const auto slot = Select(state, u32, hit, Binary(state, spv::OpISub, u32, id, ConstantU32(state, 1u)), ConstantU32(state, 0u));
    const auto offset = Binary(state, spv::OpIAdd, u32, ConstantU32(state, RuntimeAbi::NativeHeap::HeaderBytes + field * 4u), Binary(state, spv::OpIMul, u32, slot, ConstantU32(state, sizeof(RuntimeAbi::NativeHeap::ImageEntry))));
    const auto value = PhysicalLoadU32(state, Binary(state, spv::OpIAdd, u64, NativeTableAddress(state, false), Unary(state, spv::OpUConvert, u64, offset)));
    return Select(state, u32, hit, value, ConstantU32(state, 0u));
}

std::uint32_t NonUniformIndex(SpirvEmitterState& state, std::uint32_t index) {
    state.module.EmitCapability(spv::CapabilityShaderNonUniform);
    state.module.EmitCapability(spv::CapabilitySampledImageArrayNonUniformIndexing);
    state.module.EmitCapability(spv::CapabilityStorageImageArrayNonUniformIndexing);
    state.module.EmitCapability(spv::CapabilitySampledImageArrayDynamicIndexing);
    state.module.EmitCapability(spv::CapabilityStorageImageArrayDynamicIndexing);
    if (state.spirvVersion < 0x00010500u) state.module.EmitExtension("SPV_EXT_descriptor_indexing");
    if (std::find(state.nonUniformIds.begin(), state.nonUniformIds.end(), index) == state.nonUniformIds.end()) {
        state.nonUniformIds.push_back(index);
        state.module.AddAnnotation(spv::OpDecorate, index, spv::DecorationNonUniform);
    }
    return index;
}

}

std::uint32_t EmitNativeImageLookup(SpirvEmitterState& state, std::span<const std::uint32_t, 8> words) {
    if (state.nativeImageTable == 0) FailEmit("GPU-resolved images need the native image table");
    if (state.nativeImageLookupFunction == 0) state.nativeImageLookupFunction = DefineNativeLookup(state, "native_image_lookup", false, 8u, sizeof(RuntimeAbi::NativeHeap::ImageEntry));
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeU32(state), result, state.nativeImageLookupFunction, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    return result;
}

std::uint32_t EmitNativeSamplerLookup(SpirvEmitterState& state, std::span<const std::uint32_t, 4> words) {
    if (state.nativeSamplerTable == 0) FailEmit("GPU-resolved samplers need the native sampler table");
    if (state.nativeSamplerLookupFunction == 0) state.nativeSamplerLookupFunction = DefineNativeLookup(state, "native_sampler_lookup", true, 4u, sizeof(RuntimeAbi::NativeHeap::SamplerEntry));
    const auto id = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeU32(state), id, state.nativeSamplerLookupFunction, words[0], words[1], words[2], words[3]);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto hit = Binary(state, spv::OpINotEqual, TypeBool(state), id, ConstantU32(state, 0u));
    const auto slot = Select(state, u32, hit, Binary(state, spv::OpISub, u32, id, ConstantU32(state, 1u)), ConstantU32(state, 0u));
    const auto offset = Binary(state, spv::OpIAdd, u32, ConstantU32(state, RuntimeAbi::NativeHeap::HeaderBytes + offsetof(RuntimeAbi::NativeHeap::SamplerEntry, sampler)), Binary(state, spv::OpIMul, u32, slot, ConstantU32(state, sizeof(RuntimeAbi::NativeHeap::SamplerEntry))));
    const auto value = PhysicalLoadU32(state, Binary(state, spv::OpIAdd, u64, NativeTableAddress(state, true), Unary(state, spv::OpUConvert, u64, offset)));
    return Select(state, u32, hit, value, ConstantU32(state, 0u));
}

std::uint32_t DescriptorElementPointer(SpirvEmitterState& state, std::uint32_t resultPtrType, std::uint32_t variableId, std::uint32_t arrayIndex, DescriptorBindingKind kind, std::uint32_t resource, const char* variableName) {
    if (variableId == 0) {
        ExitDescriptorBindingFailure(state, kind, resource, variableName);
    }
    const auto pointer = state.module.AllocateId();
    const bool heap = kind == DescriptorBindingKind::Samplers || ImageBindingResourceClass(kind) != ImageResourceClass::None;
    if (heap && state.program.Metadata().gpuResolvedResources) {
        // The global heap index of this access's image (sampled view, or storage view of the mip the
        // element offset names) or sampler (+1: the point-filtered variant).
        const auto offset = arrayIndex - ResourceForDescriptor(state, kind, resource);
        std::uint32_t base = 0;
        if (kind == DescriptorBindingKind::Samplers) base = state.gpuSamplerIndex;
        else if (state.gpuImageEntry != 0) base = NativeImageField(state, ImageBindingResourceClass(kind) == ImageResourceClass::Storage ? offsetof(RuntimeAbi::NativeHeap::ImageEntry, storage) / 4u : offsetof(RuntimeAbi::NativeHeap::ImageEntry, sampled) / 4u);
        if (base == 0) FailEmit("GPU-resolved descriptor access has no table lookup");
        const auto index = NonUniformIndex(state, offset == 0 ? base : Binary(state, spv::OpIAdd, TypeU32(state), base, ConstantU32(state, offset)));
        state.module.AddFunction(spv::OpAccessChain, resultPtrType, pointer, variableId, index);
        state.module.AddAnnotation(spv::OpDecorate, pointer, spv::DecorationNonUniform);
        return pointer;
    }
    const auto index = heap && !state.registrationModules ? state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::DescriptorIndex(static_cast<std::uint32_t>(kind), arrayIndex), arrayIndex) : ConstantU32(state, arrayIndex);
    state.module.AddFunction(spv::OpAccessChain, resultPtrType, pointer, variableId, index);
    return pointer;
}

std::uint32_t RuntimeImageDword(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t member) {
    if (state.program.Metadata().gpuResolvedResources && state.gpuImageEntry != 0) {
        // The T# words are the access's own registers; one runtime mode (flags 0).
        constexpr auto descriptor = offsetof(RuntimeAbi::ResourceMetadata, descriptor) / sizeof(std::uint32_t);
        if (member >= descriptor && member < descriptor + 8u) return state.gpuImageWords.at(member - descriptor);
        if (member == offsetof(RuntimeAbi::ResourceMetadata, firstElement) / sizeof(std::uint32_t)) return NativeImageField(state, offsetof(RuntimeAbi::NativeHeap::ImageEntry, sampled) / 4u);
        return ConstantU32(state, 0u);
    }
    const auto index = state.runtimeImageMetadata != 0u ? state.runtimeImageMetadata : ConstantU32(state, resource);
    const auto dword = Binary(state, spv::OpIAdd, TypeU32(state), ConstantU32(state, state.program.Metadata().bindings.ImageMetadataDword() + member), Binary(state, spv::OpIMul, TypeU32(state), index, ConstantU32(state, sizeof(RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t))));
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer, state.shaderDataStorageVariable, ConstantU32(state, 0u), dword);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
    return value;
}

std::uint32_t RuntimeMipCount(SpirvEmitterState& state, std::uint32_t resource) {
    const auto word = RuntimeImageDword(state, resource, offsetof(RuntimeAbi::ResourceMetadata, descriptor) / sizeof(std::uint32_t) + 3u);
    const auto first = EmitAndConstant(state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), word, ConstantU32(state, 12u)), 0xfu);
    const auto last = EmitAndConstant(state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), word, ConstantU32(state, 16u)), 0xfu);
    return Binary(state, spv::OpIAdd, TypeU32(state), Binary(state, spv::OpISub, TypeU32(state), last, first), ConstantU32(state, 1u));
}

std::uint32_t TableImageIndex(SpirvEmitterState& state, DescriptorBindingKind kind, std::uint32_t resource) {
    const bool storage = ImageBindingResourceClass(kind) == ImageResourceClass::Storage;
    const auto supported = [&](std::uint32_t capability) {
        return std::find(state.supportedCapabilities.begin(), state.supportedCapabilities.end(), capability) != state.supportedCapabilities.end();
    };
    const auto dynamic = storage ? spv::CapabilityStorageImageArrayDynamicIndexing : spv::CapabilitySampledImageArrayDynamicIndexing;
    if (!supported(dynamic)) {
        ExitDescriptorBindingFailure(state, kind, resource, "bindless image table needs image array dynamic indexing, which the device lacks");
    }
    state.module.EmitCapability(dynamic);
    if (state.runtimeImageMetadata == 0u) FailEmit("bindless image has no runtime metadata index");
    const auto offset = Binary(state, spv::OpIAdd, TypeU32(state), ConstantU32(state, state.program.Metadata().bindings.ImageMetadataDword() + offsetof(RuntimeAbi::ResourceMetadata, firstElement) / sizeof(std::uint32_t)), Binary(state, spv::OpIMul, TypeU32(state), state.runtimeImageMetadata, ConstantU32(state, sizeof(RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t))));
    const auto metadata = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), metadata, state.shaderDataStorageVariable, ConstantU32(state, 0u), offset);
    const auto index = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), index, metadata);
    if (!state.tableIndexNonUniform) {
        return index;
    }
    const auto nonUniform = storage ? spv::CapabilityStorageImageArrayNonUniformIndexing : spv::CapabilitySampledImageArrayNonUniformIndexing;
    const bool extension = state.spirvVersion < 0x00010500u;
    if (!supported(spv::CapabilityShaderNonUniform) || !supported(nonUniform) || (extension && std::find(state.supportedExtensions.begin(), state.supportedExtensions.end(), "SPV_EXT_descriptor_indexing") == state.supportedExtensions.end())) {
        ResourceMaterializer::CountBindlessRejection(BindlessRejection::NonUniform);
        throw std::runtime_error("bindless image table: the slot is not uniform over the workgroup and the device lacks non-uniform image indexing");
    }
    state.module.EmitCapability(spv::CapabilityShaderNonUniform);
    state.module.EmitCapability(nonUniform);
    if (extension) state.module.EmitExtension("SPV_EXT_descriptor_indexing");
    state.module.AddAnnotation(spv::OpDecorate, index, spv::DecorationNonUniform);
    return index;
}

void DecorateTableNonUniform(SpirvEmitterState& state, std::uint32_t slotId, std::uint32_t id) {
    if (slotId != 0 && state.tableIndexNonUniform) state.module.AddAnnotation(spv::OpDecorate, id, spv::DecorationNonUniform);
}

std::uint32_t ImageScalarType(SpirvEmitterState& state, IrTextureNumericClass numericClass) {
    switch (numericClass) {
    case IrTextureNumericClass::Float: return TypeF32(state);
    case IrTextureNumericClass::Uint: return TypeU32(state);
    case IrTextureNumericClass::Sint: return TypeI32(state);
    case IrTextureNumericClass::Unsupported: break;
    }
    FailEmit("invalid image numeric class");
}

std::uint32_t ImageVectorType(SpirvEmitterState& state, IrTextureNumericClass numericClass, std::uint32_t components) {
    return state.module.Type(spv::OpTypeVector, ImageScalarType(state, numericClass), components);
}

std::uint32_t ImageType(SpirvEmitterState& state, const ImageResource& image) {
    std::uint32_t sampled = 0;
    std::uint32_t format = spv::ImageFormatUnknown;
    if (image.resourceClass == ImageResourceClass::Sampled) {
        if (image.atomic) {
            FailEmit("sampled image cannot be atomic");
        }
        sampled = 1;
    } else if (image.resourceClass == ImageResourceClass::Storage) {
        if (image.numericClass == IrTextureNumericClass::Sint || image.numericClass == IrTextureNumericClass::Unsupported) {
            FailEmit("storage image numeric class is not supported");
        }
        sampled = 2;
        if (image.atomic) {
            if (image.numericClass != IrTextureNumericClass::Uint) {
                FailEmit("atomic storage image must be uint");
            }
            format = image.atomic64 ? spv::ImageFormatR64ui : spv::ImageFormatR32ui;
        }
    } else {
        FailEmit("invalid image resource class");
    }
    const auto& info = RdnaImageDimensionInfoFor(image.dimension);
    if (info.multisampled != 0u) {
        if (sampled == 2u) {
            if (std::ranges::find(state.supportedCapabilities, spv::CapabilityStorageImageMultisample) == state.supportedCapabilities.end()) FailEmit("storage image multisampling is unavailable on the target device");
            state.module.EmitCapability(spv::CapabilityStorageImageMultisample);
        }
        if (info.arrayed != 0u) state.module.EmitCapability(spv::CapabilityImageMSArray);
    }
    const auto scalar = image.atomic64 ? TypeScalarU64(state) : ImageScalarType(state, image.numericClass);
    return state.module.Type(spv::OpTypeImage, scalar, info.spirvDimension, image.depthCompare ? 1u : 0u, info.arrayed, info.multisampled, sampled, format);
}

std::uint32_t ImageViewSizeType(SpirvEmitterState& state, RdnaImageDimension dimension) {
    switch (RdnaImageDimensionInfoFor(dimension).coordinateComponents) {
    case 1u: return TypeU32(state);
    case 2u: return TypeU32Vector(state, 2);
    case 3u: return TypeU32Vector(state, 3);
    default: FailEmit("image view size component count is invalid");
    }
}

std::uint32_t LoadSampledImageDescriptor(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t slotId) {
    const auto& imageResource = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (imageResource.resourceClass != ImageResourceClass::Sampled) {
        FailEmit("sampled image descriptor requested for a non-sampled image");
    }
    const auto kind = DescriptorBindingForImage(imageResource);
    const auto arrayIndex = ResourceForDescriptor(state, kind, resource);
    const auto variable = state.imageVariables.at(ImageBindingIndex(kind));
    const auto imageType = ImageType(state, imageResource);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType);
    if (variable == 0) {
        ExitDescriptorBindingFailure(state, kind, resource, "sampled image descriptor array was not emitted");
    }
    std::uint32_t pointer = 0;
    if (slotId == 0) {
        pointer = DescriptorElementPointer(state, pointerType, variable, arrayIndex, kind, resource, "sampled image descriptor array was not emitted");
    } else {
        pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpAccessChain, pointerType, pointer, variable, TableImageIndex(state, kind, resource));
        DecorateTableNonUniform(state, slotId, pointer);
    }
    const auto image = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, imageType, image, pointer);
    DecorateTableNonUniform(state, slotId, image);
    return image;
}

std::uint32_t LoadSamplerDescriptor(SpirvEmitterState& state, std::uint32_t sampler) {
    const bool point = state.runtimeImage != nullptr && (state.runtimeImage->numericClass == IrTextureNumericClass::Sint || state.runtimeImage->conversionFormat != IrBufferFormat::Invalid || state.runtimeImage->depthBits);
    const auto arrayIndex = ResourceForDescriptor(state, DescriptorBindingKind::Samplers, sampler) + (point ? 1u : 0u);
    const auto samplerType = state.module.Type(spv::OpTypeSampler);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, samplerType);
    const auto pointer = DescriptorElementPointer(state, pointerType, state.samplerVariable, arrayIndex, DescriptorBindingKind::Samplers, sampler, "sampler descriptor array was not emitted");
    const auto samplerId = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, samplerType, samplerId, pointer);
    return samplerId;
}

std::uint32_t MakeSampledImage(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t sampler, std::uint32_t slotId) {
    const auto& imageResource = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    const auto image = LoadSampledImageDescriptor(state, resource, slotId);
    const auto samplerId = LoadSamplerDescriptor(state, sampler);
    const auto sampledImage = state.module.AllocateId();
    const auto sampledType = state.module.Type(spv::OpTypeSampledImage, ImageType(state, imageResource));
    state.module.AddFunction(spv::OpSampledImage, sampledType, sampledImage, image, samplerId);
    DecorateTableNonUniform(state, slotId, sampledImage);
    return sampledImage;
}

std::uint32_t StorageImageDescriptorPointer(SpirvEmitterState& state, std::uint32_t resource) {
    const auto& image = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (image.resourceClass != ImageResourceClass::Storage) {
        FailEmit("storage image descriptor requested for a non-storage image");
    }
    const auto kind = DescriptorBindingForImage(image);
    const auto arrayIndex = ResourceForDescriptor(state, kind, resource);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, ImageType(state, image));
    const auto variable = state.imageVariables.at(ImageBindingIndex(kind));
    return DescriptorElementPointer(state, pointerType, variable, arrayIndex, kind, resource, "storage image descriptor array was not emitted");
}

void EmitStorageImageWrite(SpirvEmitterState& state, std::uint32_t resource, std::uint32_t mipLod, std::uint32_t coord, std::uint32_t texel, std::uint32_t sample) {
    const auto& image = state.runtimeImage != nullptr ? *state.runtimeImage : state.program.Info().images.at(resource);
    if (image.resourceClass != ImageResourceClass::Storage) {
        FailEmit("storage image write requested for a non-storage image");
    }
    if (!image.atomic) {
        state.module.EmitCapability(spv::CapabilityStorageImageWriteWithoutFormat);
    }
    const auto kind = DescriptorBindingForImage(image);
    const auto arrayIndex = ResourceForDescriptor(state, kind, resource);
    const auto imageType = ImageType(state, image);
    const auto pointerType = state.module.Type(spv::OpTypePointer, spv::StorageClassUniformConstant, imageType);
    const auto variable = state.imageVariables.at(ImageBindingIndex(kind));
    const auto loadAt = [&](std::uint32_t index) {
        const auto pointer = DescriptorElementPointer(state, pointerType, variable, index, kind, resource, "storage image descriptor array was not emitted");
        const auto descriptor = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, imageType, descriptor, pointer);
        return descriptor;
    };
    if (RdnaImageDimensionInfoFor(image.dimension).multisampled != 0u) {
        if (sample == 0u || image.mipMode == ImageMipMode::DynamicStorage) FailEmit("multisampled storage image requires a sample and cannot have mip levels");
        state.module.AddFunction(spv::OpImageWrite, loadAt(arrayIndex), coord, texel, spv::ImageOperandsSampleMask, sample);
        return;
    }
    if (image.mipMode != ImageMipMode::DynamicStorage) {
        state.module.AddFunction(spv::OpImageWrite, loadAt(arrayIndex), coord, texel);
        return;
    }
    if (image.mipCount == 0u) {
        ExitDescriptorBindingFailure(state, kind, resource, "dynamic storage image has no mip descriptors");
    }
    const auto count = state.registrationModules ? RuntimeMipCount(state, resource) : state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::MipCountBase + resource, image.mipCount);
    const auto mergeLabel = state.module.AllocateId();
    std::vector<std::uint32_t> labels(image.mipCount);
    std::vector<std::uint32_t> words{spv::OpSwitch, mipLod, mergeLabel};
    for (std::uint32_t mip = 0; mip < image.mipCount; mip++) {
        labels[mip] = state.module.AllocateId();
        words.push_back(mip);
        words.push_back(labels[mip]);
    }
    state.module.AddFunction(spv::OpSelectionMerge, mergeLabel, spv::SelectionControlMaskNone);
    state.module.AddFunction(std::span<const std::uint32_t>(words));
    for (std::uint32_t mip = 0; mip < image.mipCount; mip++) {
        EmitLabel(state, labels[mip]);
        const auto active = Binary(state, spv::OpULessThan, TypeBool(state), ConstantU32(state, mip), count);
        const auto writeLabel = state.module.AllocateId();
        const auto nextLabel = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, nextLabel, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, active, writeLabel, nextLabel);
        EmitLabel(state, writeLabel);
        state.module.AddFunction(spv::OpImageWrite, loadAt(arrayIndex + mip), coord, texel);
        state.module.AddFunction(spv::OpBranch, nextLabel);
        EmitLabel(state, nextLabel);
        state.module.AddFunction(spv::OpBranch, mergeLabel);
    }
    EmitLabel(state, mergeLabel);
}

const RdnaImageDimensionInfo& RdnaImageDimensionInfoFor(RdnaImageDimension dimension) {
    for (const auto& info : ImageDimensions) {
        if (info.dimension == dimension) {
            return info;
        }
    }
    FailEmit("image dimension " + std::to_string(static_cast<std::uint32_t>(dimension)) + " is invalid");
}

}
