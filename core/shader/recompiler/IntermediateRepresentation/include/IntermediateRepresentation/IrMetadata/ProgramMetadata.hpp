#ifndef CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_PROGRAMMETADATA_HPP
#define CORE_SHADER_RECOMPILIER_INTERMEDIATEREPRESENTATION_INCLUDE_INTERMEDIATEREPRESENTATION_IRMETADATA_PROGRAMMETADATA_HPP

#include "ControlFlow/ControlFlowGraph.hpp"
#include "IntermediateRepresentation/IrValue.hpp"
#include "IntermediateRepresentation/IrMetadata/ControlFlowInfo.hpp"
#include "IntermediateRepresentation/IrMetadata/DescriptorBinding.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourceInfo.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace ShaderRecompiler {

struct IrProgramMetadata {
    FailureKind cfgFailureKind = FailureKind::None;
    std::string failureReason;
    std::vector<BlockInfo> blockInfo;
    std::vector<ExportInfo> exportInfo;
    std::uint32_t pixelLinearInputs = 0;
    std::uint32_t pixelPerspectiveInputs = 0;
    std::uint32_t pixelVertexOrderInputs = 0;
    bool barycentricEmulation = false;
    // SpirvTarget::gpuResolvedResources: no SRT read is folded into the CPU-flattened table and
    // every buffer takes its V# at run time (the shader resolves its resources on the GPU).
    bool gpuResolvedResources = false;
    // With gpuResolvedResources: the SRT reads a descriptor (T#, S#, V#, address) is built from are
    // folded into the flattened table, which the driver fills at the draw's call site.
    bool descriptorSlots = false;
    bool hostFloat = false;
    bool invalidatesCaches = false;
    std::vector<IrValue*> dynamicReads;
    bool shaderInfoComplete = false;
    IrBindingLayout bindings;
    bool bindingLayoutComplete = false;
    std::uint32_t tessellationLocalStride = 0;
    std::uint32_t vectorWindowBase = 0;
    std::uint32_t vectorWindowSize = 0;
    // The scratch vector register a debug probe carries its traced value in (DebugProbe): it is never
    // addressed relatively, so it stays a plain register even in a program with a window.
    std::uint32_t probeRegister = 0x100;
    bool vectorWindowIndexedWrites = false;
    std::uint32_t tessellationControlStride = 0;
    std::uint32_t tessellationPatchStride = 0;
};

}

#endif
