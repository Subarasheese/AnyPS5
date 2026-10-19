#include "Translation/DispatchInstructions.hpp"
#include "Translation/TranslationContext.hpp"
#include "Recompiler.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

void DispatchInstruction(IrBuilder& builder, const RdnaInstruction& instruction, const ControlFlowGraph& cfg, const TranslateOptions& options) {
    throw std::runtime_error("DispatchInstruction not implemented");
}

void TranslationContext::TranslateInstruction(const RdnaInstruction& decoded) {
    RdnaInstruction instruction = decoded;
    instruction.destination = destinationOperand(decoded);
    currentOpcode = instruction.op;
    currentProgramCounter = instruction.programCounter;
    if (instruction.op == RdnaOpcode::Unknown || instruction.op == RdnaOpcode::Count) {
        throw std::runtime_error("decoded opcode has no IR translation at pc " + std::to_string(instruction.programCounter));
    }
    if (instruction.op == RdnaOpcode::Unsupported) {
        throw std::runtime_error(instruction.unsupportedReason.empty() ? "unsupported decoded instruction at pc " + std::to_string(instruction.programCounter) : std::string(instruction.unsupportedReason));
    }
    checkRoundingUse(instruction);
    f32DenormalFlush = f32DenormalFlushFor(instruction);
    bool translated = false;
    switch (instruction.family) {
        case RdnaInstructionFamily::SOP1:
        case RdnaInstructionFamily::SOP2:
        case RdnaInstructionFamily::SOPK:
        case RdnaInstructionFamily::SOPC:
        case RdnaInstructionFamily::SOPP:
            translated = emitScalar(instruction);
            break;
        case RdnaInstructionFamily::VOP1:
        case RdnaInstructionFamily::VOP2:
        case RdnaInstructionFamily::VOP3:
        case RdnaInstructionFamily::VOP3P:
        case RdnaInstructionFamily::VOPC:
            translated = emitVector(instruction) || emitInterpolation(instruction);
            break;
        case RdnaInstructionFamily::SMEM:
        case RdnaInstructionFamily::MUBUF:
        case RdnaInstructionFamily::MTBUF:
        case RdnaInstructionFamily::FLAT:
        case RdnaInstructionFamily::DS:
        case RdnaInstructionFamily::MIMG:
            translated = emitMemory(instruction);
            break;
        case RdnaInstructionFamily::VINTRP:
            translated = emitInterpolation(instruction);
            break;
        case RdnaInstructionFamily::EXP:
            eXP(instruction);
            translated = true;
            break;
        default:
            break;
    }
    if (!translated) {
        throw std::runtime_error("opcode has no IR translation at pc " + std::to_string(instruction.programCounter));
    }
    if (const DebugProbe probe = DebugProbeConfig(); probe.enabled && instruction.programCounter == probe.programCounter) {
        program.Metadata().probeRegister = 255u;
        RdnaOperand source{};
        source.kind = RdnaOperandKind::VectorRegister;
        source.reg = probe.vgpr;
        RdnaOperand target{};
        target.kind = RdnaOperandKind::VectorRegister;
        target.reg = 255u;
        writeOperand(target, &readRawU32(source).Value());
    }
}

namespace {
std::atomic<bool> g_debugProbeActive{false};
}

void SetDebugProbeActive(bool active) {
    g_debugProbeActive.store(active);
}

bool DebugProbeActive() {
    return g_debugProbeActive.load();
}

namespace {
thread_local std::uint64_t g_probeProgramHash = 0;
}

std::uint64_t DebugProbeCodeHash() {
    static const std::uint64_t wanted = [] {
        const char* text = std::getenv("APS5_PROBE_CODE");
        return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull;
    }();
    return wanted;
}

void SetDebugProbeProgramHash(std::uint64_t hash) {
    g_probeProgramHash = hash;
}

std::uint64_t DebugProbeProgramHash() {
    return g_probeProgramHash;
}

std::uint32_t DebugProbeMarkAllTarget() {
    static const std::uint32_t target = [] {
        const char* text = std::getenv("APS5_PROBE_MARK_ALL");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : 0x100u;
    }();
    return target;
}

bool RayTracingStrict() {
    static const bool strict = [] {
        const char* text = std::getenv("APS5_RAYTRACING");
        return text != nullptr && std::strcmp(text, "strict") == 0;
    }();
    return strict;
}

bool RayTracingMiss() {
    static const bool miss = [] {
        const char* text = std::getenv("APS5_RAYTRACING");
        return text != nullptr && std::strcmp(text, "miss") == 0;
    }();
    return miss;
}

DebugProbe DebugProbeConfig() {
    static const DebugProbe parsed = [] {
        DebugProbe result;
        const char* text = std::getenv("APS5_PROBE");
        if (text == nullptr) return result;
        char* end = nullptr;
        result.programCounter = static_cast<std::uint32_t>(std::strtoul(text, &end, 16));
        if (end == nullptr || *end != ':') return result;
        result.vgpr = static_cast<std::uint32_t>(std::strtoul(end + 1, &end, 10));
        if (end != nullptr && *end == ':') result.shift = static_cast<std::uint32_t>(std::strtoul(end + 1, nullptr, 10));
        if (const char* mrt = std::getenv("APS5_PROBE_MRT")) result.mrt = static_cast<std::uint32_t>(std::strtoul(mrt, nullptr, 10));
        // APS5_PROBE_MARK=<mrt>: that color target gets a sentinel wherever the probed program drew,
        // so the probed pixels can be told from the pixels the other programs wrote.
        if (const char* mark = std::getenv("APS5_PROBE_MARK")) result.mark = static_cast<std::uint32_t>(std::strtoul(mark, nullptr, 10));
        result.enabled = result.vgpr < 255u;
        return result;
    }();
    DebugProbe probe = parsed;
    // A probe limited to one program (APS5_PROBE_CODE) needs no driver to mark the draw that uses it,
    // and leaves every other program translated as it is: the disk cache stays on, and only the probed
    // program's cache key carries the probe (ShaderDiskCache appendProbeKey).
    const auto wanted = DebugProbeCodeHash();
    probe.enabled = parsed.enabled && (wanted != 0 ? wanted == g_probeProgramHash : DebugProbeActive());
    return probe;
}

void DispatchInstruction(TranslationContext& context, const RdnaInstruction& instruction) {
    throw std::runtime_error("DispatchInstruction not implemented");
}

}
