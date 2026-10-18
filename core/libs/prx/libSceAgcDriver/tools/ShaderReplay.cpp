// agc_shader_replay <shader_*.req>...: replays compute recompile requests the driver saved with
// APS5_DUMP_SHADERS=1, running the same resource analysis and recompile steps without the game.
#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/RequestMemoryView.hpp"
#include "Recompiler.hpp"
#include "ShaderDiskCache.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"

#if ANYPS5_ENABLE_SPIRV_TOOLS
#include "spirv-tools/libspirv.hpp"
#endif

#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <set>
#include <cstdlib>

namespace {

std::string ReadText(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream stream;
    stream << file.rdbuf();
    auto text = stream.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
    return text;
}

// The error text without the serialized request that Recompile appends.
std::string FirstLine(const std::string& text) {
    const auto request = text.find("\nRecompileRequest:");
    auto shown = request == std::string::npos ? text : text.substr(0, request);
    if (shown.size() > 2000) shown.resize(2000);
    return shown;
}

bool g_disassemble = false;
bool g_assembly = false;
bool g_memory = false;
bool g_spirv = false;
bool g_prepare = false;
bool g_materialize = false;
bool g_graph = false;
bool g_maintenance8 = false;
bool g_code = false;

bool Replay(const char* path) {
    auto request = ShaderRecompiler::RequestSerializer{}.Deserialize(ReadText(path));
    // APS5_GPU_RESOLVED=1: the native renderer's mode (SpirvTarget::gpuResolvedResources).
    static const bool gpuResolved = std::getenv("APS5_GPU_RESOLVED") != nullptr;
    if (gpuResolved) {
        request.request.target.gpuResolvedResources = true;
        request.request.target.registrationModules = true;
        request.request.target.nativeImageTable = 1u;
        request.request.target.nativeSamplerTable = 1u;
        if (std::getenv("APS5_SPANNED_WAVE") != nullptr) request.request.target.spannedWaves = true;
        // As the native renderer translates: FP32 denormals left to the host GPU (NativeFloatMode).
        if (request.request.context.floatMode && std::getenv("APS5_NATIVE_EXACT_DENORMALS") == nullptr) request.request.context.floatMode->floatMode |= 0x30u;
    }
    std::printf("%s: %zu code words, %zu user data, %zu memory regions, wave%u\n", path, request.request.shader.code.size(), request.request.context.userData.size(), request.request.context.memory.size(), request.request.context.waveSize);
    if (request.request.context.compute.has_value()) {
        const auto& compute = *request.request.context.compute;
        std::printf("  compute: threads %ux%ux%u, lds %u dwords, group ids %d%d%d, tg size %d, thread id components %u\n", compute.numThreads[0], compute.numThreads[1], compute.numThreads[2], compute.ldsSizeDwords, compute.groupIdEnable[0], compute.groupIdEnable[1], compute.groupIdEnable[2], compute.tgSizeEnable, compute.threadIdComponentCount);
        if (compute.PartialGroups()) std::printf("  partial groups: dispatch of %ux%ux%u threads\n", compute.partialThreads[0], compute.partialThreads[1], compute.partialThreads[2]);
    }
    if (request.request.graphics.has_value()) {
        const auto& graphics = *request.request.graphics;
        std::printf("  graphics: %zu linked programs, draw %u indices of %u bytes at 0x%llx, %u instances\n", graphics.linkedPrograms.size(), graphics.draw.indexCount, graphics.draw.indexElementBytes, static_cast<unsigned long long>(graphics.draw.indexAddress), graphics.draw.instanceCount);
        if (graphics.mesh.has_value()) {
            const auto& mesh = *graphics.mesh;
            std::printf("  mesh: input primitive %u, %u primitives / %u vertices per group, max %u vertices / %u primitives, %u threads, lds %u dwords, provoking %u, ESGS item %u\n", mesh.inputPrimitive, mesh.primitivesPerGroup, mesh.verticesPerGroup, mesh.maxVertices, mesh.maxPrimitives, mesh.threadsPerGroup, mesh.ldsSizeDwords, mesh.provokingVertex, mesh.esgsItemSize);
        }
    }
    if (g_memory) {
        // --mem: the captured inputs. User data words are printed; each memory region is written to
        // mem_<code address>_<guest address>.bin next to the request for inspection with other tools.
        const auto& context = request.request.context;
        for (std::size_t i = 0; i < context.userData.size(); ++i) std::printf("  user[%zu] = 0x%08x\n", i, context.userData[i]);
        for (const auto& region : context.memory) {
            char name[96];
            std::snprintf(name, sizeof(name), "mem_%llx_%llx.bin", static_cast<unsigned long long>(request.request.shader.codeAddress), static_cast<unsigned long long>(region.guestAddress));
            if (std::FILE* file = std::fopen(name, "wb")) {
                std::fwrite(region.bytes.data(), 1, region.bytes.size(), file);
                std::fclose(file);
            }
            std::printf("  region 0x%llx + 0x%zx -> %s\n", static_cast<unsigned long long>(region.guestAddress), region.bytes.size(), name);
        }
    }
    if (g_code) {
        std::string name(path);
        name = name.substr(name.find_last_of("/\\") + 1) + ".code";
        const auto& code = request.request.shader.code;
        std::ofstream file(name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(code.data()), static_cast<std::streamsize>(code.size() * sizeof(code[0])));
        if (!file) {
            std::printf("  could not write %s\n", name.c_str());
            return false;
        }
        std::printf("  code -> %s\n", name.c_str());
    }
    if (g_assembly) {
        const auto program = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(request.request.shader.code);
        // Raw first words carry what the text omits (branch offsets, waitcnt fields).
        for (const auto& instruction : program.instructions) std::printf("raw=%08x %s\n", instruction.rawWords[0], ShaderRecompiler::RdnaInstructionToString(instruction).c_str());
    }
    if (g_graph) {
        const auto program = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(request.request.shader.code);
        auto graph = ShaderRecompiler::GraphBuilder{}.Build(program);
        std::printf("control flow graph:\n%s", ShaderRecompiler::GraphToString(graph).c_str());
        try {
            ShaderRecompiler::Structurizer{}.Structurize(graph);
            std::printf("structured control flow graph:\n%s", ShaderRecompiler::GraphToString(graph).c_str());
        } catch (const std::exception& error) {
            std::printf("  structurization failed: %s\n", FirstLine(error.what()).c_str());
            return false;
        }
    }
    if (g_materialize && g_maintenance8) request.request.target.nonConstantImageOffsets = true;
    if (!g_materialize) {
        try {
            auto program = ShaderRecompiler::PrepareResourceProgram(request.request);
            constexpr ShaderRecompiler::ResourceMaterializer materializer;
            static_cast<void>(materializer.ExtractPlan(program));
        } catch (const std::exception& error) {
            std::printf("  resource analysis failed: %s\n", FirstLine(error.what()).c_str());
            return false;
        }
    }
    if (g_prepare || g_materialize) {
        const char* phase = "prepare";
        try {
            const auto handle = ShaderRecompiler::PrepareShader(request.request);
            std::printf("  prepared: %zu SPIR-V words\n", ShaderRecompiler::GetPreparedArtifact(*handle).spirv.size());
            if (g_materialize) {
                phase = "prepared invocation";
                const auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(request.request, handle);
                if (!invocation.has_value()) throw std::runtime_error("request does not match its prepared shader");
                ShaderRecompiler::RequestMemoryView memory(request.request.context.memory);
                const auto runtime = memory.MakeRuntime(request.request.context.userData, request.request.shader.codeAddress);
                phase = "capture";
                const auto capture = invocation->Capture(runtime);
                phase = "materialize";
                const auto result = invocation->Materialize(*capture);
                std::printf("  materialized: %zu SPIR-V words\n", result->spirv.size());
            }
            return true;
        } catch (const std::exception& error) {
            std::printf("  %s failed: %s\n", phase, FirstLine(error.what()).c_str());
            return false;
        }
    }
    try {
        auto program = ShaderRecompiler::PrepareResourceProgram(request.request);
        constexpr ShaderRecompiler::ResourceMaterializer materializer;
        static_cast<void>(materializer.ExtractPlan(program));
    } catch (const std::exception& error) {
        std::printf("  resource analysis failed: %s\n", FirstLine(error.what()).c_str());
        return false;
    }
    try {
        if (g_maintenance8) request.request.target.nonConstantImageOffsets = true;
        const auto result = ShaderRecompiler::Recompile(request.request);
        std::size_t roles[8] = {};
        for (const auto& binding : result.bindings) roles[std::min<std::size_t>(static_cast<std::size_t>(binding.role), 7)] += binding.count;
        std::printf("  recompiled: %zu SPIR-V words; bindings: buffers %zu images %zu samplers %zu gds %zu bda %zu fault %zu srt %zu data %zu\n", result.spirv.size(), roles[0], roles[1], roles[2], roles[3], roles[4], roles[5], roles[6], roles[7]);
        if (g_spirv) {
            std::string name(path);
            name = name.substr(name.find_last_of("/\\") + 1) + ".spv";
            std::ofstream(name, std::ios::binary).write(reinterpret_cast<const char*>(result.spirv.data()), static_cast<std::streamsize>(result.spirv.size() * sizeof(result.spirv[0])));
        }
        if (g_memory) {
            for (const auto& binding : result.bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
                for (std::size_t i = 0; i + 4 <= binding.guestDescriptor.size(); i += 4) {
                    const auto* v = binding.guestDescriptor.data() + i;
                    const bool written = i / 4 >= binding.bufferWritten.size() || binding.bufferWritten[i / 4];
                    std::printf("  buffer %zu: V# %08x %08x %08x %08x%s\n", i / 4, v[0], v[1], v[2], v[3], written ? " written" : "");
                }
            }
        }
#if ANYPS5_ENABLE_SPIRV_TOOLS
        if (g_disassemble) {
            spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
            std::string text;
            if (tools.Disassemble(result.spirv.data(), result.spirv.size(), &text, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)) std::printf("%s\n", text.c_str());
        }
#else
        if (g_disassemble) std::printf("  (--dis needs ANYPS5_ENABLE_SPIRV_TOOLS=ON)\n");
#endif
        return true;
    } catch (const std::exception& error) {
        std::printf("  recompile failed: %s\n", FirstLine(error.what()).c_str());
        return false;
    }
}

}

int main(int argc, char** argv) {
    if (argc == 4 && std::string_view(argv[1]) == "--deduplicate-list") {
        // Use the compiler's complete artifact key, including the code bytes,
        // layout, target, and switches. Resource snapshots can differ without
        // requiring another base shader artifact; hashes alone are not proof.
        if (const auto* lane = std::getenv("APS5_SINGLE_LANE"); lane && *lane) {
            std::fprintf(stderr, "deduplication requires APS5_SINGLE_LANE to be unset\n");
            return 2;
        }
        std::ifstream input(argv[2]);
        std::ofstream output(argv[3]);
        if (!input || !output) return 2;
        std::set<std::vector<std::byte>> keys;
        std::uint64_t count = 0;
        for (std::string path; std::getline(input, path);) {
            if (path.empty()) continue;
            try {
                auto request = ShaderRecompiler::RequestSerializer{}.Deserialize(ReadText(path.c_str()));
                const auto& r = request.request;
                const bool vertex = r.shader.stage == ShaderRecompiler::ShaderStage::Vertex ||
                    r.shader.stage == ShaderRecompiler::ShaderStage::Local;
                const auto subgroup = vertex && !r.target.vertexSubgroups ? 1u : r.target.subgroupSize;
                std::vector<std::byte> key;
                ShaderRecompiler::ShaderDiskCache::BuildKey(r, subgroup, key);
                if (keys.insert(std::move(key)).second) output << path << '\n';
                if (++count % 10000u == 0) {
                    std::printf("Deduplicated %llu requests into %zu artifact keys\n",
                        static_cast<unsigned long long>(count), keys.size());
                    std::fflush(stdout);
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "%s: %s\n", path.c_str(), FirstLine(error.what()).c_str());
                return 1;
            }
        }
        output.flush();
        if (!input.eof() || !output) return 1;
        std::printf("Deduplicated %llu requests into %zu artifact keys\n",
            static_cast<unsigned long long>(count), keys.size());
        return 0;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: agc_shader_replay [--dis] [--asm] [--cfg] [--mem] [--spv] [--prepare] [--materialize] [--code] [--maintenance8] <shader.req>...\n  the driver writes shader_<address>.req files when APS5_DUMP_SHADERS is set\n");
        return 2;
    }
    int failures = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dis") {
            g_disassemble = true;
            continue;
        }
        if (std::string(argv[i]) == "--asm") {
            g_assembly = true;
            continue;
        }
        if (std::string(argv[i]) == "--cfg") {
            g_graph = true;
            continue;
        }
        if (std::string(argv[i]) == "--maintenance8") {
            g_maintenance8 = true;
            continue;
        }
        if (std::string(argv[i]) == "--mem") {
            g_memory = true;
            continue;
        }
        if (std::string(argv[i]) == "--code") {
            g_code = true;
            continue;
        }
        if (std::string(argv[i]) == "--prepare") {
            g_prepare = true;
            continue;
        }
        if (std::string(argv[i]) == "--materialize") {
            g_materialize = true;
            continue;
        }
        if (std::string(argv[i]) == "--spv") {
            g_spirv = true;
            continue;
        }
        try {
            if (!Replay(argv[i])) ++failures;
        } catch (const std::exception& error) {
            std::printf("%s: could not load request: %s\n", argv[i], FirstLine(error.what()).c_str());
            ++failures;
        }
    }
    ShaderRecompiler::ShaderDiskCache::Flush();
    const auto cache = ShaderRecompiler::ShaderDiskCache::Totals();
    std::printf("Shader cache: %llu hits, %llu misses, %llu writes, %llu load failures, %llu write failures; replay failures %d\n",
        static_cast<unsigned long long>(cache.hits), static_cast<unsigned long long>(cache.misses),
        static_cast<unsigned long long>(cache.writes), static_cast<unsigned long long>(cache.loadFailures),
        static_cast<unsigned long long>(cache.writeFailures), failures);
    return failures == 0 ? 0 : 1;
}
