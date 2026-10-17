#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/StageLibraryCapture.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include <memory>
#include <cstdlib>
#include <algorithm>
#include "ShaderCacheDirectory.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <set>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: agc_graphics_library_prepare [--live] [--mesh-only] [--native-compute] <directory of captures or requests>...\n");
        return 2;
    }
    try {
        const bool nativeCompute = std::any_of(argv + 1, argv + argc, [](const char* argument) { return std::string_view(argument) == "--native-compute"; });
        auto driver = std::make_shared<AgcDriver::VulkanDevice>();
        auto& device = *driver;
        auto* native = nativeCompute ? &AgcDriver::Native::NativeDevice::Create(driver) : nullptr;
        std::uint64_t prepared = 0;
        std::uint64_t liveOnly = 0;
        std::set<std::uint64_t> registrationModules;
        std::set<std::uint64_t> liveModules;
        bool live = false;
        bool meshOnly = false;
        for (int argument = 1; argument < argc; ++argument) {
            if (std::string_view(argv[argument]) == "--native-compute") continue;
            if (std::string_view(argv[argument]) == "--live") { live = true; continue; }
            if (std::string_view(argv[argument]) == "--mesh-only") { meshOnly = true; continue; }
            for (const auto& file : std::filesystem::directory_iterator(argv[argument])) {
                if (file.is_regular_file() && file.path().extension() == ".req") {
                    std::vector<std::byte> requestBytes;
                    if (!ShaderRecompiler::ReadWholeFile(file.path(), requestBytes))
                        throw std::runtime_error("cannot read request: " + file.path().string());
                    auto request = ShaderRecompiler::RequestSerializer{}.Deserialize(
                        std::string_view(reinterpret_cast<const char*>(requestBytes.data()), requestBytes.size()));
                    const auto stage = request.request.shader.stage;
                    if (nativeCompute) {
                        if (stage != ShaderRecompiler::ShaderStage::Compute) continue;
                        request.request.target = native->ComputeTarget(request.request.context.waveSize);
                        if (request.request.context.floatMode && std::getenv("APS5_NATIVE_EXACT_DENORMALS") == nullptr) request.request.context.floatMode->floatMode |= 0x30u;
                        const auto result = ShaderRecompiler::Recompile(request.request);
                        if (!liveModules.insert(result.PipelineVariantId()).second) continue;
                        std::string failure;
                        if (!native->PrebuildCompute(result, failure)) throw std::runtime_error("native compute preparation: " + failure);
                        std::printf("Prepared native compute pipeline %llu: %s\n", static_cast<unsigned long long>(++prepared), file.path().filename().string().c_str());
                        std::fflush(stdout);
                        continue;
                    }
                    const bool fragment = stage == ShaderRecompiler::ShaderStage::Fragment;
                    const bool mesh = stage == ShaderRecompiler::ShaderStage::Mesh;
                    if (meshOnly && !mesh) continue;
                    if (!fragment && (!live || (stage != ShaderRecompiler::ShaderStage::Vertex && !mesh))) continue;
                    if (live) {
                        // Registration requests only contain the code/header.
                        // Draw captures additionally own the resource-table bytes.
                        if (request.request.context.memory.size() <= 2u) continue;
                        const auto result = ShaderRecompiler::Recompile(request.request);
                        if (!liveModules.insert(result.PipelineVariantId()).second) continue;
                        AgcDriver::Graphics::ShaderStages stages{AgcDriver::Graphics::ShaderPath::Vertex, 0u,
                            fragment ? 32u : request.request.context.waveSize,
                            fragment ? request.request.context.waveSize : 32u, {}, {}};
                        if (mesh) {
                            if (!request.request.graphics || !request.request.graphics->mesh)
                                throw std::runtime_error("mesh request has no graphics configuration");
                            stages.mesh = request.request.graphics->mesh;
                        }
                        const std::array<AgcDriver::Graphics::CompiledShader, 1> shaders{{{stage, &result, request.request.layout.pushConstantOffsetBytes}}};
                        if (!device.PrebuildGraphicsLibraries(stages, shaders))
                            throw std::runtime_error("graphics pipeline libraries are unavailable");
                        std::printf("Prepared live library %llu: %s\n", static_cast<unsigned long long>(++prepared), file.path().filename().string().c_str());
                        std::fflush(stdout);
                        continue;
                    }
                    const auto handle = ShaderRecompiler::PrepareShader(request.request);
                    const auto identity = ShaderRecompiler::GetPreparedArtifact(*handle).variantId;
                    if (!registrationModules.insert(identity).second) continue;
                    const auto result = ShaderRecompiler::RegistrationResult(request.request, *handle);
                    if (result == nullptr) {
                        ++liveOnly;
                        continue;
                    }
                    AgcDriver::Graphics::ShaderStages stages{AgcDriver::Graphics::ShaderPath::Vertex, 0u, 32u, request.request.context.waveSize, {}, {}};
                    const std::array<AgcDriver::Graphics::CompiledShader, 1> shaders{{{ShaderRecompiler::ShaderStage::Fragment, result.get(), request.request.layout.pushConstantOffsetBytes}}};
                    if (!device.PrebuildGraphicsLibraries(stages, shaders))
                        throw std::runtime_error("graphics pipeline libraries are unavailable");
                    std::printf("Prepared registration library %llu: %s\n", static_cast<unsigned long long>(++prepared), file.path().filename().string().c_str());
                    std::fflush(stdout);
                    continue;
                }
                if (nativeCompute || !file.is_regular_file()) continue;
                const auto filename = file.path().filename().string();
                const bool fragmentCapture = filename.ends_with(".fragment.bin");
                const bool vertexCapture = filename.ends_with(".vertex.bin");
                const bool meshCapture = filename.ends_with(".mesh.bin");
                if (meshOnly && !meshCapture) continue;
                if (!fragmentCapture && !vertexCapture && !meshCapture) continue;
                std::vector<std::byte> bytes;
                if (!ShaderRecompiler::ReadWholeFile(file.path(), bytes) || bytes.size() < sizeof(AgcDriver::Graphics::StageLibraryCaptureHeader))
                    throw std::runtime_error("truncated fragment capture: " + file.path().string());
                AgcDriver::Graphics::StageLibraryCaptureHeader header;
                std::memcpy(&header, bytes.data(), sizeof(header));
                auto payload = std::span<const std::byte>(bytes).subspan(sizeof(header));
                const auto stage = meshCapture ? ShaderRecompiler::ShaderStage::Mesh :
                    vertexCapture ? ShaderRecompiler::ShaderStage::Vertex : ShaderRecompiler::ShaderStage::Fragment;
                if (header.magic != header.Magic || header.sourceVersion != ShaderRecompiler::ShaderDiskCache::SourceVersion() ||
                    header.stage != static_cast<std::uint32_t>(stage) ||
                    payload.size() != header.payloadBytes || ShaderRecompiler::HashBytes(payload) != header.payloadHash ||
                    (header.vertexWave != 32u && header.vertexWave != 64u) || (header.fragmentWave != 32u && header.fragmentWave != 64u))
                    throw std::runtime_error("incompatible fragment capture: " + file.path().string());
                std::optional<ShaderRecompiler::MeshConfiguration> mesh;
                if (meshCapture) {
                    if (payload.size() < sizeof(ShaderRecompiler::MeshConfiguration))
                        throw std::runtime_error("truncated mesh configuration: " + file.path().string());
                    mesh.emplace();
                    std::memcpy(&*mesh, payload.data(), sizeof(*mesh));
                    payload = payload.subspan(sizeof(*mesh));
                }
                ShaderRecompiler::RecompileResult result;
                if (!ShaderRecompiler::ShaderDiskCache::DecodeResult(payload, result))
                    throw std::runtime_error("invalid fragment result: " + file.path().string());
                // Capture IDs belong to the original process. Give each replay a
                // distinct ID so the in-memory library cache cannot alias them.
                result.variantId = (1ull << 63u) + prepared + 1;
                result.specializationId = 0;
                AgcDriver::Graphics::ShaderStages stages{AgcDriver::Graphics::ShaderPath::Vertex, 0u, header.vertexWave, header.fragmentWave, mesh, {}};
                const std::array<AgcDriver::Graphics::CompiledShader, 1> shaders{{{stage, &result, header.pushOffset}}};
                if (!device.PrebuildGraphicsLibraries(stages, shaders))
                    throw std::runtime_error("graphics pipeline libraries are unavailable");
                std::printf("Prepared graphics library %llu (stage %u): %s\n", static_cast<unsigned long long>(++prepared), static_cast<unsigned>(stage), file.path().filename().string().c_str());
                std::fflush(stdout);
            }
        }
        ShaderRecompiler::ShaderDiskCache::Flush();
        device.SavePipelineCache();
        std::printf("Prepared %llu pipelines/libraries; %llu registration modules require live bindings\n",
                    static_cast<unsigned long long>(prepared), static_cast<unsigned long long>(liveOnly));
        return prepared == 0 ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Graphics library preparation failed: %s\n", error.what());
        return 1;
    }
}
