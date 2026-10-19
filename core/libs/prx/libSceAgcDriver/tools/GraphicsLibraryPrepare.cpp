#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/StageLibraryCapture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include <chrono>
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
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: agc_graphics_library_prepare [--live] [--mesh-only] [--native-compute] [--native-graphics] <directory of captures or requests>...\n       agc_graphics_library_prepare --native-journal\n");
        return 2;
    }
    try {
        const bool nativeCompute = std::any_of(argv + 1, argv + argc, [](const char* argument) { return std::string_view(argument) == "--native-compute"; });
        const bool nativeGraphics = std::any_of(argv + 1, argv + argc, [](const char* argument) { return std::string_view(argument) == "--native-graphics"; });
        const bool nativeJournal = std::any_of(argv + 1, argv + argc, [](const char* argument) { return std::string_view(argument) == "--native-journal"; });
        auto driver = std::make_shared<AgcDriver::VulkanDevice>();
        auto& device = *driver;
        auto* native = (nativeCompute || nativeGraphics || nativeJournal) ? &AgcDriver::Native::NativeDevice::Create(driver) : nullptr;
        if (nativeJournal) {
            const auto replay = native->ReplayPipelineJournal(true);
            if (replay == nullptr) throw std::runtime_error("native pipeline journal preparation is disabled");
            replay->Wait();
            const auto totals = AgcDriver::Graphics::PipelineJournalTotals();
            device.SavePipelineCache();
            std::printf("Native journal: %llu loaded, %llu graphics built, %llu present, %llu skipped, %llu rejected, %llu failed in %.0f ms\n",
                static_cast<unsigned long long>(totals.loadedRecipes), static_cast<unsigned long long>(totals.replayedGraphics),
                static_cast<unsigned long long>(totals.replayPresent), static_cast<unsigned long long>(totals.replaySkipped),
                static_cast<unsigned long long>(totals.replayRejected), static_cast<unsigned long long>(totals.replayFailed), totals.replayMs);
            return totals.replayFailed == 0 && totals.writeFailures == 0 ? 0 : 1;
        }
        std::uint64_t prepared = 0;
        std::uint64_t liveOnly = 0;
        std::set<std::uint64_t> registrationModules;
        std::set<std::uint64_t> liveModules;
        bool live = false;
        bool meshOnly = false;
        if (nativeCompute || nativeGraphics) {
            std::vector<std::filesystem::path> files;
            for (int argument = 1; argument < argc; ++argument) {
                if (std::string_view(argv[argument]).starts_with("--")) continue;
                for (const auto& file : std::filesystem::directory_iterator(argv[argument])) {
                    if (file.is_regular_file() && file.path().extension() == ".req") files.push_back(file.path());
                }
            }
            std::sort(files.begin(), files.end());
            std::mutex mutex;
            std::atomic<std::size_t> next{0};
            std::exception_ptr failure;
            const auto prepareFile = [&](const std::filesystem::path& path) {
                std::vector<std::byte> requestBytes;
                if (!ShaderRecompiler::ReadWholeFile(path, requestBytes)) throw std::runtime_error("cannot read request: " + path.string());
                decltype(ShaderRecompiler::RequestSerializer{}.Deserialize(std::string_view{})) request;
                try {
                    request = ShaderRecompiler::RequestSerializer{}.Deserialize(std::string_view(reinterpret_cast<const char*>(requestBytes.data()), requestBytes.size()));
                } catch (const std::exception& error) {
                    // A capture cut short when the game was stopped.
                    std::lock_guard lock(mutex);
                    std::fprintf(stderr, "Skipped unreadable request %s: %s\n", path.filename().string().c_str(), error.what());
                    return;
                }
                // APS5_PREPARE_CODE_HASH=1: print each request's program hash (the identity APS5_PROBE_CODE
                // selects, which survives the address changes between runs) and translate nothing.
                static const bool hashOnly = std::getenv("APS5_PREPARE_CODE_HASH") != nullptr;
                if (hashOnly) {
                    const auto& code = request.request.shader.code;
                    const auto head = code.subspan(0, std::min<std::size_t>(std::size_t{16}, code.size()));
                    std::lock_guard lock(mutex);
                    std::printf("%s stage %d hash %016llx\n", path.filename().string().c_str(), static_cast<int>(request.request.shader.stage),
                        static_cast<unsigned long long>(ShaderRecompiler::HashBytes(std::as_bytes(head))));
                    return;
                }
                const auto stage = request.request.shader.stage;
                if (nativeCompute) {
                    if (stage != ShaderRecompiler::ShaderStage::Compute) return;
                    request.request.target = native->ComputeTarget(request.request.context.waveSize);
                    if (request.request.context.floatMode && std::getenv("APS5_NATIVE_EXACT_DENORMALS") == nullptr) request.request.context.floatMode->floatMode |= 0x30u;
                    const auto result = ShaderRecompiler::Recompile(request.request);
                    {
                        std::lock_guard lock(mutex);
                        if (!liveModules.insert(result.PipelineVariantId()).second) return;
                    }
                    if (const char* directory = std::getenv("APS5_PREPARE_SPV_DIR")) {
                        if (std::FILE* file = std::fopen((std::filesystem::path(directory) / path.filename().replace_extension(".spv")).c_str(), "wb")) {
                            std::fwrite(result.spirv.data(), 4, result.spirv.size(), file);
                            std::fclose(file);
                        }
                    }
                    std::string reason;
                    if (!native->PrebuildCompute(result, reason)) throw std::runtime_error("native compute preparation: " + reason);
                    std::lock_guard lock(mutex);
                    std::printf("Prepared native compute pipeline %llu: %s\n", static_cast<unsigned long long>(++prepared), path.filename().string().c_str());
                    std::fflush(stdout);
                    return;
                }
                if (stage != ShaderRecompiler::ShaderStage::Vertex && stage != ShaderRecompiler::ShaderStage::Fragment && stage != ShaderRecompiler::ShaderStage::Mesh) return;
                request.request.target = native->GraphicsTarget();
                if (request.request.context.floatMode && std::getenv("APS5_NATIVE_EXACT_DENORMALS") == nullptr) request.request.context.floatMode->floatMode |= 0x30u;
                const auto began = std::chrono::steady_clock::now();
                const auto result = ShaderRecompiler::Recompile(request.request);
                const auto translated = std::chrono::steady_clock::now();
                {
                    std::lock_guard lock(mutex);
                    if (!liveModules.insert(result.PipelineVariantId()).second) return;
                }
                // APS5_PREPARE_CODE_DIR=<dir> saves each stage's guest program code, for disassembly.
                if (const char* directory = std::getenv("APS5_PREPARE_CODE_DIR")) {
                    if (std::FILE* file = std::fopen((std::filesystem::path(directory) / path.filename().replace_extension(".code")).c_str(), "wb")) {
                        std::fwrite(request.request.shader.code.data(), 4, request.request.shader.code.size(), file);
                        std::fclose(file);
                    }
                }
                // APS5_PREPARE_SPV_DIR=<dir> saves each translated stage's SPIR-V.
                if (const char* directory = std::getenv("APS5_PREPARE_SPV_DIR")) {
                    if (std::FILE* file = std::fopen((std::filesystem::path(directory) / path.filename().replace_extension(".spv")).c_str(), "wb")) {
                        std::fwrite(result.spirv.data(), 4, result.spirv.size(), file);
                        std::fclose(file);
                    }
                }
                const auto mesh = stage == ShaderRecompiler::ShaderStage::Mesh && request.request.graphics ? request.request.graphics->mesh : std::nullopt;
                const bool built = native->PrebuildGraphicsLibrary(result, stage, request.request.context.waveSize, request.request.layout.pushConstantOffsetBytes, mesh);
                const auto finished = std::chrono::steady_clock::now();
                const auto ms = [](auto from, auto to) { return std::chrono::duration<double, std::milli>(to - from).count(); };
                std::lock_guard lock(mutex);
                std::printf("Stage %s: %zu SPIR-V words, translate %.0f ms, library %.0f ms\n", path.filename().string().c_str(), result.spirv.size(), ms(began, translated), ms(translated, finished));
                if (!built) {
                    ++liveOnly;
                    std::fprintf(stderr, "Translated native graphics request requires live pipeline state: %s\n", path.filename().string().c_str());
                    return;
                }
                std::printf("Prepared native graphics library %llu: %s\n", static_cast<unsigned long long>(++prepared), path.filename().string().c_str());
                if (prepared % 32u == 0u) device.SavePipelineCache();
                std::fflush(stdout);
            };
            const auto work = [&] {
                for (;;) {
                    const auto index = next.fetch_add(1);
                    if (index >= files.size()) return;
                    {
                        std::lock_guard lock(mutex);
                        if (failure) return;
                    }
                    try {
                        prepareFile(files[index]);
                    } catch (...) {
                        std::lock_guard lock(mutex);
                        if (!failure) failure = std::current_exception();
                        return;
                    }
                }
            };
            const char* requested = std::getenv("APS5_PREPARE_THREADS");
            const unsigned threads = requested != nullptr ? std::max(1, std::atoi(requested)) : std::max(1u, std::thread::hardware_concurrency() / 2u);
            std::vector<std::thread> pool;
            for (unsigned i = 0; i < threads; ++i) pool.emplace_back(work);
            for (auto& thread : pool) thread.join();
            if (failure) std::rethrow_exception(failure);
        }
        for (int argument = 1; !nativeCompute && !nativeGraphics && argument < argc; ++argument) {
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
                if (!file.is_regular_file()) continue;
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
