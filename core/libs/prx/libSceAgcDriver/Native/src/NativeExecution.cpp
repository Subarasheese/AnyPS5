// Native backend submission executor.
//
// The emulating backend turns every draw and dispatch into Vulkan work. The native backend
// renders GT7's frame itself, so a submission only needs the effects the game's CPU code
// observes: register state, memory stores (WRITE_DATA, RELEASE_MEM, COPY_DATA, DMA_DATA),
// waits on memory, conditional execution, end-of-pipe interrupts and flips. All of these are
// executed here, in order, on the queue's worker thread. Draw and dispatch packets are
// counted and skipped.

#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <condition_variable>
#include <vector>
#include <mutex>
#include <set>

namespace AgcDriver::DriverDetail {

namespace {

bool DispatchOpcode(std::uint32_t opcode) {
    return opcode == 0x15 || opcode == 0x16;
}

// Packets the emulating executor also leaves alone (see Driver::execute).
bool IgnoredOpcode(std::uint32_t opcode) {
    return opcode == 0x42 || opcode == 0x46 || opcode == 0x58;
}

// Polls a WAIT_REG_MEM like the command processor. A wait still unsatisfied after two seconds
// is reported once with the packets that preceded it, which names the GPU work that would
// have produced the awaited value.
void WaitWithReport(const Submission& submission, std::size_t at, std::span<const std::uint32_t> packet) {
    const auto start = std::chrono::steady_clock::now();
    bool reported = false;
    while (!Pm4::WaitSatisfied(packet)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        if (reported || std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) continue;
        reported = true;
        const auto address = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
        std::uint64_t now = 0;
        std::memcpy(&now, reinterpret_cast<const void*>(address), packet.size() > 7 && ((packet[0] >> 8u) & 0xffu) == 0x93u ? 8 : 4);
        std::fprintf(stderr, "[native] queue 0x%x submission %llu: wait at dword %zu on 0x%llx (value now 0x%llx) unsatisfied after 2 s; words:",
                     submission.queue, static_cast<unsigned long long>(submission.serial), at, static_cast<unsigned long long>(address), static_cast<unsigned long long>(now));
        for (std::size_t i = 1; i < packet.size(); ++i) std::fprintf(stderr, " %08x", packet[i]);
        std::fprintf(stderr, "\n[native]   preceding packets:\n");
        std::vector<std::size_t> starts;
        for (std::size_t cursor = 0; cursor < at;) {
            starts.push_back(cursor);
            cursor += Pm4::PacketWords(submission.commands[cursor]);
        }
        const std::size_t first = starts.size() > 40 ? starts.size() - 40 : 0;
        for (std::size_t k = first; k < starts.size(); ++k) {
            const auto cursor = starts[k];
            const auto header = submission.commands[cursor];
            const auto words = Pm4::PacketWords(header);
            std::fprintf(stderr, "[native]     %6zu %s", cursor, Pm4::Name(header).c_str());
            for (std::size_t i = 1; i < words && i < 9; ++i) std::fprintf(stderr, " %08x", submission.commands[cursor + i]);
            std::fprintf(stderr, "\n");
        }
    }
}

// Native compute: each program is translated once per compute configuration (GPU-resolved
// resources), its shader data block filled per dispatch with the dispatch's user data.
struct NativeProgram {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::string failure;
};

struct NativeCompute {
    std::mutex mutex;
    std::condition_variable ready;
    std::unordered_map<std::uint64_t, NativeProgram> programs;
    std::map<std::string, std::uint64_t> fallbacks;
    std::uint64_t native = 0;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

NativeCompute& Compute() {
    static NativeCompute compute;
    return compute;
}

void CountFallback(const std::string& reason) {
    auto& compute = Compute();
    std::lock_guard lock(compute.mutex);
    if (compute.fallbacks[reason]++ == 0) std::fprintf(stderr, "[native] compute dispatch not native: %s\n", reason.c_str());
}

void ReportCompute() {
    auto& compute = Compute();
    std::lock_guard lock(compute.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - compute.reported < std::chrono::seconds(5)) return;
    compute.reported = now;
    std::uint64_t skipped = 0;
    for (const auto& [reason, count] : compute.fallbacks) skipped += count;
    std::fprintf(stderr, "[native] compute: %llu native dispatches, %llu skipped, %zu programs\n", static_cast<unsigned long long>(compute.native), static_cast<unsigned long long>(skipped), compute.programs.size());
}

// The FLOAT_MODE a native translation runs with: GT7's rounding, with FP32 denormals left to the
// PC GPU (FP32_DENORM_MODE 3) as a PC build of the shaders would; the GPU flushing denormals would
// otherwise be emulated around every float operation (NVIDIA offers no FP32 flush-to-zero mode),
// which multiplies the size of GT7's large shaders for values below 1e-38. APS5_NATIVE_EXACT_DENORMALS=1
// keeps the emulation.
std::uint32_t NativeFloatMode(std::uint32_t mode) {
    static const bool exact = std::getenv("APS5_NATIVE_EXACT_DENORMALS") != nullptr;
    return exact ? mode : mode | 0x30u;
}

// A stage's shader data block for one dispatch or draw: the translation's own block (its runtime
// state, e.g. vertex component selectors) with the user-data words replaced by this invocation's.
// `pushOffset` is where the stage's push constants start; registers count from `firstUserSgpr`.
std::vector<std::uint32_t> ShaderDataFor(const ShaderRecompiler::RecompileResult& shader, std::uint32_t pushOffset, std::span<const std::uint32_t> userData, std::uint32_t firstUserSgpr) {
    std::vector<std::uint32_t> data(shader.shaderDataDwords, 0u);
    if (shader.pushDataStartDword != ~0u) {
        const auto start = static_cast<std::size_t>(shader.pushDataStartDword) * 4u;
        if (start >= pushOffset) {
            const auto offset = start - pushOffset;
            const auto bytes = std::min(data.size() * 4u, shader.pushConstants.size() > offset ? shader.pushConstants.size() - offset : std::size_t{0});
            if (bytes != 0) std::memcpy(data.data(), shader.pushConstants.data() + offset, bytes);
        }
    } else {
        for (const auto& binding : shader.bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::ShaderData) continue;
            std::copy_n(binding.guestDescriptor.begin(), std::min(binding.guestDescriptor.size(), data.size()), data.begin());
        }
    }
    for (std::size_t j = 0; j < shader.shaderDataUserRegisters.size() && j < data.size(); ++j) {
        const auto reg = shader.shaderDataUserRegisters[j];
        if (reg >= firstUserSgpr && reg - firstUserSgpr < userData.size()) data[j] = userData[reg - firstUserSgpr];
    }
    return data;
}

// Capture native stage requests, including live linkage and user data, so a
// faulting pixel program can be inspected offline with agc_shader_replay.
void CaptureNativeRequest(const ShaderRecompiler::RecompileRequest& request,
                          const std::vector<std::uint64_t>& key) {
    static const char* directory = std::getenv("APS5_NATIVE_REQUEST_DIR");
    static const std::set<std::uint64_t> selected = [] {
        std::set<std::uint64_t> programs;
        if (const char* text = std::getenv("APS5_NATIVE_REQUEST_PROGRAMS")) {
            while (*text) {
                char* end = nullptr;
                const auto address = std::strtoull(text, &end, 0);
                if (end == text) break;
                programs.insert(address);
                text = *end == ',' ? end + 1 : end;
            }
        }
        return programs;
    }();
    if (directory == nullptr || (!selected.empty() && !selected.contains(request.shader.codeAddress))) return;
    try {
        std::filesystem::create_directories(directory);
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (const auto word : key) hash = (hash ^ word) * 0x100000001b3ull;
        char name[96];
        std::snprintf(name, sizeof(name), "stage_%llx_%llx.req", static_cast<unsigned long long>(request.shader.codeAddress), static_cast<unsigned long long>(hash));
        std::ofstream file(std::filesystem::path(directory) / name);
        file << ShaderRecompiler::RequestSerializer{}.Serialize(request);
        if (!file) throw std::runtime_error("request file write failed");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[native] stage request capture failed: %s\n", error.what());
    }
}

// Native draws: each draw's stages are translated once per configuration (the recompiler's
// prepared-shader key: program, stage inputs and outputs, linkage), with resources resolved on the
// GPU; the draw records into the native batch with its pipeline state decoded from GT7's registers.
struct NativeGraphicsStage {
    Native::RecordedProgram program;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::string failure;
};

struct NativeGraphics {
    std::mutex mutex;
    std::condition_variable ready;
    // Per-draw CPU time by phase (microseconds), reported with the draw counts.
    std::array<double, 5> phaseUs{};
    std::map<std::vector<std::uint64_t>, NativeGraphicsStage> stages;
    // RECTLIST draws: the generated tessellation pair per vertex/pixel translation pair.
    std::map<std::pair<const void*, const void*>, std::shared_ptr<const ShaderRecompiler::RectListShaders>> rectangles;
    std::map<std::string, std::uint64_t> fallbacks;
    std::uint64_t native = 0;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

NativeGraphics& GraphicsWork() {
    static NativeGraphics graphics;
    return graphics;
}

void CountDrawFallback(const std::string& reason) {
    auto& graphics = GraphicsWork();
    std::lock_guard lock(graphics.mutex);
    if (graphics.fallbacks[reason]++ == 0) std::fprintf(stderr, "[native] draw not native: %s\n", reason.c_str());
}

void ReportGraphics() {
    auto& graphics = GraphicsWork();
    std::lock_guard lock(graphics.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - graphics.reported < std::chrono::seconds(5)) return;
    graphics.reported = now;
    std::uint64_t skipped = 0;
    for (const auto& [reason, count] : graphics.fallbacks) skipped += count;
    const double draws = std::max<double>(1.0, static_cast<double>(graphics.native + skipped));
    std::fprintf(stderr, "[native] draws: %llu native, %llu skipped, %zu stage translations; us per draw: state %.1f, programs %.1f, stage keys %.1f, record %.1f, total %.1f\n", static_cast<unsigned long long>(graphics.native), static_cast<unsigned long long>(skipped), graphics.stages.size(), graphics.phaseUs[0] / draws, graphics.phaseUs[1] / draws, graphics.phaseUs[2] / draws, graphics.phaseUs[3] / draws, graphics.phaseUs[4] / draws);
}

bool DrawNatively(QueueState& queue, std::span<const std::uint32_t> packet) {
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;
    auto* device = Native::NativeDevice::Get();
    if (device == nullptr) {
        CountDrawFallback("no native device yet");
        return false;
    }
    using Clock = std::chrono::steady_clock;
    struct Phases {
        Clock::time_point start = Clock::now(), last = start;
        std::array<double, 5> us{};
        void mark(std::size_t phase) {
            const auto now = Clock::now();
            us[phase] += std::chrono::duration<double, std::micro>(now - last).count();
            last = now;
        }
        ~Phases() {
            us[4] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            auto& graphics = GraphicsWork();
            std::lock_guard lock(graphics.mutex);
            for (std::size_t i = 0; i < us.size(); ++i) graphics.phaseUs[i] += us[i];
        }
    } phases;
    Pm4::DrawParameters draw{};
    Graphics::State state{};
    try {
        draw = Pm4::ResolveDraw(packet, queue);
        if (draw.indirect) {
            CountDrawFallback("indirect draw");
            return false;
        }
        if (draw.indexCount == 0 || draw.instanceCount == 0) return true;
        state = Graphics::DecodeState(queue, false, true);
    } catch (const std::exception& error) {
        CountDrawFallback(std::string("state: ") + error.what());
        return false;
    }
    if (state.stages.path == Graphics::ShaderPath::Tessellation || state.stages.path == Graphics::ShaderPath::TessellationGeometry) {
        CountDrawFallback("tessellation");
        return false;
    }
    // No pixel program (depth-only passes): the draw runs without a fragment stage when the
    // hardware would write nothing a pixel shader produces.
    const bool pixelless = Graphics::PixelProgramSkipped(queue);
    if (pixelless) {
        const auto rejection = Graphics::NullPixelProgramRejection(queue);
        if (!rejection.empty()) {
            CountDrawFallback("draw without a pixel program: " + rejection);
            return false;
        }
    }
    phases.mark(0);
    // GT7's programs for the draw, as the driver decodes them (DecodeGraphicsPrograms).
    struct Program {
        Native::RecordedProgram recorded;
        ShaderRecompiler::ShaderBinary binary;
        Stage stage;
        Role role;
        std::uint32_t userDataBase;
        std::uint32_t firstUserSgpr;
        std::uint32_t rsrc1;
        std::vector<std::uint32_t> userData;
    };
    std::vector<Program> programs;
    const auto programAddress = [&](std::uint32_t base) {
        return (static_cast<std::uint64_t>(readRegister(queue.shader, base)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, base + 1) & 0xffu) << 40u);
    };
    const auto add = [&](std::uint32_t base, Stage stage, Role role, std::uint32_t rsrc2Register, std::uint32_t userBase, std::uint32_t firstUser) {
        const auto address = programAddress(base);
        Program program{};
        if (!Native::FindRecordedShader(address, program.recorded)) return false;
        const auto offset = static_cast<std::size_t>((address - program.recorded.codeAddress) / sizeof(std::uint32_t));
        program.binary = {stage, address, std::span(*program.recorded.code).subspan(offset), program.recorded.headerAddress, std::span(*program.recorded.header)};
        program.stage = stage;
        program.role = role;
        program.userDataBase = userBase;
        program.firstUserSgpr = firstUser;
        program.rsrc1 = readRegister(queue.shader, rsrc2Register - 1);
        const auto resources = readRegister(queue.shader, rsrc2Register);
        const auto userCount = ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
        for (std::uint32_t i = 0; i < userCount; ++i) program.userData.push_back(readUserData(queue.shader, userBase + i));
        programs.push_back(std::move(program));
        return true;
    };
    if (state.stages.path == Graphics::ShaderPath::Geometry) {
        Native::RecordedProgram front;
        if (!Native::FindRecordedShader(programAddress(0xc8), front) || !add(0xc8, Stage::Mesh, Role::Main, 0x8b, 0x8c, 8)) {
            CountDrawFallback("unregistered geometry program");
            return false;
        }
        if (front.type == 4) {
            CountDrawFallback("legacy geometry back program");
            return false;
        }
        // Merged stage: eight hidden user words first (the merged user-data pointer, the mesh index
        // buffer V# at MeshIndexBufferUserWord).
        auto& main = programs.back();
        main.firstUserSgpr = 0;
        main.userData.insert(main.userData.begin(), 8, 0);
        const auto low = readRegister(queue.shader, 0x82);
        const auto high = readRegister(queue.shader, 0x83);
        main.userData[0] = low;
        main.userData[1] = high;
        if (state.stages.mesh) {
            const auto descriptor = Graphics::MeshIndexBufferDescriptor(draw);
            std::copy(descriptor.begin(), descriptor.end(), main.userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
        }
    } else if (!add(0xc8, Stage::Vertex, Role::Main, 0x8b, 0x8c, 8)) {
        CountDrawFallback("unregistered vertex program");
        return false;
    }
    if (!pixelless && !add(0x008, Stage::Fragment, Role::Fragment, 0x00b, 0x00c, 0)) {
        CountDrawFallback("unregistered pixel program");
        return false;
    }
    auto pixel = Graphics::DecodePixelStageInfo(queue.context, Graphics::ExportMappings(state), false);
    pixel.targetExportPacking = Graphics::ExportPackings(state);
    pixel.dualSourceBlend = state.dualSourceBlend;
    pixel.alphaToCoverage = state.alphaToCoverage;
    if (pixel.ancillary) pixel.rasterPrimitive = Graphics::RasterPrimitiveOf(state);
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    for (const auto& program : programs) linked.push_back({program.role, program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    phases.mark(1);
    const auto target = device->GraphicsTarget();
    const auto capacity = state.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : Graphics::PipelinePushSlotBytes;
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> results;
    std::vector<std::uint32_t> offsets;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> liveVertex;
    std::uint32_t pushOffset = 0;
    for (const auto& program : programs) {
        const bool fragment = program.stage == Stage::Fragment;
        const auto wave = fragment ? state.stages.fragmentWaveSize : state.stages.vertexWaveSize;
        // The vertex buffers of this draw (GT7's fetch tables, read live) and the same with only their
        // formats, which is what the translation depends on.
        std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex, vertexKey;
        try {
            if (!fragment) {
                vertex = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, nullptr, program.stage == Stage::Mesh, 8u - program.firstUserSgpr);
                vertexKey = vertex;
                for (std::uint32_t i = 0; i < vertexKey->resourcesNum; ++i) {
                    auto& fields = vertexKey->resources[i].fields;
                    fields[0] = 0;
                    fields[1] &= 0xffff0000u;
                    fields[2] = 0;
                }
                liveVertex = vertex;
            }
        } catch (const std::exception& error) {
            CountDrawFallback(std::string("vertex inputs: ") + error.what());
            return false;
        }
        pushOffset = Graphics::StagePushOffset(pushOffset, program.stage, target.fixedPushSlots);
        const auto room = (target.fixedPushSlots && fragment ? Graphics::PipelinePushSlotBytes : capacity) - pushOffset % Graphics::PipelinePushSlotBytes;
        std::vector<ShaderRecompiler::MemoryRegion> memory{{program.recorded.codeAddress, std::as_bytes(std::span(*program.recorded.code))}, {program.recorded.headerAddress, std::span(*program.recorded.header)}};
        const ShaderRecompiler::ShaderFloatMode floatMode{NativeFloatMode((program.rsrc1 >> 12u) & 0xffu), ((program.rsrc1 >> 21u) & 1u) != 0u, ((program.rsrc1 >> 23u) & 1u) != 0u, ((program.rsrc1 >> 26u) & 1u) != 0u};
        const ShaderRecompiler::RecompileRequest request{program.binary, {wave, program.firstUserSgpr, program.userData, std::nullopt, fragment ? std::optional(pixel) : std::nullopt, vertex, memory, floatMode}, target, {0, 0, pushOffset, room}, ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, state.stages.mesh, state.stages.tessellation, {}}};
        std::vector<std::uint64_t> key;
        {
            auto keyed = request;
            keyed.context.vertex = vertexKey;
            ShaderRecompiler::BuildPreparedShaderKey(keyed, key);
        }
        key.push_back(program.binary.codeAddress);
        key.push_back(pushOffset);
        auto& graphics = GraphicsWork();
        std::unique_lock lock(graphics.mutex);
        auto found = graphics.stages.find(key);
        if (found == graphics.stages.end()) {
            // A worker owns the linked-program copies. Preserve this draw while
            // it translates: a one-time producer must never be discarded.
            NativeGraphicsStage pending;
            pending.failure = "draw stage still translating";
            found = graphics.stages.emplace(key, std::move(pending)).first;
            lock.unlock();
            const auto index = static_cast<std::size_t>(&program - programs.data());
            std::thread([programs, index, wave, pixel, vertex, target, pushOffset, room, floatMode, mesh = state.stages.mesh, tessellation = state.stages.tessellation, key] {
                Native::BackgroundSlot slot;
                const auto& own = programs[index];
                const bool fragment = own.stage == Stage::Fragment;
                std::vector<ShaderRecompiler::LinkedProgram> linked;
                for (const auto& linkedProgram : programs) linked.push_back({linkedProgram.role, linkedProgram.binary, linkedProgram.userDataBase, linkedProgram.firstUserSgpr, linkedProgram.userData});
                std::vector<ShaderRecompiler::MemoryRegion> memory{{own.recorded.codeAddress, std::as_bytes(std::span(*own.recorded.code))}, {own.recorded.headerAddress, std::span(*own.recorded.header)}};
                const ShaderRecompiler::RecompileRequest request{own.binary, {wave, own.firstUserSgpr, own.userData, std::nullopt, fragment ? std::optional(pixel) : std::nullopt, vertex, memory, floatMode}, target, {0, 0, pushOffset, room}, ShaderRecompiler::GraphicsCompileContext{own.firstUserSgpr, linked, mesh, tessellation, {}}};
                NativeGraphicsStage made;
                made.program = own.recorded;
                CaptureNativeRequest(request, key);
                try {
                    made.result = std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
                } catch (const std::exception& error) {
                    std::string text = error.what();
                    std::replace(text.begin(), text.end(), '\n', '|');
                    made.failure = std::string(fragment ? "pixel" : "vertex") + " translation failed: " + text.substr(0, 700);
                }
                // Debug aid: APS5_NATIVE_SPV_DIR=<dir> saves each translated draw stage's SPIR-V.
                if (made.result != nullptr) {
                    static const char* directory = std::getenv("APS5_NATIVE_SPV_DIR");
                    if (directory != nullptr) {
                        char name[512];
                        std::snprintf(name, sizeof(name), "%s/stage%d_%llx.spv", directory, static_cast<int>(own.stage), static_cast<unsigned long long>(own.binary.codeAddress));
                        if (std::FILE* file = std::fopen(name, "wb")) {
                            std::fwrite(made.result->spirv.data(), 4, made.result->spirv.size(), file);
                            std::fclose(file);
                        }
                    }
                }
                auto& graphics = GraphicsWork();
                std::lock_guard done(graphics.mutex);
                graphics.stages[key] = std::move(made);
                graphics.ready.notify_all();
            }).detach();
            lock.lock();
            found = graphics.stages.find(key);
        }
        graphics.ready.wait(lock, [&] { return graphics.stages.at(key).failure != "draw stage still translating"; });
        found = graphics.stages.find(key);
        if (!found->second.failure.empty()) {
            const auto failure = found->second.failure;
            lock.unlock();
            CountDrawFallback(failure);
            return false;
        }
        results.push_back(found->second.result);
        offsets.push_back(pushOffset);
    }
    phases.mark(2);
    Native::NativeDevice::DrawWork work;
    work.state = &state;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& shader = *results[i];
        Native::NativeDevice::DrawStage stage;
        stage.shader = &shader;
        stage.stage = programs[i].stage;
        stage.pushOffset = offsets[i];
        stage.programAddress = programs[i].binary.codeAddress;
        stage.shaderData = ShaderDataFor(shader, offsets[i], programs[i].userData, programs[i].firstUserSgpr);
        // Fixed-function vertex fetch: the compiled attributes with this draw's buffers.
        if (liveVertex && !shader.vertexAttributes.empty()) {
            for (auto attribute : shader.vertexAttributes) {
                if (attribute.location < liveVertex->resourcesNum) {
                    attribute.resource = liveVertex->resources[attribute.location];
                    if (liveVertex->fetchEmbedded) attribute.fetchIndex = liveVertex->resourcesDst[attribute.location].fetchIndex;
                }
                stage.attributes.push_back(attribute);
            }
        }
        work.stages.push_back(std::move(stage));
    }
    if (!pixelless && results.back()->barycentricEmulation.active) {
        CountDrawFallback("pixel barycentrics need the generated geometry stage");
        return false;
    }
    std::shared_ptr<const ShaderRecompiler::RectListShaders> rectangle;
    if (state.rectList) {
        if (pixelless || work.stages.size() != 2 || work.stages.front().stage != Stage::Vertex) {
            CountDrawFallback("rect list without a plain vertex stage");
            return false;
        }
        auto& graphics = GraphicsWork();
        std::unique_lock lock(graphics.mutex);
        auto& slot = graphics.rectangles[{results.front().get(), results.back().get()}];
        if (slot == nullptr) {
            lock.unlock();
            std::shared_ptr<const ShaderRecompiler::RectListShaders> made;
            try {
                made = std::make_shared<const ShaderRecompiler::RectListShaders>(ShaderRecompiler::BuildRectListShaders(*results.front(), *results.back(), target));
            } catch (const std::exception& error) {
                lock.lock();
                CountDrawFallback(std::string("rect list stages: ") + error.what());
                return false;
            }
            lock.lock();
            slot = made;
        }
        rectangle = slot;
        Native::NativeDevice::DrawStage control, evaluation;
        control.shader = &rectangle->control;
        control.stage = Stage::TessellationControl;
        evaluation.shader = &rectangle->evaluation;
        evaluation.stage = Stage::TessellationEvaluation;
        work.stages.insert(work.stages.begin() + 1, {std::move(control), std::move(evaluation)});
    }
    {
        // Debug aid: APS5_NATIVE_DRAW_DUMP=<n> prints the first n draws' vertex data and vertex
        // stage user V#s' first words.
        static int left = [] { const char* text = std::getenv("APS5_NATIVE_DRAW_DUMP"); return text != nullptr ? std::atoi(text) : 0; }();
        if (left > 0) {
            --left;
            const auto floats = [](std::uint64_t address, int count) {
                std::string text;
                for (int i = 0; i < count; ++i) {
                    float value;
                    std::memcpy(&value, reinterpret_cast<const void*>(address + i * 4), 4);
                    char item[24];
                    std::snprintf(item, sizeof(item), " %g", value);
                    text += item;
                }
                return text;
            };
            for (const auto& stage : work.stages) {
                for (const auto& attribute : stage.attributes) {
                    const auto& f = attribute.resource.fields;
                    const auto address = f[0] | (static_cast<std::uint64_t>(f[1] & 0xffffu) << 32u);
                    if (address != 0) std::fprintf(stderr, "[native] dump attribute %u (%u comps, format word %08x, stride %u):%s\n", attribute.location, attribute.components, f[3], (f[1] >> 16u) & 0x3fffu, floats(address, 8).c_str());
                }
                if (stage.stage == Stage::Vertex && stage.shaderData.size() >= 4) {
                    const auto address = stage.shaderData[0] | (static_cast<std::uint64_t>(stage.shaderData[1] & 0xffffu) << 32u);
                    if (address != 0) std::fprintf(stderr, "[native] dump vertex constants 0x%llx:%s\n", static_cast<unsigned long long>(address), floats(address, 16).c_str());
                }
            }
        }
    }
    work.programAddresses = {programs.front().binary.codeAddress, programs.back().stage == Stage::Fragment ? programs.back().binary.codeAddress : 0u};
    work.indexed = draw.indexed;
    work.indexAddress = draw.indexAddress;
    work.indexSize = draw.indexSize;
    work.count = draw.indexCount;
    work.instances = draw.instanceCount;
    work.firstVertex = draw.firstVertex;
    work.firstInstance = draw.firstInstance;
    std::string failure;
    struct RecordMark {
        Phases& phases;
        ~RecordMark() { phases.mark(3); }
    } recordMark{phases};
    if (!device->Draw(work, failure)) {
        CountDrawFallback(failure);
        return false;
    }
    {
        auto& graphics = GraphicsWork();
        std::lock_guard lock(graphics.mutex);
        ++graphics.native;
    }
    ReportGraphics();
    return true;
}

// A DISPATCH_DIRECT as native compute. False (the dispatch skipped) when it cannot run natively yet.
bool DispatchNatively(QueueState& queue, std::span<const std::uint32_t> packet, bool indirect = false) {
    auto* device = Native::NativeDevice::Get();
    if (device == nullptr) {
        CountFallback("no native device yet");
        return false;
    }
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    Native::RecordedProgram program;
    if (!Native::FindRecordedShader(address, program) || program.type != 0) {
        CountFallback("unregistered compute program");
        return false;
    }
    // DCB indirect packets carry an offset from SET_BASE; ACB packets
    // carry address low/high. In both layouts the initiator is last.
    const auto initiator = packet.back();
    if (indirect && (initiator & 0x20u) != 0) {
        CountFallback("indirect dispatch of thread counts");
        return false;
    }
    const std::uint64_t indirectArguments = indirect ? Pm4::DispatchArgumentAddress(packet, queue) : 0u;
    const auto wave = (initiator & 0x8000u) != 0 ? 32u : 64u;
    const auto rsrc1 = readRegister(queue.shader, 0x212);
    const auto rsrc2 = readRegister(queue.shader, 0x213);
    std::array<std::uint32_t, 3> groups = indirect ? std::array<std::uint32_t, 3>{1u, 1u, 1u} : std::array<std::uint32_t, 3>{packet[1], packet[2], packet[3]};
    std::array<std::uint32_t, 3> partial{};
    const std::array<std::uint32_t, 3> numThreadRegisters{readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)};
    if (!indirect && (initiator & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto perGroup = std::max(numThreadRegisters[axis] & 0xffffu, 1u);
            if (threads[axis] % perGroup != 0) partial = threads;
            groups[axis] = (threads[axis] + perGroup - 1) / perGroup;
        }
    } else if (!indirect && (initiator & 0x2u) != 0) {
        partial = Pm4::PartialGroupThreads(groups, numThreadRegisters);
    }
    if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) return true;
    const bool partialGroups = partial != std::array<std::uint32_t, 3>{};
    std::uint64_t key = 0xcbf29ce484222325ull;
    for (const std::uint64_t value : {address, std::uint64_t{wave}, std::uint64_t{rsrc1}, std::uint64_t{rsrc2}, std::uint64_t{numThreadRegisters[0]}, std::uint64_t{numThreadRegisters[1]}, std::uint64_t{numThreadRegisters[2]}, std::uint64_t{partialGroups}}) {
        key ^= value;
        key *= 0x100000001b3ull;
    }
    const auto userCount = (rsrc2 >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData(userCount);
    for (std::uint32_t i = 0; i < userCount; ++i) userData[i] = readUserData(queue.shader, 0x240 + i);
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    {
        auto& compute = Compute();
        std::unique_lock lock(compute.mutex);
        auto found = compute.programs.find(key);
        if (found == compute.programs.end()) {
            // Translate off-thread, then resume this same dispatch. Skipping it
            // could lose persistent initialization data that later passes need.
            NativeProgram pending;
            pending.failure = "compute program still translating";
            found = compute.programs.emplace(key, std::move(pending)).first;
            lock.unlock();
            static std::atomic<int> traced{0};
            const bool trace = traced.fetch_add(1) < 8;
            if (trace) std::fprintf(stderr, "[native] translating compute program 0x%llx (wave%u)\n", static_cast<unsigned long long>(address), wave);
            std::optional<ShaderRecompiler::ShaderComputeStageInfo> info;
            std::string decodeFailure;
            try {
                info = Graphics::DecodeComputeStageInfo(queue.shader, std::span(*program.header));
                if (partialGroups) info->partialThreads = partial;
            } catch (const std::exception& error) {
                decodeFailure = std::string("translation failed: ") + error.what();
            }
            const ShaderRecompiler::ShaderFloatMode floatMode{NativeFloatMode((rsrc1 >> 12u) & 0xffu), ((rsrc1 >> 21u) & 1u) != 0u, ((rsrc1 >> 23u) & 1u) != 0u, ((rsrc1 >> 26u) & 1u) != 0u};
            const auto target = device->ComputeTarget(wave);
            std::thread([program, info, userData, floatMode, target, address, wave, key, trace, decodeFailure] {
                Native::BackgroundSlot slot;
                NativeProgram made;
                made.failure = decodeFailure;
                if (made.failure.empty()) {
                    try {
                        const auto codeOffset = static_cast<std::size_t>((address - program.codeAddress) / sizeof(std::uint32_t));
                        std::vector<ShaderRecompiler::MemoryRegion> memory{{program.codeAddress, std::as_bytes(std::span(*program.code))}, {program.headerAddress, std::span(*program.header)}};
                        const ShaderRecompiler::RecompileRequest request{
                            {ShaderRecompiler::ShaderStage::Compute, address, std::span(*program.code).subspan(codeOffset), program.headerAddress, std::span(*program.header)},
                            {wave, 0, userData, info, std::nullopt, std::nullopt, memory, floatMode},
                            target,
                            {0, 0, 0, 128}};
                        made.result = std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
                    } catch (const std::exception& error) {
                        std::string text = error.what();
                        if (const auto newline = text.find('\n'); newline != std::string::npos) text.resize(newline);
                        made.failure = "translation failed: " + text.substr(0, 1500);
                    }
                }
                if (trace) std::fprintf(stderr, "[native] translated 0x%llx: %s\n", static_cast<unsigned long long>(address), made.failure.empty() ? "ok" : made.failure.c_str());
                auto& compute = Compute();
                std::lock_guard done(compute.mutex);
                compute.programs[key] = std::move(made);
                compute.ready.notify_all();
            }).detach();
            lock.lock();
            found = compute.programs.find(key);
        }
        compute.ready.wait(lock, [&] { return compute.programs.at(key).failure != "compute program still translating"; });
        found = compute.programs.find(key);
        if (!found->second.failure.empty()) {
            const auto failure = found->second.failure;
            lock.unlock();
            CountFallback(failure);
            return false;
        }
        result = found->second.result;
    }
    const auto& shader = *result;
    {
        // Debug aid: APS5_NATIVE_WATCH=<program address> prints that program's user data and the
        // first words its first user pointer refers to, for its first dispatches.
        static const std::uint64_t watched = [] { const char* text = std::getenv("APS5_NATIVE_WATCH"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        static int shown = 0;
        if (watched == address && shown < 6 && userData.size() >= 2) {
            ++shown;
            const auto pointer = userData[0] | (static_cast<std::uint64_t>(userData[1]) << 32u);
            std::string words;
            for (std::uint32_t i = 0; i < 48 && pointer != 0 && device->DeviceAddress(pointer, 192) != 0; ++i) {
                char item[16];
                std::snprintf(item, sizeof(item), " %08x", reinterpret_cast<const std::uint32_t*>(pointer)[i]);
                words += item;
            }
            std::string user;
            for (const auto word : userData) {
                char item[16];
                std::snprintf(item, sizeof(item), " %x", word);
                user += item;
            }
            std::fprintf(stderr, "[native] watch user data:%s\n", user.c_str());
            std::fprintf(stderr, "[native] watch 0x%llx groups %ux%ux%u%s user %zu words, table 0x%llx:%s\n", static_cast<unsigned long long>(address), groups[0], groups[1], groups[2], indirect ? " (indirect)" : "", userData.size(), static_cast<unsigned long long>(pointer), words.c_str());
        }
    }
    auto shaderData = ShaderDataFor(shader, 0, userData, 0);
    if (shader.dispatchThreadLimitDword != ~0u && shader.dispatchThreadLimitDword + 3u <= shaderData.size()) std::copy(partial.begin(), partial.end(), shaderData.begin() + shader.dispatchThreadLimitDword);
    std::string failure;
    if (!device->Dispatch({&shader, key, groups, shaderData, address, indirectArguments}, failure)) {
        CountFallback(failure);
        return false;
    }
    {
        auto& compute = Compute();
        std::lock_guard lock(compute.mutex);
        ++compute.native;
    }
    ReportCompute();
    return true;
}

// What each queue's native executor is doing (APS5_NATIVE_STATS: printed every 5 s by a watchdog).
struct QueueActivity {
    std::atomic<std::uint64_t> serial{0};
    std::atomic<std::uint32_t> cursor{0};
    std::atomic<std::uint32_t> opcode{0};
    std::atomic<const char*> phase{"idle"};
};

std::mutex& ActivityMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::uint32_t, QueueActivity>& Activity() {
    static std::map<std::uint32_t, QueueActivity> activity;
    return activity;
}

QueueActivity& ActivityOf(std::uint32_t queue) {
    std::lock_guard lock(ActivityMutex());
    static const bool watchdog = [] {
        if (std::getenv("APS5_NATIVE_STATS") == nullptr) return false;
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                std::lock_guard lock(ActivityMutex());
                std::string line = "[native] queues:";
                for (auto& [queue, activity] : Activity()) {
                    char text[160];
                    std::snprintf(text, sizeof(text), " 0x%x=%s(submission %llu dword %u opcode 0x%x)", queue, activity.phase.load(), static_cast<unsigned long long>(activity.serial.load()), activity.cursor.load(), activity.opcode.load());
                    line += text;
                }
                std::fprintf(stderr, "%s\n", line.c_str());
            }
        }).detach();
        return true;
    }();
    static_cast<void>(watchdog);
    return Activity()[queue];
}

// A line every 5 s while the native executor runs: what the submissions contained.
struct ExecutorCounts {
    std::atomic<std::uint64_t> submissions{0}, packets{0}, dispatches{0}, draws{0}, flips{0}, waits{0};
    std::atomic<std::int64_t> reportedAt{0};
};

ExecutorCounts& Counts() {
    static ExecutorCounts counts;
    return counts;
}

void ReportExecutor() {
    auto& counts = Counts();
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counts.reportedAt.load();
    if (now - last < 5 || !counts.reportedAt.compare_exchange_strong(last, now)) return;
    std::fprintf(stderr, "[native] executor: %llu submissions, %llu packets, %llu dispatches, %llu draws, %llu flips, %llu waits\n", static_cast<unsigned long long>(counts.submissions.load()), static_cast<unsigned long long>(counts.packets.load()), static_cast<unsigned long long>(counts.dispatches.load()), static_cast<unsigned long long>(counts.draws.load()), static_cast<unsigned long long>(counts.flips.load()), static_cast<unsigned long long>(counts.waits.load()));
}

// Whether a packet's effect on the CPU side needs the native GPU work recorded before it done.
// End-of-pipe label writes deferred to the completion of the open native batch (RELEASE_MEM with
// immediate data): a wait on one of them later in the same batch is satisfied by the batch's
// in-order execution, so it neither ends the batch nor blocks.
struct PendingRelease {
    std::uint64_t address;
    std::uint64_t value;
    std::uint64_t serial;
};

struct PendingReleases {
    std::mutex mutex;
    std::vector<PendingRelease> entries;
    std::uint64_t serial = 0;
};

PendingReleases& Releases() {
    static PendingReleases releases;
    return releases;
}

bool SatisfiedByPendingRelease(std::span<const std::uint32_t> packet) {
    const auto awaited = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
    auto& releases = Releases();
    std::lock_guard lock(releases.mutex);
    for (auto it = releases.entries.rbegin(); it != releases.entries.rend(); ++it) {
        if (it->address == awaited) return Pm4::WaitComparesValue(packet, it->value);
    }
    return false;
}

// The packets whose effect another agent observes (the CPU, another queue) or that read memory the
// native batch may still be writing: the batch completes before them. Everything else (draws,
// dispatches, state, cache control) stays in the batch, which orders its own work with barriers.
bool NeedsNativeResults(std::uint32_t opcode) {
    switch (opcode) {
        case 0x20:   // SET_PREDICATION
        case 0x22:   // COND_EXEC
        case 0x24:   // DRAW_INDIRECT
        case 0x25:   // DRAW_INDEX_INDIRECT
        case 0x2C:   // DRAW_INDIRECT_MULTI
        case 0x37:   // WRITE_DATA
        case 0x38:   // DRAW_INDEX_INDIRECT_MULTI
        case 0x39:   // MEM_SEMAPHORE
        case 0x3C:   // WAIT_REG_MEM
        case 0x40:   // COPY_DATA
        case 0x41:   // CP_DMA
        case 0x45:   // COND_WRITE
        case 0x47:   // EVENT_WRITE_EOP
        case 0x48:   // EVENT_WRITE_EOS
        case 0x50:   // DMA_DATA
        case 0x8E:   // GET_LOD_STATS
        case 0x93:   // WAIT_REG_MEM_64
            return true;
        default:
            return false;
    }
}

void FlushNative() {
    static std::atomic<int> traced{0};
    if (auto* device = Native::NativeDevice::Get(); device != nullptr && device->Pending()) {
        const bool trace = traced.fetch_add(1) < 4;
        if (trace) std::fprintf(stderr, "[native] flushing native work\n");
        device->Flush();
        if (trace) std::fprintf(stderr, "[native] native work done\n");
    }
}

void ReportUnexecuted(std::uint32_t header, const char* reason) {
    static std::mutex mutex;
    static std::set<std::uint32_t> reported;
    const auto opcode = (header >> 8u) & 0xffu;
    std::lock_guard lock(mutex);
    if (reported.insert(opcode).second) std::fprintf(stderr, "[native] packet %s not executed: %s\n", Pm4::Name(header).c_str(), reason);
}

}

void Driver::executeNative(const Submission& submission) {
    struct FlipHold {
        Driver& driver;
        bool held;
        void Release() {
            if (!held) return;
            held = false;
            driver.releaseFlipHold();
        }
        ~FlipHold() { Release(); }
    } flipHold{*this, submission.holdsFlip};
    if (submission.suspend) return;

    Native::CountSubmission(submission.queue);
    const auto serial = ++Counts().submissions;
    auto& activity = ActivityOf(submission.queue);
    activity.serial = submission.serial;
    activity.phase = "executing";
    struct Idle {
        QueueActivity& activity;
        ~Idle() { activity.phase = "idle"; }
    } idle{activity};
    ReportExecutor();
    // The first submissions' packet mix (APS5_NATIVE_STATS): what the game asked of the GPU.
    static const bool stats = std::getenv("APS5_NATIVE_STATS") != nullptr;
    static const auto logFrom = std::getenv("APS5_NATIVE_LOG_FROM") != nullptr ? std::strtoull(std::getenv("APS5_NATIVE_LOG_FROM"), nullptr, 10) : ~0ull;
    const bool submissionLog = stats && (serial <= 12 || submission.serial >= logFrom);
    std::map<std::uint32_t, std::uint32_t> opcodes;
    struct LogAtEnd {
        bool active;
        const Submission& submission;
        std::map<std::uint32_t, std::uint32_t>& opcodes;
        ~LogAtEnd() {
            if (!active) return;
            std::string line;
            for (const auto& [opcode, count] : opcodes) line += " " + Pm4::Name((3u << 30u) | (opcode << 8u)) + "x" + std::to_string(count);
            std::fprintf(stderr, "[native] submission %llu queue 0x%x (%zu words, %zu flips):%s\n", static_cast<unsigned long long>(submission.serial), submission.queue, submission.commands.size(), submission.flips.size(), line.c_str());
        }
    } logAtEnd{submissionLog, submission, opcodes};
    QueueState* state = nullptr;
    {
        std::lock_guard lock(mutex);
        if (submission.queue == 0 && resetGraphics) {
            queues.erase(0);
            resetGraphics = false;
        }
        state = &queues[submission.queue];
    }
    auto& queue = *state;

    for (std::size_t cursor = 0; cursor < submission.commands.size();) {
        CheckFailure();
        const auto header = submission.commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(submission.commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        auto nextCursor = cursor + count;

        ++Counts().packets;
        activity.cursor = static_cast<std::uint32_t>(cursor);
        activity.opcode = opcode;
        activity.phase = "packet";
        if (submissionLog) opcodes[opcode]++;
        const bool waitByOrder = (opcode == 0x3c || opcode == 0x93) && !Pm4::WaitSatisfied(packet) && SatisfiedByPendingRelease(packet);
        if (NeedsNativeResults(opcode) && !waitByOrder) {
            activity.phase = "flushing native work";
            {
                // APS5_NATIVE_STATS: which packets cut the native batch.
                static std::map<std::uint32_t, std::uint64_t> cuts;
                static std::mutex cutsMutex;
                static auto reported = std::chrono::steady_clock::now();
                if (auto* device = Native::NativeDevice::Get(); device != nullptr && device->Pending()) {
                    std::lock_guard lock(cutsMutex);
                    ++cuts[opcode];
                    if (std::chrono::steady_clock::now() - reported > std::chrono::seconds(5)) {
                        reported = std::chrono::steady_clock::now();
                        std::string line;
                        for (const auto& [cut, count] : cuts) line += " " + Pm4::Name((3u << 30u) | (cut << 8u)) + "x" + std::to_string(count);
                        std::fprintf(stderr, "[native] batch cuts in 5 s:%s\n", line.c_str());
                        cuts.clear();
                    }
                }
            }
            FlushNative();
            activity.phase = "packet";
        }
        if (Pm4::Predicated(header) && queue.predication.operation != 0 && !Pm4::PredicationPasses(queue)) {
            cursor = opcode == 0x3f ? submission.conditionalEnds.at(cursor) : nextCursor;
            continue;
        }

        if (header == RenderingWaitPacketHeader) {
            if (submissionLog) std::fprintf(stderr, "[native] submission %llu: waiting until safe for rendering\n", static_cast<unsigned long long>(submission.serial));
            submission.renderingWaits.at(cursor)->Wait();
            if (submissionLog) std::fprintf(stderr, "[native] submission %llu: safe for rendering\n", static_cast<unsigned long long>(submission.serial));
        } else if (header == FlipPacketHeader && (++Counts().flips, true)) {
            // The frame's native work completes before the presenter reads the display buffer.
            activity.phase = "flushing native work";
            FlushNative();
            activity.phase = "packet";
            ++flipsCounted;
            auto frame = std::make_shared<FrameTiming>(++frameSerial);
            const auto now = FrameTiming::Clock::now();
            frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, now, now);
            flipHold.Release();
            submission.flips.at(cursor)->GpuReady(frame);
        } else if (opcode == 0x3f) {
            // Indirect buffer bodies are inlined into the submission when it is copied.
        } else if (Pm4::DrawOpcode(opcode) && (++Counts().draws, activity.phase = "native draw", true)) {
            if (!DrawNatively(queue, packet)) Native::CountSkippedDraw();
        } else if (opcode == 0x15 && (++Counts().dispatches, activity.phase = "native dispatch", true)) {
            if (!DispatchNatively(queue, packet)) Native::CountSkippedDispatch();
        } else if (opcode == 0x16 && (++Counts().dispatches, activity.phase = "native dispatch", true)) {
            if (!DispatchNatively(queue, packet, true)) Native::CountSkippedDispatch();
        } else if (DispatchOpcode(opcode)) {
            CountFallback("indirect dispatch");
            Native::CountSkippedDispatch();
        } else if ((opcode == 0x3c || opcode == 0x93) && (++Counts().waits, true)) {
            if (!waitByOrder && !Pm4::WaitSatisfied(packet)) {
                // Blocked: the scheduler then lets later-submitted queues run (the value usually
                // comes from another queue's work, as GT7's async compute signals the graphics queue).
                const auto awaited = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
                activity.phase = "waiting on memory";
                noteWaitBlocked(submission.queue, awaited, true);
                struct Unblock {
                    Driver& driver;
                    std::uint32_t queue;
                    std::uint64_t awaited;
                    ~Unblock() { driver.noteWaitBlocked(queue, awaited, false); }
                } unblock{*this, submission.queue, awaited};
                WaitWithReport(submission, cursor, packet);
            }
        } else if (opcode == 0x22) {
            if (Pm4::ReadCondition(packet) == 0) nextCursor = submission.conditionalEnds.at(cursor);
        } else if (!IgnoredOpcode(opcode)) {
            if (opcode == 0x49) {
                // End of pipe: the label write and interrupt happen once the native work recorded
                // so far has completed, without ending the batch.
                std::vector<std::uint32_t> copy(packet.begin(), packet.end());
                auto* device = Native::NativeDevice::Get();
                const auto queueId = submission.queue;
                // A release with immediate data satisfies waits on it within the batch.
                const auto dataSelect = copy[2] >> 29u;
                const auto destination = copy[3] | (static_cast<std::uint64_t>(copy[4]) << 32u);
                std::uint64_t serial = 0;
                if (device != nullptr && device->Pending() && (dataSelect == 1 || dataSelect == 2) && destination != 0) {
                    auto& releases = Releases();
                    std::lock_guard lock(releases.mutex);
                    serial = ++releases.serial;
                    const auto value = dataSelect == 1 ? static_cast<std::uint64_t>(copy[5]) : (copy[5] | (static_cast<std::uint64_t>(copy[6]) << 32u));
                    releases.entries.push_back({destination, value, serial});
                }
                auto release = [copy = std::move(copy), &queue, queueId, header, serial] {
                    if (serial != 0) {
                        auto& releases = Releases();
                        std::lock_guard lock(releases.mutex);
                        std::erase_if(releases.entries, [&](const PendingRelease& entry) { return entry.serial == serial; });
                    }
                    try {
                        Pm4::Execute(copy, queue);
                    } catch (const std::runtime_error& error) {
                        ReportUnexecuted(header, error.what());
                    }
                    const auto interruptSelect = (copy[2] >> 24u) & 7u;
                    if (interruptSelect != 0 && interruptSelect != 3) AgcDriverDeliverEopInterrupt(queueId);
                };
                if (device != nullptr) device->AfterBatch(std::move(release));
                else release();
            } else {
                try {
                    if (opcode == 0x50 && Native::NativeDevice::Get() != nullptr && Native::NativeDevice::Get()->DmaGds(packet)) {}
                    else if (Pm4::IndirectRegisterOpcode(opcode)) Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(cursor), queue);
                    else Pm4::Execute(packet, queue);
                } catch (const std::runtime_error& error) {
                    ReportUnexecuted(header, error.what());
                }
            }
        }
        cursor = nextCursor;
    }
    FlushNative();
    // A rewindable command buffer: the game writes the rest later and marks it valid; it runs
    // as the continuation of this submission.
    if (submission.rewindTail != nullptr) executeRewindTail(submission);
}

}
