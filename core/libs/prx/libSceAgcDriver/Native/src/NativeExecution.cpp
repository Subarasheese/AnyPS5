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
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Native/include/FrontEnd.hpp"
#include "prx/libSceAgcDriver/Native/include/CompileStatus.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeImages.hpp"
#include "ShaderCacheDirectory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "ShaderDiskCache.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
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
bool SatisfiedByPendingRelease(std::span<const std::uint32_t> packet);

// Per GT7 queue, the native batch its latest GPU work was recorded into: an end-of-pipe release on
// that queue completes with that batch, as on a GPU whose queues run independently.
std::atomic<std::uint64_t>& QueueWorkSerial(std::uint32_t queue) {
    static std::array<std::atomic<std::uint64_t>, 256> serials{};
    return serials[queue & 0xffu];
}

void NoteQueueWork(std::uint32_t queue) {
    if (auto* device = Native::NativeDevice::Get(); device != nullptr) QueueWorkSerial(queue).store(device->RecordingSerial(), std::memory_order_release);
}

void WaitWithReport(const Submission& submission, std::size_t at, std::span<const std::uint32_t> packet) {
    const auto start = std::chrono::steady_clock::now();
    bool reported = false;
    while (!Pm4::WaitSatisfied(packet) && !SatisfiedByPendingRelease(packet)) {
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
    Native::RecordedProgram program;
    std::optional<ShaderRecompiler::RecompileRequest> classic;
    std::shared_ptr<const ShaderRecompiler::SourceHandle> descriptors;
    std::shared_ptr<const ShaderRecompiler::IrResourcePlan> flattened;
};

struct VerifiedSpans {
    std::uint64_t generation = 0;
    std::array<std::pair<std::uint64_t, std::uint64_t>, 16> spans{};
    std::size_t next = 0;
};

thread_local VerifiedSpans verifiedSpans;

bool Readable(std::uint64_t address, std::uint64_t bytes) {
    if (address == 0 || bytes > ~address) return false;
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    auto& verified = verifiedSpans;
    if (verified.generation != generation) verified = {generation};
    for (const auto& [begin, end] : verified.spans) {
        if (address >= begin && address + bytes <= end) return true;
    }
    const auto begin = address & ~0xffffull, end = (address + bytes + 0xffffull) & ~0xffffull;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), static_cast<std::size_t>(end - begin))) return GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes));
    verified.spans[verified.next++ % verified.spans.size()] = {begin, end};
    return true;
}

bool ReadGuestWord(void*, std::uint64_t address, std::uint32_t* value) {
    Native::NotePoolUse("walk", address);
    if (!Readable(address, 4)) return false;
    std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(*value));
    return true;
}

bool ReadGuestWords(void*, std::uint64_t address, std::uint32_t* values, std::uint64_t count) {
    Native::NotePoolUse("walk", address);
    if (!Readable(address, count * 4u)) return false;
    std::memcpy(values, reinterpret_cast<const void*>(address), static_cast<std::size_t>(count * 4u));
    return true;
}

bool GuestRangeAccessible(void*, std::uint64_t address, std::uint64_t bytes) {
    return Readable(address, bytes);
}

bool NoPendingWrite(void*, std::uint64_t, std::uint64_t) {
    return false;
}

// The plan whose SRT reads a translation folded into its flattened table (DescriptorRole::FlattenedSrt),
// walked per dispatch or draw at the call site; null when the translation has no such table.
std::shared_ptr<const ShaderRecompiler::IrResourcePlan> FlattenedPlan(const ShaderRecompiler::RecompileRequest& request, const ShaderRecompiler::RecompileResult& result) {
    const bool flattened = std::any_of(result.bindings.begin(), result.bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; });
    return flattened || !result.nativeImageSources.empty() || !result.nativeSamplerSources.empty() ? ShaderRecompiler::GetResourcePlan(request) : nullptr;
}

// The call site's walk of a translation's plan: its flattened SRT, and the T#s (8 words each) then
// S#s (4 words each) of its image and sampler slots in `slotWords`.
std::vector<std::uint32_t> FlattenedSrt(const ShaderRecompiler::IrResourcePlan* plan, const ShaderRecompiler::RecompileResult& result, std::span<const std::uint32_t> userData, std::uint64_t codeAddress, std::vector<std::uint32_t>& slotWords) {
    std::vector<std::uint32_t> flat;
    slotWords.clear();
    if (plan == nullptr) return flat;
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = userData;
    runtime.shaderBase = codeAddress;
    runtime.readMemory = &ReadGuestWord;
    runtime.readSpecializationMemory = &ReadGuestWord;
    runtime.accessible = &GuestRangeAccessible;
    runtime.pendingWrite = &NoPendingWrite;
    runtime.readWords = &ReadGuestWords;
    std::vector<std::uint32_t> sources(result.nativeImageSources);
    sources.insert(sources.end(), result.nativeSamplerSources.begin(), result.nativeSamplerSources.end());
    try {
        constexpr ShaderRecompiler::SrtWalker walker;
        std::vector<ShaderRecompiler::DescriptorValue> descriptors;
        std::vector<std::uint8_t> active;
        // Images read from a descriptor table at a run-time key (TableColumn: GT7's MakeLightProbeTex
        // takes its store targets from a buffer row picked by the workgroup) have no single T# at the
        // call site: the shader reads the row's T# and finds it in the image heap. Each row's T# is
        // registered here so that lookup succeeds; the table's buffer is evaluated with the slots.
        std::vector<std::pair<std::size_t, const ShaderRecompiler::TableColumn*>> tables;
        for (std::size_t index = 0; index < result.nativeImageSources.size(); ++index) {
            const auto source = result.nativeImageSources[index];
            if (source >= plan->descriptorSources.size() || !plan->descriptorSources[source].tableColumn.has_value()) continue;
            const auto& column = *plan->descriptorSources[source].tableColumn;
            if (column.sampler || column.address || column.indirect || column.dwordCount != 8u || column.stride == 0u) continue;
            tables.emplace_back(sources.size(), &column);
            sources.push_back(column.heapSource);
        }
        walker.EvaluateRuntimeSources(*plan, sources, runtime, descriptors, flat, {}, active);
        slotWords.reserve(result.nativeImageSources.size() * 8 + result.nativeSamplerSources.size() * 4);
        const auto slots = std::min(descriptors.size(), result.nativeImageSources.size() + result.nativeSamplerSources.size());
        for (std::size_t index = 0; index < slots; ++index) {
            const auto width = index < result.nativeImageSources.size() ? 8u : 4u;
            slotWords.insert(slotWords.end(), descriptors[index].dwords.begin(), descriptors[index].dwords.begin() + width);
        }
        for (const auto& [heap, column] : tables) {
            if (heap >= descriptors.size()) continue;
            const auto& buffer = descriptors[heap].dwords;
            const auto base = buffer[0] | (static_cast<std::uint64_t>(buffer[1] & 0xffffu) << 32u);
            const auto stride = (buffer[1] >> 16u) & 0x3fffu;
            const std::uint64_t size = stride == 0u ? buffer[2] : static_cast<std::uint64_t>(stride) * buffer[2];
            const std::uint64_t keys = column->maxKey == 0xffffffffu ? 4096u : std::min<std::uint64_t>(4096u, column->maxKey / column->stride + 1u);
            for (std::uint64_t key = 0; key < keys; ++key) {
                const auto position = static_cast<std::uint64_t>(column->addend) + key * column->stride + column->offset;
                if (position + 32u > size || !GuestRangeAccessible(nullptr, base + position, 32u)) break;
                std::array<std::uint32_t, 8> words{};
                std::memcpy(words.data(), reinterpret_cast<const void*>(base + position), sizeof(words));
                if ((words[0] | (words[1] & 0xffu)) != 0u) Native::NoteSampledDescriptor(std::span<const std::uint32_t, 8>(words));
            }
        }
    } catch (const std::exception& error) {
        Native::ReportGraphicsIssue("descriptor words not walked at the call site", error.what());
        flat.clear();
        slotWords.clear();
    }
    return flat;
}

// APS5_NATIVE_SLOT_SRT: translations take their descriptors from call-site slots, which also give
// the draw's textures (no CPU capture of the classic translation per draw).
bool SlotDescriptors() {
    static const bool enabled = std::getenv("APS5_NATIVE_SLOT_SRT") != nullptr;
    return enabled;
}

ShaderRecompiler::RecompileRequest ClassicRequest(ShaderRecompiler::RecompileRequest request) {
    request.target.gpuResolvedResources = false;
    return request;
}

std::shared_ptr<const ShaderRecompiler::SourceHandle> ResolveDescriptors(const ShaderRecompiler::RecompileRequest& classic) {
    try {
        return ShaderRecompiler::ResolveSource(classic);
    } catch (const std::exception& error) {
        Native::ReportGraphicsIssue("texture descriptors cannot be resolved ahead of use", error.what());
        return nullptr;
    }
}

std::vector<std::uint64_t> AddressList(const char* name) {
    std::vector<std::uint64_t> addresses;
    if (const char* text = std::getenv(name)) {
        while (*text) {
            char* end = nullptr;
            const auto address = std::strtoull(text, &end, 0);
            if (end == text) break;
            addresses.push_back(address);
            text = *end == ',' ? end + 1 : end;
        }
    }
    return addresses;
}

bool GuestBytesNonzero(std::uint64_t address, std::uint64_t bytes) {
    std::array<std::byte, 4096> chunk{};
    const auto step = std::max<std::uint64_t>(bytes / 64u, chunk.size());
    for (std::uint64_t at = 0; at < bytes; at += step) {
        const auto take = static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), bytes - at));
        if (GuestMemory::CopyMapped(address + at, std::span(chunk.data(), take)) != GuestMemory::Compare::Equal) return false;
        if (std::any_of(chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(take), [](std::byte b) { return b != std::byte{0}; })) return true;
    }
    return false;
}

void InspectCapture(const ShaderRecompiler::ResourceCapture& capture, std::uint64_t program, std::uint64_t target, bool stores) {
    auto* device = Native::NativeDevice::Get();
    if (device == nullptr) return;
    for (const auto& buffer : capture.snapshot.buffers) {
        if (buffer.dwordCount < 4u) continue;
        const auto base = static_cast<std::uint64_t>(buffer.dwords[0]) | (static_cast<std::uint64_t>(buffer.dwords[1] & 0xffffu) << 32u);
        const auto stride = (buffer.dwords[1] >> 16u) & 0x3fffu;
        const auto bytes = std::max<std::uint64_t>(stride != 0 ? static_cast<std::uint64_t>(stride) * buffer.dwords[2] : buffer.dwords[2], 1u);
        if (base == 0 || buffer.dwords[2] == 0) continue;
        char detail[160];
        if (device->DeviceAddress(base, 1) == 0) {
            std::snprintf(detail, sizeof(detail), "program 0x%llx buffer 0x%llx (%llu bytes)", static_cast<unsigned long long>(program), static_cast<unsigned long long>(base), static_cast<unsigned long long>(bytes));
            Native::ReportGraphicsIssue("buffer not visible to the GPU (reads return 0)", detail);
        }
        if (stores) device->MarkTextureMemoryWritten(base, base + bytes);
    }
    static const auto targets = AddressList("APS5_NATIVE_TRACE_TARGETS");
    static const auto programs = AddressList("APS5_NATIVE_TRACE_PS");
    const bool traced = (target != 0 && std::find(targets.begin(), targets.end(), target) != targets.end()) || std::find(programs.begin(), programs.end(), program) != programs.end();
    if (!traced) return;
    static std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
    for (const auto& image : capture.snapshot.images) {
        if (image.dwordCount != 8u || seen.size() >= 2000) continue;
        const auto base = (static_cast<std::uint64_t>(image.dwords[0]) | (static_cast<std::uint64_t>(image.dwords[1] & 0xffu) << 32u)) << 8u;
        if (!seen.insert({program ^ (target << 1u), base}).second) continue;
        try {
            const auto resource = Graphics::DecodeTextureResource(image.dwords);
            const auto bytes = Graphics::DescribeSurface(resource).guestBytes;
            if (const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR"); directory != nullptr && GuestMemory::Accessible(reinterpret_cast<const void*>(resource.baseAddress), bytes)) {
                std::filesystem::create_directories(directory);
                char name[160];
                std::snprintf(name, sizeof(name), "trace_guest_%012llx_%ux%u_fmt%u_tile%u.bin", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned>(resource.tileMode));
                std::ofstream(std::filesystem::path(directory) / name, std::ios::binary).write(reinterpret_cast<const char*>(resource.baseAddress), static_cast<std::streamsize>(bytes));
                device->RequestSurfaceDump(resource.baseAddress);
            }
            std::fprintf(stderr, "[trace] program 0x%llx target 0x%llx samples 0x%llx %ux%u format %u tile %u bytes %llu registered %d guest-nonzero %d words %08x %08x %08x %08x %08x %08x %08x %08x\n",
                static_cast<unsigned long long>(program), static_cast<unsigned long long>(target), static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned>(resource.tileMode), static_cast<unsigned long long>(bytes),
                device->TextureRegistered(std::span<const std::uint32_t, 8>(image.dwords)) ? 1 : 0, GuestBytesNonzero(resource.baseAddress, bytes) ? 1 : 0,
                image.dwords[0], image.dwords[1], image.dwords[2], image.dwords[3], image.dwords[4], image.dwords[5], image.dwords[6], image.dwords[7]);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[trace] program 0x%llx target 0x%llx samples an undecodable T#: %s\n", static_cast<unsigned long long>(program), static_cast<unsigned long long>(target), error.what());
        }
    }
}

using KeyDigest = std::array<std::uint64_t, 2>;

KeyDigest Digest(std::span<const std::uint64_t> words) {
    std::uint64_t a = 0x9e3779b97f4a7c15ull ^ words.size(), b = 0xc2b2ae3d27d4eb4full + words.size();
    for (const auto word : words) {
        a = (a ^ word) * 0xff51afd7ed558ccdull;
        a ^= a >> 32u;
        b = (b + word) * 0xc4ceb9fe1a85ec53ull;
        b ^= b >> 29u;
    }
    return {a ^ (b >> 17u), b ^ (a << 13u)};
}

struct KeyDigestHash {
    std::size_t operator()(const KeyDigest& digest) const noexcept { return static_cast<std::size_t>(digest[0] ^ (digest[1] * 0x9e3779b97f4a7c15ull)); }
};

std::shared_ptr<const ShaderRecompiler::ResourceCapture> CaptureMemoized(const ShaderRecompiler::RecompileRequest& request, bool toClassic, const ShaderRecompiler::SourceHandle& handle) {
    const auto classic = toClassic ? ClassicRequest(request) : request;
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = classic.context.userData;
    runtime.shaderBase = classic.shader.codeAddress;
    runtime.readMemory = &ReadGuestWord;
    runtime.readSpecializationMemory = &ReadGuestWord;
    runtime.accessible = &GuestRangeAccessible;
    runtime.pendingWrite = &NoPendingWrite;
    runtime.readWords = &ReadGuestWords;
    return ShaderRecompiler::CaptureResources(classic, runtime, handle);
}

void RegisterSampledDescriptors(const ShaderRecompiler::RecompileRequest& request, bool toClassic, const ShaderRecompiler::SourceHandle& handle, std::uint64_t target = 0, bool stores = false, std::vector<std::uint64_t>* sampled = nullptr) {
    try {
        const auto capture = CaptureMemoized(request, toClassic, handle);
        for (const auto& image : capture->snapshot.images) {
            if (image.dwordCount != 8u) continue;
            Native::NoteSampledDescriptor(std::span<const std::uint32_t, 8>(image.dwords));
            if (sampled != nullptr) sampled->push_back((static_cast<std::uint64_t>(image.dwords[0]) | (static_cast<std::uint64_t>(image.dwords[1] & 0xffu) << 32u)) << 8u);
        }
        InspectCapture(*capture, request.shader.codeAddress, target, stores);
    } catch (const std::exception& error) {
        Native::ReportGraphicsIssue("texture descriptors not resolved before a draw", error.what());
    }
}

struct NativeCompute {
    std::mutex mutex;
    std::condition_variable ready;
    std::unordered_map<std::uint64_t, NativeProgram> programs;
    std::map<std::string, std::uint64_t> fallbacks;
    std::uint64_t native = 0;
    std::uint64_t frontEnd = 0;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

NativeCompute& Compute() {
    static NativeCompute compute;
    return compute;
}

void CountFallback(const std::string& reason) {
    auto& compute = Compute();
    std::lock_guard lock(compute.mutex);
    ++compute.fallbacks[reason];
    Native::ReportGraphicsIssue("compute dispatch skipped", reason);
}

void ReportCompute() {
    auto& compute = Compute();
    std::lock_guard lock(compute.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - compute.reported < std::chrono::seconds(5)) return;
    compute.reported = now;
    std::uint64_t skipped = 0;
    for (const auto& [reason, count] : compute.fallbacks) skipped += count;
    if (Native::Logs()) std::fprintf(stderr, "[native] compute: %llu native dispatches (%llu prepared at the game's call site), %llu skipped, %zu programs\n", static_cast<unsigned long long>(compute.native), static_cast<unsigned long long>(compute.frontEnd), static_cast<unsigned long long>(skipped), compute.programs.size());
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
std::vector<std::uint32_t> ShaderDataFor(const ShaderRecompiler::RecompileResult& shader, std::uint32_t pushOffset, std::span<const std::uint32_t> userData, std::uint32_t firstUserSgpr, std::span<const std::uint32_t> flattenedSrt = {}) {
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
    // User-data pointers translated here, once per invocation, instead of by every read.
    const auto bases = shader.nativeBaseKeys.size() * 4u;
    if (bases != 0 && data.size() >= bases + ShaderRecompiler::RuntimeAbi::NativeHeap::InvocationAddressDwords) {
        auto* device = Native::NativeDevice::Get();
        const auto word = [&](std::uint32_t key, std::uint32_t index) -> std::uint32_t {
            if ((key & ShaderRecompiler::NativeBaseFlat) != 0) return index < flattenedSrt.size() ? flattenedSrt[index] : 0u;
            return index >= firstUserSgpr && index - firstUserSgpr < userData.size() ? userData[index - firstUserSgpr] : 0u;
        };
        auto* slot = data.data() + data.size() - ShaderRecompiler::RuntimeAbi::NativeHeap::InvocationAddressDwords - bases;
        for (const auto key : shader.nativeBaseKeys) {
            const auto high = word(key, (key >> ShaderRecompiler::NativeBaseHighShift) & ShaderRecompiler::NativeBaseWordMask);
            const auto guest = word(key, key & ShaderRecompiler::NativeBaseWordMask) | static_cast<std::uint64_t>((key & ShaderRecompiler::NativeBaseVsharp) != 0 ? high & 0xffffu : high) << 32u;
            Native::NotePoolUse((key & ShaderRecompiler::NativeBaseVsharp) != 0 ? "vsharp" : "pointer", guest);
            std::uint64_t extent = 0;
            const auto address = device != nullptr && guest != 0 ? device->DeviceSpan(guest, extent) : 0;
            slot[0] = static_cast<std::uint32_t>(address);
            slot[1] = static_cast<std::uint32_t>(address >> 32u);
            slot[2] = static_cast<std::uint32_t>(std::min<std::uint64_t>(address != 0 ? extent : 0, 0xfffffffcu));
            slot[3] = 0;
            slot += 4;
        }
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
    std::shared_ptr<const ShaderRecompiler::SourceHandle> descriptors;
    std::shared_ptr<const ShaderRecompiler::IrResourcePlan> flattened;
    bool waitedOut = false;
};

struct NativeGraphics {
    std::mutex mutex;
    std::condition_variable ready;
    // Per-draw CPU time by phase (microseconds), reported with the draw counts.
    std::array<double, 5> phaseUs{};
    std::uint64_t frontEnd = 0;
    std::unordered_map<KeyDigest, NativeGraphicsStage, KeyDigestHash> stages;
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

bool TranslationCached(const ShaderRecompiler::RecompileRequest& request) {
    if (!ShaderRecompiler::ShaderDiskCache::Enabled()) return false;
    const bool vertex = request.shader.stage == ShaderRecompiler::ShaderStage::Vertex || request.shader.stage == ShaderRecompiler::ShaderStage::Local;
    std::vector<std::byte> key;
    ShaderRecompiler::ShaderDiskCache::BuildKey(request, vertex && !request.target.vertexSubgroups ? 1u : request.target.subgroupSize, key);
    std::error_code error;
    return std::filesystem::exists(ShaderRecompiler::ShaderDiskCache::EntryDirectory() / ShaderRecompiler::ShaderDiskCache::EntryName(key), error);
}

thread_local bool prefetchingDraws = false;

struct DrawLookahead {
    const Submission* submission = nullptr;
    QueueState* queue = nullptr;
    std::size_t cursor = 0;
    bool done = false;
};
thread_local DrawLookahead drawLookahead;

void RunDrawLookahead();

void CountDrawFallback(const std::string& reason) {
    if (prefetchingDraws) return;
    auto& graphics = GraphicsWork();
    std::lock_guard lock(graphics.mutex);
    ++graphics.fallbacks[reason];
    Native::ReportGraphicsIssue("draw skipped", reason);
}

void ReportGraphics() {
    auto& graphics = GraphicsWork();
    std::lock_guard lock(graphics.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - graphics.reported < std::chrono::seconds(5)) return;
    graphics.reported = now;
    std::uint64_t skipped = 0;
    for (const auto& [reason, count] : graphics.fallbacks) skipped += count;
    static std::uint64_t lastDraws = 0;
    static std::array<double, 5> lastPhases{};
    const double windowDraws = std::max<double>(1.0, static_cast<double>(graphics.native + skipped - lastDraws));
    std::array<double, 5> window{};
    for (std::size_t i = 0; i < window.size(); ++i) window[i] = graphics.phaseUs[i] - lastPhases[i];
    if (Native::Logs()) std::fprintf(stderr, "[native] draws: %llu native, %llu skipped, %zu stage translations; last 5 s: %.0f draws, %.0f ms in DrawNatively, us per draw: state %.1f, programs %.1f, stage keys %.1f, record %.1f, total %.1f\n", static_cast<unsigned long long>(graphics.native), static_cast<unsigned long long>(skipped), graphics.stages.size(), windowDraws, window[4] / 1000.0, window[0] / windowDraws, window[1] / windowDraws, window[2] / windowDraws, window[3] / windowDraws, window[4] / windowDraws);
    static std::uint64_t lastFrontEnd = 0, lastNative = 0;
    if (Native::Logs()) std::fprintf(stderr, "[native] last 5 s: %llu draws prepared at the game's call site, %llu decoded by the executor\n", static_cast<unsigned long long>(graphics.frontEnd - lastFrontEnd), static_cast<unsigned long long>((graphics.native - lastNative) - (graphics.frontEnd - lastFrontEnd)));
    lastFrontEnd = graphics.frontEnd;
    lastNative = graphics.native;
    if (Native::Logs() && skipped != 0) {
        std::vector<std::pair<std::uint64_t, std::string>> reasons;
        for (const auto& [reason, count] : graphics.fallbacks) reasons.emplace_back(count, reason);
        std::sort(reasons.rbegin(), reasons.rend());
        std::string line;
        for (std::size_t i = 0; i < reasons.size() && i < 6; ++i) line += " " + std::to_string(reasons[i].first) + " " + reasons[i].second.substr(0, 90) + ";";
        std::fprintf(stderr, "[native] skipped draws by reason:%s\n", line.c_str());
    }
    lastDraws = graphics.native + skipped;
    lastPhases = graphics.phaseUs;
}

thread_local std::uint32_t frontEndDraw = 0;

struct PreparedDraw {
    Graphics::State state;
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> results;
    std::shared_ptr<const ShaderRecompiler::RectListShaders> rectangle;
    Native::NativeDevice::DrawWork work;
    // The packet as GT7's call wrote it: a draw whose packet changed before it ran is reported.
    std::vector<std::uint32_t> packet;
    bool ready = false;
};

bool RecordPreparedDraw(PreparedDraw& prepared) {
    auto* device = Native::NativeDevice::Get();
    std::string failure;
    if (device == nullptr || !device->Draw(prepared.work, failure)) {
        CountDrawFallback(device == nullptr ? std::string("no native device yet") : failure);
        return false;
    }
    {
        auto& graphics = GraphicsWork();
        std::lock_guard lock(graphics.mutex);
        ++graphics.native;
        ++graphics.frontEnd;
    }
    ReportGraphics();
    return true;
}

bool DrawNatively(QueueState& queue, std::span<const std::uint32_t> packet, bool prefetch = false, PreparedDraw* prepareOnly = nullptr) {
    if (!prefetch && prepareOnly == nullptr && frontEndDraw != 0) {
        const auto id = frontEndDraw;
        frontEndDraw = 0;
        if (Native::FrontEndValidating()) Native::CompareFrontEndDraw(id, queue);
        if (auto prepared = std::static_pointer_cast<PreparedDraw>(Native::TakeFrontEndDraw(id)); prepared != nullptr && prepared->ready) {
            if (!std::equal(prepared->packet.begin(), prepared->packet.end(), packet.begin(), packet.end())) {
                char detail[160];
                std::snprintf(detail, sizeof(detail), "opcode 0x%x: words 1..%zu %08x %08x %08x %08x", (packet[0] >> 8u) & 0xffu, packet.size() - 1, packet.size() > 1 ? packet[1] : 0u, packet.size() > 2 ? packet[2] : 0u, packet.size() > 3 ? packet[3] : 0u, packet.size() > 4 ? packet[4] : 0u);
                Native::ReportGraphicsIssue("draw packet changed after its call (prepared values used)", detail);
            }
            return RecordPreparedDraw(*prepared);
        }
    }
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
        // Indirect draws run on the GPU from the records GT7's earlier work wrote (vkCmdDrawIndexed
        // Indirect[Count]); the index buffer is the declared INDEX_BUFFER_SIZE range.
        if (!draw.indirect && (draw.indexCount == 0 || draw.instanceCount == 0)) return true;
        state = Graphics::DecodeState(queue, false, true);
        // Debug aid: APS5_DEBUG_DEPTH_ALWAYS=<color target address> draws into that target without
        // depth testing or writing (separates depth rejection from shading in a black pass).
        static const std::uint64_t depthAlways = [] { const char* text = std::getenv("APS5_DEBUG_DEPTH_ALWAYS"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        if (depthAlways != 0 && !state.colors.empty() && state.colors.front().address == depthAlways) {
            state.depthCompare = VK_COMPARE_OP_ALWAYS;
            state.depthWrite = false;
        }
    } catch (const std::exception& error) {
        static const bool debugReject = std::getenv("APS5_DEBUG_DEPTH_REJECT") != nullptr;
        static std::atomic<int> rejectLogs{0};
        if (debugReject && std::strstr(error.what(), "register 0x200") != nullptr && rejectLogs.fetch_add(1) < 60) std::fprintf(stderr, "[native] state rejected (%s): %s\n", prepareOnly != nullptr ? "call site" : prefetch ? "lookahead" : "executor", error.what());
        CountDrawFallback(std::string("state: ") + error.what());
        return false;
    }
    // Debug aid: APS5_DEBUG_WATCH_TARGET=<address,...> logs draws rendering into a color or depth
    // target within 256 MB above a listed address (before any skip), with the time since the first such draw.
    {
        static const auto watched = AddressList("APS5_DEBUG_WATCH_TARGET");
        static const auto started = std::chrono::steady_clock::now();
        static std::atomic<int> logged{0};
        const auto listed = [&](std::uint64_t address) { return std::any_of(watched.begin(), watched.end(), [&](std::uint64_t base) { return address >= base && address - base < (std::uint64_t{1} << 28u); }); };
        bool hit = state.depth && listed(state.depth->address);
        for (const auto& color : state.colors) hit = hit || listed(color.address);
        // APS5_DEBUG_WATCH_TARGET=all: each distinct depth target once.
        static const bool all = [] { const char* text = std::getenv("APS5_DEBUG_WATCH_TARGET"); return text != nullptr && std::strcmp(text, "all") == 0; }();
        static std::mutex seenMutex;
        static std::set<std::uint64_t> seen;
        if (all && state.depth) {
            std::lock_guard lock(seenMutex);
            hit = seen.insert(state.depth->address ^ (static_cast<std::uint64_t>(prepareOnly != nullptr) << 63u)).second;
        }
        // APS5_DEBUG_PROGRAM_HEADS=<dir>: each watched draw's pixel program's first 64 code bytes,
        // written once as <dir>/<address>.bin (named offline by tools/name_programs.py), and up to
        // 64 KiB of its code as <address>.code.
        static const char* heads = std::getenv("APS5_DEBUG_PROGRAM_HEADS");
        if (heads != nullptr && hit) {
            const auto ps = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x8)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x9) & 0xffu) << 40u);
            static std::mutex headsMutex;
            static std::set<std::uint64_t> written;
            std::lock_guard lock(headsMutex);
            if (ps != 0 && written.insert(ps).second && GuestMemory::Accessible(reinterpret_cast<const void*>(ps), 64)) {
                std::filesystem::create_directories(heads);
                char name[32];
                std::snprintf(name, sizeof(name), "%llx.bin", static_cast<unsigned long long>(ps));
                std::ofstream(std::filesystem::path(heads) / name, std::ios::binary).write(reinterpret_cast<const char*>(ps), 64);
                // The whole program too (up to 64 KiB of readable code), for offline disassembly.
                std::size_t bytes = 64;
                while (bytes < 65536 && GuestMemory::Accessible(reinterpret_cast<const void*>(ps), bytes * 2)) bytes *= 2;
                std::snprintf(name, sizeof(name), "%llx.code", static_cast<unsigned long long>(ps));
                std::ofstream(std::filesystem::path(heads) / name, std::ios::binary).write(reinterpret_cast<const char*>(ps), static_cast<std::streamsize>(bytes));
            }
        }
        if ((all || !watched.empty()) && hit && logged.fetch_add(1) < 300) {
            std::string masks;
            for (std::size_t i = 0; i < state.colors.size(); ++i) {
                char item[48];
                std::snprintf(item, sizeof(item), " mrt%zu 0x%llx mask %x blend %u", i, static_cast<unsigned long long>(state.colors[i].address), i < state.blends.size() ? state.blends[i].colorWriteMask : 0u, i < state.blends.size() ? state.blends[i].blendEnable : 0u);
                masks += item;
            }
            std::fprintf(stderr, "[native] watch targets:%s\n", masks.c_str());
            std::fprintf(stderr, "[native] watch target (%s): color 0x%llx depth 0x%llx %ux%u count %u x%u path %d pixelless %d test %d write %d compare %d clear %g vs 0x%llx ps 0x%llx at %.2f s\n", prepareOnly != nullptr ? "call site" : "executor", static_cast<unsigned long long>(state.colors.empty() ? 0ull : state.colors.front().address), static_cast<unsigned long long>(state.depth ? state.depth->address : 0ull),
                state.renderExtent.width, state.renderExtent.height, draw.indexCount, draw.instanceCount, static_cast<int>(state.stages.path), Graphics::PixelProgramSkipped(queue) ? 1 : 0, state.depthTest, state.depthWrite, static_cast<int>(state.depthCompare), state.depth ? state.depth->clearDepth : -1.0f,
                static_cast<unsigned long long>((static_cast<std::uint64_t>(readRegister(queue.shader, 0xc8)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0xc9) & 0xffu) << 40u)), static_cast<unsigned long long>((static_cast<std::uint64_t>(readRegister(queue.shader, 0x8)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x9) & 0xffu) << 40u)), std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }
    }
    // Debug experiment: APS5_DEBUG_SKIP_DEPTH=<low>-<high> drops draws whose depth target lies in that
    // range (shadow casters, to tell shadowing apart from other lighting inputs). Not a fix.
    {
        static const std::pair<std::uint64_t, std::uint64_t> skipDepth = [] {
            std::pair<std::uint64_t, std::uint64_t> range{0, 0};
            if (const char* text = std::getenv("APS5_DEBUG_SKIP_DEPTH")) {
                char* end = nullptr;
                range.first = std::strtoull(text, &end, 0);
                if (*end == '-') range.second = std::strtoull(end + 1, nullptr, 0);
            }
            return range;
        }();
        if (skipDepth.second != 0 && state.depth && state.depth->address >= skipDepth.first && state.depth->address < skipDepth.second) return true;
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
    if (!prefetch) {
        static const std::uint64_t dumpProgram = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_PS"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        static const double dumpAt = [] { const char* text = std::getenv("APS5_NATIVE_DUMP_TARGETS_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto started = std::chrono::steady_clock::now();
        static bool dumped = false;
        if (dumpProgram != 0 && !dumped && std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= dumpAt) {
            for (const auto& program : programs) {
                if (program.binary.codeAddress != dumpProgram && program.recorded.codeAddress != dumpProgram) continue;
                dumped = true;
                const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR");
                const std::filesystem::path root = directory != nullptr ? directory : "native-targets";
                std::filesystem::create_directories(root);
                std::ofstream list(root / "program-inputs.txt");
                for (const auto& stage : programs) {
                    list << "stage " << static_cast<int>(stage.stage) << " program 0x" << std::hex << stage.binary.codeAddress << " user data";
                    for (const auto word : stage.userData) list << " " << word;
                    list << std::dec << "\n";
                    for (std::size_t i = 0; i + 1 < stage.userData.size(); ++i) {
                        const auto address = static_cast<std::uint64_t>(stage.userData[i]) | (static_cast<std::uint64_t>(stage.userData[i + 1] & 0xffffu) << 32u);
                        if (address < 0x100000000ull || (address & 3u) != 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), 4096)) continue;
                        char name[96];
                        std::snprintf(name, sizeof(name), "stage%d_user%zu_%llx.bin", static_cast<int>(stage.stage), i, static_cast<unsigned long long>(address));
                        std::ofstream(root / name, std::ios::binary).write(reinterpret_cast<const char*>(address), 4096);
                        list << "  user[" << i << "] -> 0x" << std::hex << address << std::dec << " " << name << "\n";
                    }
                }
                break;
            }
        }
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
    std::vector<std::shared_ptr<const ShaderRecompiler::IrResourcePlan>> flattenedPlans;
    std::vector<std::uint32_t> offsets;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> liveVertex;
    std::vector<std::uint64_t> sampled;
    std::uint32_t pushOffset = 0;
    for (const bool launchOnly : {true, false}) {
    if (!launchOnly && prefetch) return true;
    pushOffset = 0;
    for (const auto& program : programs) {
        const bool fragment = program.stage == Stage::Fragment;
        const auto wave = fragment ? state.stages.fragmentWaveSize : state.stages.vertexWaveSize;
        // The vertex buffers of this draw (GT7's fetch tables, read live) and the same with only their
        // formats, which is what the translation depends on.
        std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex, vertexKey;
        try {
            if (!fragment) {
                vertex = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, nullptr, program.stage == Stage::Mesh, 8u - program.firstUserSgpr);
                const auto layout = readRegister(queue.context, 0x207) & Graphics::PositionExportLayoutBits;
                if (layout != 0u) {
                    auto& cache = *program.recorded.positionVectorUse;
                    const bool cacheable = program.binary.codeAddress == program.recorded.codeAddress;
                    auto use = cacheable ? cache.load(std::memory_order_acquire) : 0u;
                    if (use == 0u) {
                        use = PositionExportLayout(nullptr, 0, program.binary.code, layout) != 0u ? 2u : 1u;
                        if (cacheable) cache.store(use, std::memory_order_release);
                    }
                    vertex->paClVsOutCntl = use == 2u ? layout : 0u;
                }
                vertex->paClVsOutCntl |= state.layerExports;
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
        // The translation's key within this device: the request without the device's own target.
        static const ShaderRecompiler::SpirvTarget keyTarget{};
        std::vector<std::uint64_t> key;
        {
            const ShaderRecompiler::RecompileRequest keyed{program.binary, {wave, program.firstUserSgpr, program.userData, std::nullopt, fragment ? std::optional(pixel) : std::nullopt, vertexKey, {}, floatMode}, keyTarget, {0, 0, pushOffset, room}, ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, state.stages.mesh, state.stages.tessellation, {}}};
            ShaderRecompiler::BuildPreparedShaderKey(keyed, key);
        }
        key.push_back(program.binary.codeAddress);
        key.push_back(pushOffset);
        const auto digest = Digest(key);
        auto& graphics = GraphicsWork();
        std::unique_lock lock(graphics.mutex);
        auto found = graphics.stages.find(digest);
        if (found == graphics.stages.end()) {
            // A worker owns the linked-program copies. Preserve this draw while
            // it translates: a one-time producer must never be discarded.
            NativeGraphicsStage pending;
            pending.failure = "draw stage still translating";
            found = graphics.stages.emplace(digest, std::move(pending)).first;
            lock.unlock();
            const auto index = static_cast<std::size_t>(&program - programs.data());
            std::thread([programs, index, wave, pixel, vertex, target, pushOffset, room, floatMode, mesh = state.stages.mesh, tessellation = state.stages.tessellation, key, digest, job = std::make_shared<Native::CompileJob>()] {
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
                if (!TranslationCached(request)) job->Compiling();
                try {
                    const auto began = std::chrono::steady_clock::now();
                    made.result = std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
                    made.flattened = FlattenedPlan(request, *made.result);
                    made.descriptors = ResolveDescriptors(ClassicRequest(request));
                    const auto recompiled = std::chrono::steady_clock::now();
                    if (auto* native = Native::NativeDevice::Get(); native != nullptr) {
                        try {
                            static_cast<void>(native->PrebuildGraphicsLibrary(*made.result, own.stage, wave, pushOffset, mesh));
                        } catch (const std::exception&) {
                        }
                    }
                    const auto ms = [](auto from, auto to) { return std::chrono::duration<double, std::milli>(to - from).count(); };
                    const auto finished = std::chrono::steady_clock::now();
                    if (Native::Logs() && ms(began, finished) >= 500.0) std::fprintf(stderr, "[native] slow draw stage %llx (stage %d, %zu SPIR-V words): translate %.0f ms, library %.0f ms\n", static_cast<unsigned long long>(own.binary.codeAddress), static_cast<int>(own.stage), made.result->spirv.size(), ms(began, recompiled), ms(recompiled, finished));
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
                graphics.stages[digest] = std::move(made);
                graphics.ready.notify_all();
            }).detach();
            lock.lock();
            found = graphics.stages.find(digest);
            if (!prefetch && !drawLookahead.done) {
                lock.unlock();
                RunDrawLookahead();
                lock.lock();
            }
        }
        if (launchOnly) continue;
        {
            const Native::CompileWait waiting;
            const auto translated = [&] { return graphics.stages.at(digest).failure != "draw stage still translating"; };
            if (Native::CompileWaitLimit().count() == 0) graphics.ready.wait(lock, translated);
            else if (graphics.stages.at(digest).waitedOut ? !translated() : !graphics.ready.wait_for(lock, Native::CompileWaitNow(), translated)) {
                graphics.stages.at(digest).waitedOut = true;
                Native::NoteCompileWaitTimedOut();
                lock.unlock();
                CountDrawFallback("draw stage still compiling");
                return false;
            }
        }
        found = graphics.stages.find(digest);
        if (!found->second.failure.empty()) {
            const auto failure = found->second.failure;
            lock.unlock();
            CountDrawFallback(failure);
            return false;
        }
        if (found->second.descriptors != nullptr && !SlotDescriptors()) {
            const auto descriptors = found->second.descriptors;
            lock.unlock();
            RegisterSampledDescriptors(request, true, *descriptors, state.colors.empty() ? 0ull : state.colors.front().address, false, &sampled);
            lock.lock();
            found = graphics.stages.find(digest);
        }
        results.push_back(found->second.result);
        flattenedPlans.push_back(found->second.flattened);
        offsets.push_back(pushOffset);
    }
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
        stage.flattenedSrt = FlattenedSrt(flattenedPlans[i].get(), shader, programs[i].userData, programs[i].binary.codeAddress, stage.slotWords);
        stage.shaderData = ShaderDataFor(shader, offsets[i], programs[i].userData, programs[i].firstUserSgpr, stage.flattenedSrt);
        for (std::size_t image = 0; image < shader.nativeImageSources.size() && stage.slotWords.size() >= (image + 1) * 8; ++image) {
            const auto* words = stage.slotWords.data() + image * 8;
            if (const auto base = (words[0] | (static_cast<std::uint64_t>(words[1] & 0xffu) << 32u)) << 8u; base != 0) sampled.push_back(base);
        }
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
        // Debug aid: APS5_DEBUG_PS_DATA=<pixel program address> prints, for its first two draws, the
        // pixel stage's user data and flattened SRT, and 48 floats behind every user-data pointer pair.
        static const std::uint64_t debugPixel = [] { const char* text = std::getenv("APS5_DEBUG_PS_DATA"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        // APS5_DEBUG_PS_DATA_TARGET=<color target>: the first pixel program loaded above the eboot
        // bundles (a course material) drawing into that target from APS5_DEBUG_CS_DATA_AT seconds instead.
        static const std::uint64_t debugTarget = [] { const char* text = std::getenv("APS5_DEBUG_PS_DATA_TARGET"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        static const double debugAt = [] { const char* text = std::getenv("APS5_DEBUG_CS_DATA_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto debugStarted = std::chrono::steady_clock::now();
        static std::uint64_t chosenPixel = 0;
        static int debugLeft = 2;
        // APS5_DEBUG_PS_EXPOSURE=<color target>: for each distinct pixel program drawing into it, the
        // exposure triple its pass block carries (block +0x440), which scales everything it writes.
        {
            static const std::uint64_t exposureTarget = [] { const char* text = std::getenv("APS5_DEBUG_PS_EXPOSURE"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
            static std::mutex exposureMutex;
            static std::set<std::uint64_t> exposureSeen;
            if (exposureTarget != 0 && !state.colors.empty() && state.colors.front().address == exposureTarget && !programs.empty() && programs.back().stage == Stage::Fragment) {
                const auto& pixelProgram = programs.back();
                std::lock_guard lock(exposureMutex);
                if (exposureSeen.size() < 24 && exposureSeen.insert(pixelProgram.binary.codeAddress).second && pixelProgram.userData.size() >= 2) {
                    const auto block = pixelProgram.userData[0] | (static_cast<std::uint64_t>(pixelProgram.userData[1] & 0xffffu) << 32u);
                    float exposure[4] = {-1, -1, -1, -1};
                    const bool readable = block >= 0x1000000000ull && block <= 0xffffffffffull && GuestMemory::Accessible(reinterpret_cast<const void*>(block + 0x440), 16);
                    if (readable) std::memcpy(exposure, reinterpret_cast<const void*>(block + 0x440), sizeof(exposure));
                    std::fprintf(stderr, "[native] ps 0x%llx block 0x%llx exposure %g %g %g (w %g)%s\n", static_cast<unsigned long long>(pixelProgram.binary.codeAddress),
                        static_cast<unsigned long long>(block), exposure[0], exposure[1], exposure[2], exposure[3], readable ? "" : " (block unreadable)");
                }
            }
        }
        // Debug experiment: APS5_DEBUG_SKIP_PROGRAM=<code hash,...> drops the draws of those pixel
        // programs, to see what a pass contributes. Not a fix.
        {
            static const std::vector<std::uint64_t> skipped = [] {
                std::vector<std::uint64_t> values;
                if (const char* text = std::getenv("APS5_DEBUG_SKIP_PROGRAM")) {
                    while (*text != '\0') {
                        char* end = nullptr;
                        const auto value = std::strtoull(text, &end, 16);
                        if (end == text) break;
                        values.push_back(value);
                        text = *end == ',' ? end + 1 : end;
                    }
                }
                return values;
            }();
            if (!skipped.empty() && !programs.empty() && programs.back().stage == Stage::Fragment) {
                const auto code = programs.back().binary.code;
                const auto head = code.subspan(0, std::min<std::size_t>(std::size_t{16}, code.size()));
                if (std::find(skipped.begin(), skipped.end(), ShaderRecompiler::HashBytes(std::as_bytes(head))) != skipped.end()) return true;
            }
        }
        // Debug experiment: APS5_DEBUG_SKIP_DST_ALPHA=<color target> drops the draws into that target
        // that multiply the destination by the source alpha (blend ZERO, SRC_ALPHA), to see whether
        // such a pass is what erases the target. Not a fix.
        {
            static const std::uint64_t wanted = [] { const char* text = std::getenv("APS5_DEBUG_SKIP_DST_ALPHA"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
            if (wanted != 0 && !state.colors.empty() && state.colors.front().address == wanted && !state.blends.empty()) {
                const auto& blend = state.blends.front();
                if (blend.blendEnable && blend.srcColorBlendFactor == VK_BLEND_FACTOR_ZERO && blend.dstColorBlendFactor == VK_BLEND_FACTOR_SRC_ALPHA) {
                    static std::atomic<int> noted{0};
                    if (noted.fetch_add(1) < 4) std::fprintf(stderr, "[native] skipping a destination-alpha multiply into 0x%llx (ps 0x%llx)\n", static_cast<unsigned long long>(wanted), static_cast<unsigned long long>(programs.empty() ? 0ull : programs.back().binary.codeAddress));
                    return true;
                }
            }
        }
        // APS5_DEBUG_PROGRAM_IDS=1: the identity APS5_PROBE_MARK_ALL writes for each pixel program,
        // with its address, so a dumped identity map can be read back to programs.
        {
            static const bool wantIds = std::getenv("APS5_DEBUG_PROGRAM_IDS") != nullptr;
            if (wantIds && !programs.empty() && programs.back().stage == Stage::Fragment) {
                const auto& pixelProgram = programs.back();
                static std::mutex idMutex;
                static std::set<std::uint64_t> idSeen;
                std::lock_guard lock(idMutex);
                if (idSeen.size() < 400 && idSeen.insert(pixelProgram.binary.codeAddress).second) {
                    const auto code = pixelProgram.binary.code;
                    const auto head = code.subspan(0, std::min<std::size_t>(std::size_t{16}, code.size()));
                    const auto hash = ShaderRecompiler::HashBytes(std::as_bytes(head));
                    const auto& blend = state.blends.empty() ? Graphics::State{}.blends.emplace_back() : state.blends.front();
                    std::fprintf(stderr, "[native] program id %llu: ps 0x%llx hash %016llx target 0x%llx mrt0 mask %x blend %d (src %d dst %d op %d) targets %zu depth test %d write %d\n",
                        static_cast<unsigned long long>(hash & 0x7ffull), static_cast<unsigned long long>(pixelProgram.binary.codeAddress), static_cast<unsigned long long>(hash),
                        static_cast<unsigned long long>(state.colors.empty() ? 0ull : state.colors.front().address), blend.colorWriteMask, blend.blendEnable ? 1 : 0,
                        static_cast<int>(blend.srcColorBlendFactor), static_cast<int>(blend.dstColorBlendFactor), static_cast<int>(blend.colorBlendOp),
                        state.colors.size(), state.depthTest ? 1 : 0, state.depthWrite ? 1 : 0);
                    std::string mapping;
                    for (const auto& color : state.colors) {
                        char item[72];
                        std::snprintf(item, sizeof(item), " export%u->0x%llx(fmt %d mask %x)", color.exportIndex, static_cast<unsigned long long>(color.address), static_cast<int>(color.format),
                            color.exportIndex < state.blends.size() ? state.blends[color.exportIndex].colorWriteMask : 0u);
                        mapping += item;
                    }
                    std::fprintf(stderr, "[native]   id %llu targets:%s\n", static_cast<unsigned long long>(hash & 0x7ffull), mapping.c_str());
                }
            }
        }
        static int seenTarget = 0;
        if (debugTarget != 0 && !state.colors.empty() && state.colors.front().address == debugTarget && seenTarget < 3) {
            ++seenTarget;
            std::fprintf(stderr, "[native] ps data target draw: %zu programs, last stage %d, last program 0x%llx, %.1f s\n", programs.size(), programs.empty() ? -1 : static_cast<int>(programs.back().stage), programs.empty() ? 0ull : static_cast<unsigned long long>(programs.back().binary.codeAddress), std::chrono::duration<double>(std::chrono::steady_clock::now() - debugStarted).count());
        }
        if (debugTarget != 0 && chosenPixel == 0 && !programs.empty() && programs.back().stage == Stage::Fragment && !state.colors.empty() && state.colors.front().address == debugTarget && programs.back().binary.codeAddress >= 0x1259000000ull && programs.back().binary.codeAddress < 0x2000000000ull && std::chrono::duration<double>(std::chrono::steady_clock::now() - debugStarted).count() >= debugAt) chosenPixel = programs.back().binary.codeAddress;
        if ((debugPixel != 0 || chosenPixel != 0) && debugLeft > 0 && !programs.empty() && programs.back().stage == Stage::Fragment && programs.back().binary.codeAddress == (debugPixel != 0 ? debugPixel : chosenPixel)) {
            --debugLeft;
            const auto& pixelProgram = programs.back();
            std::string words;
            for (const auto word : pixelProgram.userData) {
                char item[12];
                std::snprintf(item, sizeof(item), " %08x", word);
                words += item;
            }
            std::fprintf(stderr, "[native] ps 0x%llx user data (first sgpr %u):%s\n", static_cast<unsigned long long>(pixelProgram.binary.codeAddress), pixelProgram.firstUserSgpr, words.c_str());
            for (const auto& stage : work.stages) {
                if (stage.stage != Stage::Fragment) continue;
                std::string flat;
                for (const auto word : stage.flattenedSrt) {
                    char item[12];
                    std::snprintf(item, sizeof(item), " %08x", word);
                    flat += item;
                }
                std::fprintf(stderr, "[native] ps flattened srt (%zu words):%s\n", stage.flattenedSrt.size(), flat.c_str());
            }
            for (std::size_t i = 0; i + 1 < pixelProgram.userData.size(); ++i) {
                const auto address = pixelProgram.userData[i] | (static_cast<std::uint64_t>(pixelProgram.userData[i + 1] & 0xffffu) << 32u);
                static const int words = [] { const char* text = std::getenv("APS5_DEBUG_PS_DATA_WORDS"); return text != nullptr ? std::max(48, std::atoi(text)) : 48; }();
                if (address < 0x1000000000ull || address > 0xffffffffffull || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(words) * 4u)) continue;
                std::string values;
                for (int k = 0; k < words; ++k) {
                    std::uint32_t word;
                    std::memcpy(&word, reinterpret_cast<const void*>(address + k * 4), 4);
                    char item[16];
                    std::snprintf(item, sizeof(item), " %08x", word);
                    values += item;
                }
                std::fprintf(stderr, "[native] ps user pointer [%zu] 0x%llx:%s\n", i, static_cast<unsigned long long>(address), values.c_str());
                // The V#s in the block: up to 64 dwords of each buffer (hex, then the float value).
                for (int k = 0; k + 3 < 48; k += 4) {
                    std::array<std::uint32_t, 4> v{};
                    std::memcpy(v.data(), reinterpret_cast<const void*>(address + k * 4), sizeof(v));
                    const auto base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xffffu) << 32u);
                    const auto count = std::min<std::uint32_t>(64u, std::max<std::uint32_t>(1u, ((v[1] >> 16u) & 0x3fffu) * v[2] / 4u));
                    if (v[2] == 0 || (v[3] & 0xfffu) == 0 || base < 0x1000000000ull || base > 0xffffffffffull || !GuestMemory::Accessible(reinterpret_cast<const void*>(base), count * 4u)) continue;
                    std::string buffer;
                    for (std::uint32_t w = 0; w < count; ++w) {
                        std::uint32_t word;
                        std::memcpy(&word, reinterpret_cast<const void*>(base + w * 4), 4);
                        float value;
                        std::memcpy(&value, &word, 4);
                        char item[32];
                        std::snprintf(item, sizeof(item), " %08x(%g)", word, value);
                        buffer += item;
                    }
                    std::fprintf(stderr, "[native]   ps V# at [%zu]+0x%x 0x%llx:%s\n", i, k * 4, static_cast<unsigned long long>(base), buffer.c_str());
                }
            }
        }
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
    work.sampled = std::move(sampled);
    work.programAddresses = {programs.front().binary.codeAddress, programs.back().stage == Stage::Fragment ? programs.back().binary.codeAddress : 0u};
    // The vertex fetch folds its base-vertex and base-instance user data into the draw (the
    // recompiler subtracts them from the vertex and instance index): the draw adds them, as the CP
    // adds the draw offsets it writes to those SGPRs. An indirect record supplies them instead.
    const auto& front = programs.front();
    const auto userWord = [&](std::int32_t sgpr) -> std::optional<std::uint32_t> {
        if (sgpr < static_cast<std::int32_t>(front.firstUserSgpr)) return std::nullopt;
        const auto index = static_cast<std::size_t>(sgpr) - front.firstUserSgpr;
        return index < front.userData.size() ? std::optional(front.userData[index]) : std::nullopt;
    };
    const auto& fetch = *results.front();
    if (draw.indirect) {
        // The record's vertexOffset and firstInstance reach the shader as the GPU's base vertex and
        // base instance: correct when the SGPRs the CP would write are the ones the fetch folds.
        const auto& indirect = *draw.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            if (location == 0x280u || location < front.userDataBase) return -1;
            const auto word = location - front.userDataBase + (8u - front.firstUserSgpr);
            return word < front.userData.size() ? static_cast<std::int32_t>(front.firstUserSgpr + word) : -1;
        };
        const auto vertexSgpr = sgprOf(indirect.baseVertexLocation), instanceSgpr = sgprOf(indirect.startInstanceLocation);
        const char* unsupported = nullptr;
        if (!draw.indexed) unsupported = "indirect draw: non-indexed";
        else if (indirect.recordBytes != 20) unsupported = "indirect draw: record layout";
        else if (vertexSgpr != fetch.vertexOffsetSgpr) unsupported = "indirect draw: base vertex reaches the shader outside its vertex fetch";
        else if (instanceSgpr != fetch.instanceOffsetSgpr) unsupported = "indirect draw: start instance reaches the shader outside its vertex fetch";
        else if (indirect.drawIndexEnabled && sgprOf(indirect.drawIndexLocation) >= 0) unsupported = "indirect draw: draw index";
        else if (state.stages.mesh || state.stages.tessellation) unsupported = "indirect draw: mesh or tessellation path";
        if (unsupported != nullptr) {
            CountDrawFallback(unsupported);
            return false;
        }
        work.indirectArguments = indirect.arguments;
        work.indirectCount = indirect.count;
        work.indirectStride = indirect.stride;
        work.indirectCountAddress = indirect.countIndirect ? indirect.countAddress : 0u;
    } else {
        if (fetch.vertexOffsetSgpr >= 0) draw.firstVertex += userWord(fetch.vertexOffsetSgpr).value_or(0u);
        if (fetch.instanceOffsetSgpr >= 0) draw.firstInstance = userWord(fetch.instanceOffsetSgpr).value_or(0u);
    }
    work.indexed = draw.indexed;
    work.indexAddress = draw.indexAddress;
    work.indexSize = draw.indexSize;
    work.count = draw.indexCount;
    work.instances = draw.instanceCount;
    work.firstVertex = draw.firstVertex;
    work.firstInstance = draw.firstInstance;
    if (prepareOnly != nullptr) {
        prepareOnly->state = std::move(state);
        prepareOnly->results = std::move(results);
        prepareOnly->rectangle = std::move(rectangle);
        prepareOnly->work = std::move(work);
        prepareOnly->work.state = &prepareOnly->state;
        prepareOnly->ready = true;
        return true;
    }
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
struct PreparedDispatch {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    std::uint64_t key = 0;
    std::array<std::uint32_t, 3> groups{};
    std::vector<std::uint32_t> shaderData;
    std::uint64_t address = 0;
    std::uint64_t indirectArguments = 0;
    bool ready = false;
    std::vector<std::uint32_t> flattenedSrt;
    std::vector<std::uint32_t> slotWords;
    bool skip = false;
};

bool RecordPreparedDispatch(PreparedDispatch& prepared) {
    if (prepared.skip) return true;
    auto* device = Native::NativeDevice::Get();
    std::string failure;
    if (device == nullptr || !device->Dispatch({prepared.result.get(), prepared.key, prepared.groups, prepared.shaderData, prepared.address, prepared.indirectArguments, prepared.flattenedSrt, prepared.slotWords}, failure)) {
        CountFallback(device == nullptr ? std::string("no native device yet") : failure);
        return false;
    }
    {
        auto& compute = Compute();
        std::lock_guard lock(compute.mutex);
        ++compute.native;
        ++compute.frontEnd;
    }
    ReportCompute();
    return true;
}

bool DispatchNatively(QueueState& queue, std::span<const std::uint32_t> packet, bool indirect = false, PreparedDispatch* prepareOnly = nullptr) {
    if (prepareOnly == nullptr && frontEndDraw != 0) {
        const auto id = frontEndDraw;
        frontEndDraw = 0;
        if (auto prepared = std::static_pointer_cast<PreparedDispatch>(Native::TakeFrontEndDraw(id)); prepared != nullptr && prepared->ready) return RecordPreparedDispatch(*prepared);
    }
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
    // Debug experiment: APS5_DEBUG_SKIP_DISPATCH=<code hash or address,...> drops those compute
    // dispatches, to see what a pass contributes to the frame. Not a fix.
    {
        static const std::vector<std::uint64_t> skipped = [] {
            std::vector<std::uint64_t> values;
            if (const char* text = std::getenv("APS5_DEBUG_SKIP_DISPATCH")) {
                while (*text != '\0') {
                    char* end = nullptr;
                    const auto value = std::strtoull(text, &end, 16);
                    if (end == text) break;
                    values.push_back(value);
                    text = *end == ',' ? end + 1 : end;
                }
            }
            return values;
        }();
        if (!skipped.empty() && program.code != nullptr) {
            const std::span<const std::uint32_t> code{*program.code};
            const auto head = code.subspan(0, std::min<std::size_t>(std::size_t{16}, code.size()));
            const auto hash = ShaderRecompiler::HashBytes(std::as_bytes(head));
            if (std::find(skipped.begin(), skipped.end(), hash) != skipped.end() || std::find(skipped.begin(), skipped.end(), address) != skipped.end()) {
                static std::mutex skipMutex;
                static std::set<std::uint64_t> skipSeen;
                std::lock_guard lock(skipMutex);
                if (skipSeen.insert(address).second) std::fprintf(stderr, "[native] skipping dispatches of cs 0x%llx (hash %016llx)\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(hash));
                if (prepareOnly != nullptr) {
                    prepareOnly->skip = true;
                    prepareOnly->ready = true;
                }
                return true;
            }
        }
    }
    // Debug aid: APS5_DEBUG_CS_DATA=<compute program address> prints, for its first two dispatches
    // from APS5_DEBUG_CS_DATA_AT seconds, the user data, 64 words behind each user-data pointer, and 32 floats behind each V# found there.
    {
        static const std::uint64_t debugCompute = [] { const char* text = std::getenv("APS5_DEBUG_CS_DATA"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
        // APS5_DEBUG_CS_POKE=<program>:<srt byte offset of a V#>:<byte offset>:<word> writes that word into
        // the buffer before each of the program's dispatches (a debug experiment, not a fix).
        static const std::array<std::uint64_t, 4> poke = [] {
            std::array<std::uint64_t, 4> values{};
            if (const char* text = std::getenv("APS5_DEBUG_CS_POKE")) {
                char* end = nullptr;
                for (auto& value : values) {
                    value = std::strtoull(text, &end, 0);
                    if (*end != ':') break;
                    text = end + 1;
                }
            }
            return values;
        }();
        if (poke[0] != 0 && address == poke[0]) {
            const auto srt = DriverDetail::readUserData(queue.shader, 0x24cu) | (static_cast<std::uint64_t>(DriverDetail::readUserData(queue.shader, 0x24du) & 0xffffu) << 32u);
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(srt + poke[1]), 16)) {
                const auto* v = reinterpret_cast<const std::uint32_t*>(srt + poke[1]);
                const auto base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xffffu) << 32u);
                if (GuestMemory::Accessible(reinterpret_cast<const void*>(base + poke[2]), 4)) *reinterpret_cast<std::uint32_t*>(base + poke[2]) = static_cast<std::uint32_t>(poke[3]);
            }
        }
        static std::atomic<int> debugLeft{2};
        static const double debugAt = [] { const char* text = std::getenv("APS5_DEBUG_CS_DATA_AT"); return text != nullptr ? std::atof(text) : 0.0; }();
        static const auto debugStarted = std::chrono::steady_clock::now();
        if (debugCompute != 0 && address == debugCompute && std::chrono::duration<double>(std::chrono::steady_clock::now() - debugStarted).count() >= debugAt && debugLeft.fetch_sub(1) > 0) {
            std::array<std::uint32_t, 16> user{};
            std::string words;
            for (std::uint32_t i = 0; i < user.size(); ++i) {
                user[i] = DriverDetail::readUserData(queue.shader, 0x240u + i);
                char item[12];
                std::snprintf(item, sizeof(item), " %08x", user[i]);
                words += item;
            }
            std::fprintf(stderr, "[native] cs 0x%llx user data:%s\n", static_cast<unsigned long long>(address), words.c_str());
            const auto dumpFloats = [](const char* label, std::uint64_t at, int count) {
                if (at < 0x1000000000ull || at > 0xffffffffffull || !GuestMemory::Accessible(reinterpret_cast<const void*>(at), static_cast<std::size_t>(count) * 4u)) return false;
                std::string values;
                for (int k = 0; k < count; ++k) {
                    std::uint32_t word;
                    std::memcpy(&word, reinterpret_cast<const void*>(at + k * 4), 4);
                    float value;
                    std::memcpy(&value, &word, 4);
                    char item[32];
                    std::snprintf(item, sizeof(item), " %08x(%g)", word, value);
                    values += item;
                }
                std::fprintf(stderr, "[native]   %s 0x%llx:%s\n", label, static_cast<unsigned long long>(at), values.c_str());
                return true;
            };
            for (std::size_t i = 0; i + 1 < user.size(); ++i) {
                const auto pointer = user[i] | (static_cast<std::uint64_t>(user[i + 1] & 0xffffu) << 32u);
                if (!dumpFloats("pointer", pointer, 128)) continue;
                for (int k = 0; k + 3 < 128; k += 4) {
                    std::array<std::uint32_t, 4> v{};
                    std::memcpy(v.data(), reinterpret_cast<const void*>(pointer + k * 4), sizeof(v));
                    const auto base = v[0] | (static_cast<std::uint64_t>(v[1] & 0xffffu) << 32u);
                    char label[48];
                    std::snprintf(label, sizeof(label), "V# at +0x%x", k * 4);
                    if (v[2] != 0 && (v[3] & 0xfffu) != 0) dumpFloats(label, base, static_cast<int>(std::min<std::uint32_t>(128u, std::max<std::uint32_t>(1u, ((v[1] >> 16u) & 0x3fffu) * v[2] / 4u))));
                }
            }
        }
    }
    // DCB indirect packets carry an offset from SET_BASE; ACB packets
    // carry address low/high. In both layouts the initiator is last.
    const auto initiator = packet.back();
    // DISPATCH_INDIRECT with thread dimensions: the arguments are thread counts written by earlier
    // GPU work. They are read once that work has completed, and the dispatch then runs as a direct
    // thread-count dispatch (partial last groups included). Call-site preparation cannot read them
    // yet, so that dispatch is resolved when the executor reaches it.
    std::array<std::uint32_t, 3> indirectThreads{};
    const bool threadArguments = indirect && (initiator & 0x20u) != 0;
    if (threadArguments) {
        if (prepareOnly != nullptr) return false;
        device->Flush();
        GuestMemory::Read(Pm4::DispatchArgumentAddress(packet, queue), std::as_writable_bytes(std::span(indirectThreads)), 4);
        indirect = false;
    }
    const std::uint64_t indirectArguments = indirect ? Pm4::DispatchArgumentAddress(packet, queue) : 0u;
    const auto wave = (initiator & 0x8000u) != 0 ? 32u : 64u;
    const auto rsrc1 = readRegister(queue.shader, 0x212);
    const auto rsrc2 = readRegister(queue.shader, 0x213);
    std::array<std::uint32_t, 3> groups = indirect ? std::array<std::uint32_t, 3>{1u, 1u, 1u} : std::array<std::uint32_t, 3>{packet[1], packet[2], packet[3]};
    std::array<std::uint32_t, 3> partial{};
    const std::array<std::uint32_t, 3> numThreadRegisters{readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)};
    if (!indirect && (initiator & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads = threadArguments ? indirectThreads : std::array<std::uint32_t, 3>{packet[1], packet[2], packet[3]};
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
    std::shared_ptr<const ShaderRecompiler::IrResourcePlan> flattened;
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
            if (trace && Native::Logs()) std::fprintf(stderr, "[native] translating compute program 0x%llx (wave%u)\n", static_cast<unsigned long long>(address), wave);
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
            std::thread([program, info, userData, floatMode, target, address, wave, key, trace, decodeFailure, job = std::make_shared<Native::CompileJob>()] {
                Native::BackgroundSlot slot(true);
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
                        if (!TranslationCached(request)) job->Compiling();
                        made.result = std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
                        made.flattened = FlattenedPlan(request, *made.result);
                        made.program = program;
                        made.classic = ClassicRequest(request);
                        made.descriptors = ResolveDescriptors(*made.classic);
                    } catch (const std::exception& error) {
                        std::string text = error.what();
                        if (const auto newline = text.find('\n'); newline != std::string::npos) text.resize(newline);
                        char program[48];
                        std::snprintf(program, sizeof(program), "program 0x%llx: ", static_cast<unsigned long long>(address));
                        made.failure = "translation failed: " + std::string(program) + text.substr(0, 1500);
                    }
                }
                if (trace && Native::Logs()) std::fprintf(stderr, "[native] translated 0x%llx: %s\n", static_cast<unsigned long long>(address), made.failure.empty() ? "ok" : made.failure.c_str());
                if (made.result != nullptr) {
                    static const char* directory = std::getenv("APS5_NATIVE_SPV_DIR");
                    if (directory != nullptr) {
                        char name[512];
                        std::snprintf(name, sizeof(name), "%s/compute_%llx.spv", directory, static_cast<unsigned long long>(address));
                        if (std::FILE* file = std::fopen(name, "wb")) {
                            std::fwrite(made.result->spirv.data(), 4, made.result->spirv.size(), file);
                            std::fclose(file);
                        }
                    }
                }
                auto& compute = Compute();
                std::lock_guard done(compute.mutex);
                compute.programs[key] = std::move(made);
                compute.ready.notify_all();
            }).detach();
            lock.lock();
            found = compute.programs.find(key);
        }
        {
            const Native::CompileWait waiting;
            compute.ready.wait(lock, [&] { return compute.programs.at(key).failure != "compute program still translating"; });
        }
        found = compute.programs.find(key);
        if (!found->second.failure.empty()) {
            const auto failure = found->second.failure;
            lock.unlock();
            CountFallback(failure);
            return false;
        }
        result = found->second.result;
        flattened = found->second.flattened;
        if (found->second.descriptors != nullptr && found->second.classic.has_value() && !SlotDescriptors()) {
            auto classic = *found->second.classic;
            const auto descriptors = found->second.descriptors;
            lock.unlock();
            classic.context.userData = userData;
            RegisterSampledDescriptors(classic, false, *descriptors, 0, device->WritesGuestMemory(*result));
        }
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
    std::vector<std::uint32_t> slotWords;
    auto flat = FlattenedSrt(flattened.get(), shader, userData, address, slotWords);
    auto shaderData = ShaderDataFor(shader, 0, userData, 0, flat);
    if (shader.dispatchThreadLimitDword != ~0u && shader.dispatchThreadLimitDword + 3u <= shaderData.size()) std::copy(partial.begin(), partial.end(), shaderData.begin() + shader.dispatchThreadLimitDword);
    if (prepareOnly != nullptr) {
        *prepareOnly = {result, key, groups, std::move(shaderData), address, indirectArguments, true, std::move(flat), std::move(slotWords)};
        return true;
    }
    std::string failure;
    if (!device->Dispatch({&shader, key, groups, shaderData, address, indirectArguments, flat, slotWords}, failure)) {
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
    if (Native::Logs()) std::fprintf(stderr, "[native] executor: %llu submissions, %llu packets, %llu dispatches, %llu draws, %llu flips, %llu waits\n", static_cast<unsigned long long>(counts.submissions.load()), static_cast<unsigned long long>(counts.packets.load()), static_cast<unsigned long long>(counts.dispatches.load()), static_cast<unsigned long long>(counts.draws.load()), static_cast<unsigned long long>(counts.flips.load()), static_cast<unsigned long long>(counts.waits.load()));
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

struct WaitStats {
    std::mutex mutex;
    std::map<std::uint64_t, std::string> writers;
    struct Entry {
        std::uint64_t count = 0;
        double ms = 0.0;
        std::uint32_t queue = 0;
    };
    std::map<std::uint64_t, Entry> waits;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

WaitStats& Waits() {
    static WaitStats stats;
    return stats;
}

void NoteLabelWriter(std::uint64_t address, std::uint32_t queue, const char* how) {
    if (!Native::Logs() || address == 0) return;
    auto& stats = Waits();
    std::lock_guard lock(stats.mutex);
    char text[48];
    std::snprintf(text, sizeof(text), "%s q%u", how, queue);
    stats.writers[address] = text;
}

void NoteBlockedWait(std::uint64_t address, std::uint32_t queue, double ms) {
    if (!Native::Logs()) return;
    auto& stats = Waits();
    std::lock_guard lock(stats.mutex);
    auto& entry = stats.waits[address];
    ++entry.count;
    entry.ms += ms;
    entry.queue = queue;
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.reported < std::chrono::seconds(5)) return;
    stats.reported = now;
    std::vector<std::pair<std::uint64_t, WaitStats::Entry>> sorted(stats.waits.begin(), stats.waits.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    double total = 0.0;
    for (const auto& [where, waited] : sorted) total += waited.ms;
    std::fprintf(stderr, "[native] blocked memory waits in 5 s: %.0f ms total\n", total);
    for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 10); ++i) {
        const auto found = stats.writers.find(sorted[i].first);
        std::fprintf(stderr, "[native]   0x%llx q%u: %llu waits, %.1f ms, written by %s\n", static_cast<unsigned long long>(sorted[i].first), sorted[i].second.queue, static_cast<unsigned long long>(sorted[i].second.count), sorted[i].second.ms, found != stats.writers.end() ? found->second.c_str() : "no packet seen (CPU or shader)");
    }
    stats.waits.clear();
}

struct QueueBusy {
    std::mutex mutex;
    std::map<std::uint32_t, std::pair<double, double>> queues;
    std::map<std::uint32_t, std::uint64_t> submissions;
    std::map<std::uint32_t, double> offCpu;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

QueueBusy& Busy() {
    static QueueBusy busy;
    return busy;
}

void NoteQueueTime(std::uint32_t queue, double busyMs, double blockedMs, double offCpuMs) {
    if (!Native::Logs()) return;
    auto& busy = Busy();
    std::lock_guard lock(busy.mutex);
    auto& entry = busy.queues[queue];
    entry.first += busyMs;
    entry.second += blockedMs;
    busy.offCpu[queue] += offCpuMs;
    ++busy.submissions[queue];
    const auto now = std::chrono::steady_clock::now();
    const auto window = std::chrono::duration<double, std::milli>(now - busy.reported).count();
    if (window < 5000.0) return;
    busy.reported = now;
    std::string line;
    for (const auto& [id, times] : busy.queues) {
        char item[128];
        std::snprintf(item, sizeof(item), " q%u: %.0f%% executing (%.0f%% off-CPU, %.0f%% blocked on other queues), %llu submissions;", id, 100.0 * times.first / window, times.first > 0 ? 100.0 * busy.offCpu[id] / times.first : 0.0, times.first > 0 ? 100.0 * times.second / times.first : 0.0, static_cast<unsigned long long>(busy.submissions[id]));
        line += item;
    }
    std::fprintf(stderr, "[native] queue workers over %.1f s:%s\n", window / 1000.0, line.c_str());
    busy.queues.clear();
    busy.submissions.clear();
    busy.offCpu.clear();
}

thread_local double blockedThisSubmission = 0.0;

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
        if (trace && Native::Logs()) std::fprintf(stderr, "[native] flushing native work\n");
        device->Flush();
        if (trace && Native::Logs()) std::fprintf(stderr, "[native] native work done\n");
    }
}

void SubmitNative() {
    if (auto* device = Native::NativeDevice::Get(); device != nullptr) device->Submit();
}

void ReportUnexecuted(std::uint32_t header, const char* reason) {
    static std::mutex mutex;
    static std::set<std::uint32_t> reported;
    const auto opcode = (header >> 8u) & 0xffu;
    std::lock_guard lock(mutex);
    static_cast<void>(reported.insert(opcode));
    Native::ReportGraphicsIssue("command packet not executed", Pm4::Name(header) + ": " + reason);
}

}

namespace {

bool LookaheadRegisterPacket(std::uint32_t opcode, std::span<const std::uint32_t> packet) {
    switch (opcode) {
        case 0x10: case 0x11: case 0x12: case 0x13: case 0x26: case 0x2a: case 0x2f:
        case 0x69: case 0x76: case 0x79: case 0x7a: case 0x81: case 0x91:
            return true;
        case 0x37: return ((packet[1] >> 8u) & 0xfu) == 0;
        default: return false;
    }
}

void RunDrawLookahead() {
    auto& ahead = drawLookahead;
    if (ahead.done || ahead.submission == nullptr || ahead.queue == nullptr) return;
    ahead.done = true;
    static const std::size_t limit = [] {
        const char* text = std::getenv("APS5_NATIVE_LOOKAHEAD");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{512};
    }();
    if (limit == 0) return;
    const auto& commands = ahead.submission->commands;
    QueueState queue = *ahead.queue;
    const bool previous = prefetchingDraws;
    prefetchingDraws = true;
    std::size_t draws = 0;
    for (std::size_t cursor = ahead.cursor; cursor < commands.size() && draws < limit;) {
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        const auto here = cursor;
        cursor += count;
        if (here == ahead.cursor) continue;
        try {
            if (Pm4::IndirectRegisterOpcode(opcode)) Pm4::ExecuteIndirectRegisters(packet, ahead.submission->registerLists.at(here), queue);
            else if (opcode == 0x40 && Pm4::CopyDataToRegister(packet)) Pm4::Execute(packet, queue);
            else if (LookaheadRegisterPacket(opcode, packet)) Pm4::Execute(packet, queue);
            else if (Pm4::DrawOpcode(opcode) && !Pm4::IndirectDrawOpcode(opcode)) {
                ++draws;
                static_cast<void>(DrawNatively(queue, packet, true));
            }
        } catch (const std::exception&) {
        }
    }
    prefetchingDraws = previous;
}

}

void Driver::executeNative(const Submission& submission) {
    struct BusyTimer {
        std::uint32_t queue;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        double blockedBefore = blockedThisSubmission;
        double cpuStart = ThreadCpuMs();
        static double ThreadCpuMs() {
            timespec now{};
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
            return static_cast<double>(now.tv_sec) * 1000.0 + static_cast<double>(now.tv_nsec) / 1e6;
        }
        ~BusyTimer() {
            const auto wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            NoteQueueTime(queue, wall, blockedThisSubmission - blockedBefore, wall - (ThreadCpuMs() - cpuStart));
        }
    } busyTimer{submission.queue};
    Native::TimelineEvent("exec begin", submission.queue, submission.serial);
    struct ExecEnd {
        std::uint32_t queue;
        std::uint64_t serial;
        ~ExecEnd() { Native::TimelineEvent("exec end", queue, serial); }
    } execEnd{submission.queue, submission.serial};
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
    drawLookahead = {&submission, &queue, 0, false};
    if (auto* native = Native::NativeDevice::Get(); native != nullptr) native->NoteSubmission();
    struct EndLookahead {
        ~EndLookahead() { drawLookahead = {}; }
    } endLookahead;

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
        if (opcode == 0x10 && packet.size() >= 3 && packet[1] == Native::FrontEndDrawMarker) frontEndDraw = packet[2];
        else if (opcode != 0x27 && opcode != 0x2d && opcode != 0x25 && opcode != 0x15 && opcode != 0x16) frontEndDraw = 0;
        activity.phase = "packet";
        if (submissionLog) opcodes[opcode]++;
        const bool waitByOrder = (opcode == 0x3c || opcode == 0x93) && !Pm4::WaitSatisfied(packet) && SatisfiedByPendingRelease(packet);
        bool onGpu = false;
        if ((opcode == 0x37 || opcode == 0x50) && !(Pm4::Predicated(header) && queue.predication.operation != 0)) {
            if (auto* device = Native::NativeDevice::Get(); device != nullptr) {
                try {
                    onGpu = device->TransferOnGpu(packet);
                    if (onGpu) NoteQueueWork(submission.queue);
                } catch (const std::exception& error) {
                    ReportUnexecuted(header, error.what());
                }
            }
            if (onGpu && opcode == 0x37 && packet.size() >= 5 && ((packet[1] >> 8u) & 0xfu) != 0u) {
                const auto destination = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
                const auto value = packet.size() >= 6 ? (packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u)) : static_cast<std::uint64_t>(packet[4]);
                NoteLabelWriter(destination, submission.queue, "WRITE_DATA");
                std::uint64_t serial = 0;
                {
                    auto& releases = Releases();
                    std::lock_guard lock(releases.mutex);
                    serial = ++releases.serial;
                    releases.entries.push_back({destination, value, serial});
                }
                Native::NativeDevice::Get()->AfterBatch([serial] {
                    auto& releases = Releases();
                    std::lock_guard lock(releases.mutex);
                    std::erase_if(releases.entries, [&](const PendingRelease& entry) { return entry.serial == serial; });
                });
            }
        }
        if (NeedsNativeResults(opcode) && !waitByOrder && !onGpu) {
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
                        if (Native::Logs()) std::fprintf(stderr, "[native] batch cuts in 5 s:%s\n", line.c_str());
                        cuts.clear();
                    }
                }
            }
            if (opcode == 0x3c || opcode == 0x93) SubmitNative();
            else FlushNative();
            activity.phase = "packet";
        }
        if (Pm4::Predicated(header) && queue.predication.operation != 0 && !Pm4::PredicationPasses(queue)) {
            cursor = opcode == 0x3f ? submission.conditionalEnds.at(cursor) : nextCursor;
            continue;
        }

        if (header == RenderingWaitPacketHeader) {
            if (submissionLog) std::fprintf(stderr, "[native] submission %llu: waiting until safe for rendering\n", static_cast<unsigned long long>(submission.serial));
            const auto safeAt = std::chrono::steady_clock::now();
            Native::TimelineEvent("safe-for-rendering wait", submission.queue);
            submission.renderingWaits.at(cursor)->Wait();
            const auto waitedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - safeAt).count();
            blockedThisSubmission += waitedMs;
            NoteBlockedWait(0, submission.queue, waitedMs);
            if (submissionLog) std::fprintf(stderr, "[native] submission %llu: safe for rendering\n", static_cast<unsigned long long>(submission.serial));
        } else if (header == FlipPacketHeader && (++Counts().flips, true)) {
            Native::TimelineEvent("flip packet", submission.queue);
            // The frame's native work completes before the presenter reads the display buffer.
            activity.phase = "flushing native work";
            SubmitNative();
            activity.phase = "packet";
            ++flipsCounted;
            if (auto* native = Native::NativeDevice::Get(); native != nullptr) native->NoteFlip();
            auto frame = std::make_shared<FrameTiming>(++frameSerial);
            const auto now = FrameTiming::Clock::now();
            frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, now, now);
            flipHold.Release();
            submission.flips.at(cursor)->GpuReady(frame);
        } else if (opcode == 0x3f) {
            // Indirect buffer bodies are inlined into the submission when it is copied.
        } else if (Pm4::DrawOpcode(opcode) && (++Counts().draws, activity.phase = "native draw", true)) {
            drawLookahead.cursor = cursor;
            if (!DrawNatively(queue, packet)) Native::CountSkippedDraw();
            NoteQueueWork(submission.queue);
        } else if (opcode == 0x15 && (++Counts().dispatches, activity.phase = "native dispatch", true)) {
            if (!DispatchNatively(queue, packet)) Native::CountSkippedDispatch();
            NoteQueueWork(submission.queue);
        } else if (opcode == 0x16 && (++Counts().dispatches, activity.phase = "native dispatch", true)) {
            if (!DispatchNatively(queue, packet, true)) Native::CountSkippedDispatch();
            NoteQueueWork(submission.queue);
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
                const auto blockedAt = std::chrono::steady_clock::now();
                Native::TimelineEvent("wait blocked", submission.queue, awaited);
                WaitWithReport(submission, cursor, packet);
                Native::TimelineEvent("wait done", submission.queue, awaited);
                const auto blockedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - blockedAt).count();
                blockedThisSubmission += blockedMs;
                NoteBlockedWait(awaited, submission.queue, blockedMs);
            }
        } else if (opcode == 0x22) {
            if (Pm4::ReadCondition(packet) == 0) nextCursor = submission.conditionalEnds.at(cursor);
        } else if (IgnoredOpcode(opcode)) {
            if (auto* device = Native::NativeDevice::Get(); device != nullptr) device->RequireBarrier();
        } else {
            if (opcode == 0x49) {
                if (auto* native = Native::NativeDevice::Get(); native != nullptr) native->RequireBarrier();
                // End of pipe: the label write and interrupt happen once the native work recorded
                // so far has completed, without ending the batch.
                std::vector<std::uint32_t> copy(packet.begin(), packet.end());
                auto* device = Native::NativeDevice::Get();
                const auto queueId = submission.queue;
                // A release with immediate data satisfies waits on it within the batch.
                const auto dataSelect = copy[2] >> 29u;
                const auto destination = copy[3] | (static_cast<std::uint64_t>(copy[4]) << 32u);
                NoteLabelWriter(destination, queueId, "RELEASE_MEM");
                Native::NotePoolUse("label", destination);
                std::uint64_t serial = 0;
                if (device != nullptr && device->Pending() && (dataSelect == 1 || dataSelect == 2) && destination != 0) {
                    auto& releases = Releases();
                    std::lock_guard lock(releases.mutex);
                    serial = ++releases.serial;
                    const auto value = dataSelect == 1 ? static_cast<std::uint64_t>(copy[5]) : (copy[5] | (static_cast<std::uint64_t>(copy[6]) << 32u));
                    releases.entries.push_back({destination, value, serial});
                }
                auto release = [copy = std::move(copy), &queue, queueId, header, serial, destination] {
                    Native::TimelineEvent("label write", queueId, destination);
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
                if (device != nullptr) device->AfterBatch(std::move(release), QueueWorkSerial(queueId).load(std::memory_order_acquire));
                else release();
            } else if (onGpu) {
            } else {
                if (opcode == 0x37 && packet.size() >= 4) NoteLabelWriter(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), submission.queue, "WRITE_DATA");
                if (opcode == 0x50 && packet.size() >= 6) NoteLabelWriter(packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u), submission.queue, "DMA_DATA");
                try {
                    if (opcode == 0x50 && Native::NativeDevice::Get() != nullptr && Native::NativeDevice::Get()->DmaGds(packet)) NoteQueueWork(submission.queue);
                    else if (Pm4::IndirectRegisterOpcode(opcode)) Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(cursor), queue);
                    else Pm4::Execute(packet, queue);
                } catch (const std::runtime_error& error) {
                    ReportUnexecuted(header, error.what());
                }
            }
        }
        cursor = nextCursor;
    }
    SubmitNative();
    // A rewindable command buffer: the game writes the rest later and marks it valid; it runs
    // as the continuation of this submission.
    if (submission.rewindTail != nullptr) executeRewindTail(submission);
}

std::shared_ptr<void> PrepareFrontEndDraw(QueueState& queue, std::span<const std::uint32_t> packet) {
    auto prepared = std::make_shared<PreparedDraw>();
    try {
        if (!DrawNatively(queue, packet, false, prepared.get()) || !prepared->ready) return nullptr;
        prepared->packet.assign(packet.begin(), packet.end());
    } catch (const std::exception& error) {
        Native::ReportGraphicsIssue("draw not prepared at the call site", error.what());
        return nullptr;
    }
    return prepared;
}

std::shared_ptr<void> PrepareFrontEndDispatch(QueueState& queue, std::span<const std::uint32_t> packet, bool indirect) {
    auto prepared = std::make_shared<PreparedDispatch>();
    try {
        if (!DispatchNatively(queue, packet, indirect, prepared.get()) || !prepared->ready) return nullptr;
    } catch (const std::exception& error) {
        Native::ReportGraphicsIssue("dispatch not prepared at the call site", error.what());
        return nullptr;
    }
    return prepared;
}

}
