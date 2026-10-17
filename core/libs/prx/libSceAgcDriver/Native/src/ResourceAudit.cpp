// GT7's texture and render-target initializers (the statically linked AGC core functions 0x27d120
// and 0x1638690): in native GPU mode every descriptor they make goes to the native image registry
// (NativeImages.hpp). The resource-creation audit (APS5_NATIVE_AUDIT=1) also reports which of the
// textures GT7's shaders sample were made by them (emulated path).

#include "prx/libSceAgcDriver/Native/include/GameHooks.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeImages.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace AgcDriver::Native {

namespace {

struct Produced {
    std::uint32_t width = 0, height = 0, format = 0, kind = 0;
};

struct Audit {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Produced> textures;
    std::unordered_map<std::uint64_t, Produced> targets;
    std::unordered_set<std::uint64_t> sampled;
    std::unordered_map<std::uint64_t, std::array<std::uint32_t, 8>> missing;
    std::uint64_t textureCalls = 0, targetCalls = 0;
};

Audit& State() {
    static Audit audit;
    return audit;
}

using TextureInit = std::uint64_t (*)(std::uint32_t* out, const std::uint64_t* spec);
using TargetInit = std::uint64_t (*)(std::uint64_t* out, const std::uint64_t* spec);
TextureInit originalTextureInit = nullptr;
TargetInit originalTargetInit = nullptr;

std::uint64_t TextureBase(const std::uint32_t* words) {
    return ((static_cast<std::uint64_t>(words[0]) | (static_cast<std::uint64_t>(words[1]) << 32u)) & 0xffffffffffull) << 8u;
}

// The T# fields the audit prints: width/height (word 2/3 bits as AnyPS5 decodes them) and format.
Produced Describe(const std::uint32_t* words, std::uint32_t kind) {
    Produced p;
    p.width = ((words[2] >> 30u) | ((words[3] & 0xfffu) << 2u)) + 1u;
    p.height = ((words[3] >> 14u) & 0x3fffu) + 1u;
    p.format = (words[1] >> 20u) & 0x1ffu;
    p.kind = kind;
    return p;
}

std::uint64_t TextureInitHook(std::uint32_t* out, const std::uint64_t* spec) {
    const auto result = originalTextureInit(out, spec);
    if (GpuNative()) {
        if (static_cast<std::uint32_t>(result) == 0 && out != nullptr) NoteTextureDescriptor(std::span<const std::uint32_t, 8>(out, 8));
        return result;
    }
    auto& audit = State();
    std::lock_guard lock(audit.mutex);
    ++audit.textureCalls;
    if (static_cast<std::uint32_t>(result) == 0 && out != nullptr) audit.textures[TextureBase(out)] = Describe(out, 0);
    return result;
}

std::uint64_t TargetInitHook(std::uint64_t* out, const std::uint64_t* spec) {
    const auto result = originalTargetInit(out, spec);
    if (GpuNative()) {
        if (static_cast<std::uint32_t>(result) == 0 && spec != nullptr && spec[0] != 0) NoteRenderTarget(spec[0], 0);
        return result;
    }
    auto& audit = State();
    std::lock_guard lock(audit.mutex);
    ++audit.targetCalls;
    // Spec +0x00: the color surface's address (the initializer aligns and validates it).
    if (static_cast<std::uint32_t>(result) == 0 && spec != nullptr && spec[0] != 0) {
        Produced p;
        p.width = static_cast<std::uint32_t>(spec[4] >> 32u);
        p.height = static_cast<std::uint32_t>(spec[5]);
        p.kind = 1;
        audit.targets[spec[0]] = p;
    }
    return result;
}

void Report() {
    auto& audit = State();
    std::lock_guard lock(audit.mutex);
    std::size_t covered = 0;
    for (const auto base : audit.sampled) covered += audit.textures.count(base) + audit.targets.count(base) != 0;
    std::fprintf(stderr, "[audit] texture inits %llu (%zu surfaces), target inits %llu (%zu surfaces); sampled %zu surfaces, %zu made by an initializer, %zu not\n", static_cast<unsigned long long>(audit.textureCalls), audit.textures.size(), static_cast<unsigned long long>(audit.targetCalls), audit.targets.size(), audit.sampled.size(), covered, audit.missing.size());
    int shown = 0;
    for (const auto& [base, words] : audit.missing) {
        if (++shown > 12) break;
        const auto p = Describe(words.data(), 2);
        std::fprintf(stderr, "[audit]   not initialized: 0x%llx %ux%u format %u words %08x %08x %08x %08x %08x %08x %08x %08x\n", static_cast<unsigned long long>(base), p.width, p.height, p.format, words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]);
    }
}

const bool installed = [] {
    if (!Enabled() || (std::getenv("APS5_NATIVE_AUDIT") == nullptr && !GpuNative())) return false;
    if (!GameCodeMapped(0x27d120, 16u) || !GameCodeMapped(0x1638690, 16u)) return false;
    originalTextureInit = reinterpret_cast<TextureInit>(InstallGameHook("texture init", 0x27d120, reinterpret_cast<void*>(&TextureInitHook)));
    originalTargetInit = reinterpret_cast<TargetInit>(InstallGameHook("render target init", 0x1638690, reinterpret_cast<void*>(&TargetInitHook)));
    if (GpuNative()) return true;
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            Report();
        }
    }).detach();
    return true;
}();

}

bool Audits() {
    return installed;
}

void NoteSampledTexture(const std::uint32_t* words) {
    const auto base = TextureBase(words);
    auto& audit = State();
    std::lock_guard lock(audit.mutex);
    if (!audit.sampled.insert(base).second) return;
    if (audit.textures.count(base) == 0 && audit.targets.count(base) == 0) {
        std::array<std::uint32_t, 8> copy;
        std::memcpy(copy.data(), words, sizeof(copy));
        audit.missing.emplace(base, copy);
    }
}

}
