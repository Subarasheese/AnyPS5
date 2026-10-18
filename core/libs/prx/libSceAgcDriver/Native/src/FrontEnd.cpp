#include "prx/libSceAgcDriver/Native/include/FrontEnd.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <span>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace AgcDriver::DriverDetail {
std::shared_ptr<void> PrepareFrontEndDraw(QueueState& queue, std::span<const std::uint32_t> packet);
std::shared_ptr<void> PrepareFrontEndDispatch(QueueState& queue, std::span<const std::uint32_t> packet, bool indirect);
}

namespace AgcDriver::Native {

bool FrontEndActive();

namespace {

struct Segment {
    const std::uint32_t* bottom = nullptr;
    QueueState state;
};

struct Signature {
    std::vector<std::uint64_t> values;
    std::vector<const char*> labels;
    void add(const char* label, std::uint64_t value) {
        labels.push_back(label);
        values.push_back(value);
    }
};

Signature DrawSignature(const QueueState& queue, bool mergedWords = true) {
    Signature signature;
    try {
        const auto state = Graphics::DecodeState(queue, false, true);
        signature.add("shader path", static_cast<std::uint64_t>(state.stages.path));
        signature.add("depth target", state.depth ? state.depth->address : 0u);
        signature.add("depth format", state.depth ? static_cast<std::uint64_t>(state.depth->format) : 0u);
        signature.add("depth test", static_cast<std::uint64_t>(state.depthTest) | (static_cast<std::uint64_t>(state.depthWrite) << 1u) | (static_cast<std::uint64_t>(state.depthCompare) << 2u) | (static_cast<std::uint64_t>(state.stencilTest) << 8u));
        signature.add("color target count", state.colors.size());
        for (const auto& color : state.colors) {
            signature.add("color target", color.address);
            signature.add("color format", static_cast<std::uint64_t>(color.format));
        }
        for (const auto& blend : state.blends) {
            signature.add("blend", static_cast<std::uint64_t>(blend.blendEnable) | (static_cast<std::uint64_t>(blend.srcColorBlendFactor) << 1u) | (static_cast<std::uint64_t>(blend.dstColorBlendFactor) << 9u) | (static_cast<std::uint64_t>(blend.colorBlendOp) << 17u) | (static_cast<std::uint64_t>(blend.srcAlphaBlendFactor) << 25u) | (static_cast<std::uint64_t>(blend.dstAlphaBlendFactor) << 33u) | (static_cast<std::uint64_t>(blend.alphaBlendOp) << 41u) | (static_cast<std::uint64_t>(blend.colorWriteMask) << 49u));
        }
        signature.add("render extent", (static_cast<std::uint64_t>(state.renderExtent.width) << 32u) | state.renderExtent.height);
        signature.add("topology", static_cast<std::uint64_t>(state.topology) | (static_cast<std::uint64_t>(state.rectList) << 8u));
        std::uint64_t viewport = 0;
        std::memcpy(&viewport, &state.viewport.width, sizeof(viewport));
        signature.add("viewport", viewport);
        signature.add("scissor", (static_cast<std::uint64_t>(state.scissor.extent.width) << 32u) | state.scissor.extent.height);
    } catch (const std::exception&) {
        signature.add("state undecodable", 1);
    }
    const auto program = [&](const char* name, const char* user, std::uint32_t base, std::uint32_t rsrc2Register, std::uint32_t userBase) {
        const auto low = queue.shader.find(base), high = queue.shader.find(base + 1), resources = queue.shader.find(rsrc2Register);
        if (low == queue.shader.end() || high == queue.shader.end() || resources == queue.shader.end()) {
            signature.add(name, 0);
            return;
        }
        signature.add(name, (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u));
        const auto userCount = ((resources->second >> 1u) & 0x1fu) | (((resources->second >> 27u) & 1u) << 5u);
        for (std::uint32_t i = 0; i < userCount; ++i) signature.add(user, DriverDetail::readUserData(queue.shader, userBase + i));
    };
    program("vertex program", "vertex user data", 0xc8, 0x8b, 0x8c);
    program("pixel program", "pixel user data", 0x008, 0x00b, 0x00c);
    if (mergedWords) signature.add("merged stage words", (static_cast<std::uint64_t>(DriverDetail::readUserData(queue.shader, 0x82)) << 32u) | DriverDetail::readUserData(queue.shader, 0x83));
    signature.add("index buffer", queue.indexBase);
    signature.add("index type and instances", queue.indexType | (static_cast<std::uint64_t>(queue.instanceCount) << 8u));
    return signature;
}

struct FrontEnd {
    std::mutex mutex;
    std::mutex segmentsMutex;
    std::unordered_map<const void*, Segment> segments;
    std::mutex preparedMutex;
    QueueState global;
    std::unordered_map<std::uint32_t, Signature> snapshots;
    std::unordered_map<std::uint32_t, std::shared_ptr<void>> prepared;
    std::unordered_map<std::uint32_t, bool> mergedWritten;
    std::atomic<std::uint32_t> nextId{0};
    std::uint64_t compared = 0, exact = 0, unknown = 0;
    std::map<std::string, std::uint64_t> differences;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

FrontEnd& State() {
    static FrontEnd state;
    return state;
}

std::uint32_t SampleEvery() {
    static const std::uint32_t every = [] {
        const char* text = std::getenv("APS5_FRONTEND_VALIDATE");
        return text != nullptr ? std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(std::atoi(text))) : 0u;
    }();
    return every;
}


Registers& RegistersFor(QueueState& state, std::uint32_t opcode) {
    if (opcode == 0x69u || opcode == 0x9fu) return state.context;
    if (opcode == 0x76u || opcode == 0x63u) return state.shader;
    return state.userConfig;
}


}

bool FrontEndValidating() {
    return SampleEvery() != 0;
}

bool FrontEndDraws() {
    static const bool enabled = std::getenv("APS5_FRONTEND_DRAWS") != nullptr;
    return enabled;
}

bool FrontEndActive() {
    return FrontEndValidating() || FrontEndDraws();
}

std::shared_ptr<void> TakeFrontEndDraw(std::uint32_t id) {
    auto& frontEnd = State();
    std::lock_guard lock(frontEnd.preparedMutex);
    const auto found = frontEnd.prepared.find(id);
    if (found == frontEnd.prepared.end()) return nullptr;
    auto prepared = std::move(found->second);
    frontEnd.prepared.erase(found);
    return prepared;
}

void CompareFrontEndDraw(std::uint32_t id, const QueueState& queue) {
    auto& frontEnd = State();
    std::lock_guard lock(frontEnd.mutex);
    const auto found = frontEnd.snapshots.find(id);
    if (found == frontEnd.snapshots.end()) {
        ++frontEnd.unknown;
    } else {
        ++frontEnd.compared;
        const auto merged = frontEnd.mergedWritten[id];
        frontEnd.mergedWritten.erase(id);
        const auto executed = DrawSignature(queue, merged);
        const auto& recorded = found->second;
        bool identical = executed.values.size() == recorded.values.size();
        for (std::size_t i = 0; i < std::max(executed.values.size(), recorded.values.size()); ++i) {
            if (i < executed.values.size() && i < recorded.values.size() && executed.values[i] == recorded.values[i] && executed.labels[i] == recorded.labels[i]) continue;
            identical = false;
            const std::string label = i < executed.labels.size() ? executed.labels[i] : recorded.labels[i];
            if (frontEnd.differences[label]++ < 4) std::fprintf(stderr, "[frontend] %s: executor 0x%llx, front-end 0x%llx\n", label.c_str(), static_cast<unsigned long long>(i < executed.values.size() ? executed.values[i] : 0), static_cast<unsigned long long>(i < recorded.values.size() ? recorded.values[i] : 0));
            break;
        }
        if (identical) ++frontEnd.exact;
        frontEnd.snapshots.erase(found);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - frontEnd.reported < std::chrono::seconds(5)) return;
    frontEnd.reported = now;
    std::string line;
    for (const auto& [label, count] : frontEnd.differences) line += " " + label + " x" + std::to_string(count) + ";";
    std::fprintf(stderr, "[frontend] %llu draws compared, %llu decode identically, %llu without a front-end record; first differing field:%s\n", static_cast<unsigned long long>(frontEnd.compared), static_cast<unsigned long long>(frontEnd.exact), static_cast<unsigned long long>(frontEnd.unknown), line.c_str());
    frontEnd.compared = frontEnd.exact = frontEnd.unknown = 0;
    frontEnd.differences.clear();
}

}

using namespace AgcDriver;
using namespace AgcDriver::Native;

namespace {

Segment& LocalSegment(const void* buffer, const std::uint32_t* bottom) {
    struct Cached {
        const void* buffer = nullptr;
        Segment* segment = nullptr;
    };
    thread_local std::array<Cached, 4> cache{};
    thread_local std::size_t next = 0;
    for (auto& entry : cache) {
        if (entry.buffer == buffer && entry.segment != nullptr) {
            entry.segment->bottom = bottom;
            return *entry.segment;
        }
    }
    auto& frontEnd = State();
    std::lock_guard lock(frontEnd.segmentsMutex);
    auto& segment = frontEnd.segments[buffer];
    segment.bottom = bottom;
    cache[next++ % cache.size()] = {buffer, &segment};
    return segment;
}

void NoteGlobal(std::uint32_t opcode, std::uint32_t offset, std::uint32_t value) {
    if (!FrontEndValidating()) return;
    auto& frontEnd = State();
    std::lock_guard lock(frontEnd.mutex);
    RegistersFor(frontEnd.global, opcode).insert_or_assign(offset, value);
}

}

extern "C" void AgcFrontEndRegisters_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t opcode, const volatile void* registers, std::uint32_t count) {
    if (!FrontEndActive() || registers == nullptr) return;
    auto& segment = LocalSegment(buffer, bottom);
    auto& target = RegistersFor(segment.state, opcode);
    const auto* words = static_cast<const volatile std::uint32_t*>(registers);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto offset = words[i * 2u];
        if (offset == 0xffffffffu) continue;
        target.insert_or_assign(offset & ~0x70000000u, words[i * 2u + 1u]);
        NoteGlobal(opcode, offset & ~0x70000000u, words[i * 2u + 1u]);
    }
}

extern "C" void AgcFrontEndRegisterRange_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t opcode, std::uint32_t offset, const std::uint32_t* values, std::uint32_t count) {
    if (!FrontEndActive() || values == nullptr) return;
    auto& target = RegistersFor(LocalSegment(buffer, bottom).state, opcode);
    for (std::uint32_t i = 0; i < count; ++i) {
        target.insert_or_assign(offset + i, values[i]);
        NoteGlobal(opcode, offset + i, values[i]);
    }
}

extern "C" void AgcFrontEndIndexState_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t field, std::uint64_t value) {
    if (!FrontEndActive()) return;
    auto& state = LocalSegment(buffer, bottom).state;
    switch (static_cast<FrontEndIndexField>(field)) {
        case FrontEndIndexField::Buffer: state.indexBase = value; break;
        case FrontEndIndexField::Size: state.indexType = static_cast<std::uint32_t>(value); break;
        case FrontEndIndexField::Count: state.indexBufferSize = static_cast<std::uint32_t>(value); break;
        case FrontEndIndexField::Instances: state.instanceCount = static_cast<std::uint32_t>(value); break;
        case FrontEndIndexField::DrawIndirectBase: state.drawIndirectBase = value; break;
        case FrontEndIndexField::DispatchIndirectBase: state.dispatchIndirectBase = value; break;
    }
}

extern "C" void AgcFrontEndReset_nid_postfix(const void* buffer, const std::uint32_t* bottom) {
    if (!FrontEndActive()) return;
    LocalSegment(buffer, bottom).state.ClearContext();
}

extern "C" std::uint32_t AgcFrontEndDraw_nid_postfix(const void* buffer, const std::uint32_t* bottom) {
    if (!FrontEndActive()) return 0;
    auto& frontEnd = State();
    const auto id = frontEnd.nextId.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (!FrontEndValidating() || id % SampleEvery() != 0) return id;
    const auto& own = LocalSegment(buffer, bottom).state;
    std::lock_guard lock(frontEnd.mutex);
    if (frontEnd.snapshots.size() > 4096) return id;
    QueueState view = own;
    for (const auto& [offset, value] : frontEnd.global.shader) {
        if (!own.shader.contains(offset)) view.shader.insert_or_assign(offset, value);
    }
    frontEnd.mergedWritten[id] = own.shader.contains(0x82u);
    frontEnd.snapshots[id] = DrawSignature(view, own.shader.contains(0x82u));
    return id;
}

extern "C" void AgcFrontEndPrepareDraw_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t id, const std::uint32_t* packet, std::uint32_t words) {
    if (!FrontEndDraws() || id == 0 || packet == nullptr) return;
    auto prepared = DriverDetail::PrepareFrontEndDraw(LocalSegment(buffer, bottom).state, std::span<const std::uint32_t>(packet, words));
    if (prepared == nullptr) return;
    auto& frontEnd = State();
    std::lock_guard lock(frontEnd.preparedMutex);
    if (frontEnd.prepared.size() > 16384) frontEnd.prepared.clear();
    frontEnd.prepared[id] = std::move(prepared);
}

extern "C" void AgcFrontEndPrepareDispatch_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t id, const std::uint32_t* packet, std::uint32_t words, bool indirect) {
    if (!FrontEndDraws() || id == 0 || packet == nullptr) return;
    auto prepared = DriverDetail::PrepareFrontEndDispatch(LocalSegment(buffer, bottom).state, std::span<const std::uint32_t>(packet, words), indirect);
    if (prepared == nullptr) return;
    auto& frontEnd = State();
    std::lock_guard lock(frontEnd.preparedMutex);
    if (frontEnd.prepared.size() > 16384) frontEnd.prepared.clear();
    frontEnd.prepared[id] = std::move(prepared);
}
