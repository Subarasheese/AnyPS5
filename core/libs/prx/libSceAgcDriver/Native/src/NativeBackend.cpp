#include <thread>
#include <condition_variable>
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include <errno.h>
#include <dlfcn.h>
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include "prx/libSceAgcDriver/Native/include/GameHooks.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <link.h>
#include <map>
#include <set>


namespace AgcDriver::Native {

namespace {

bool EnvSet(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

struct RecordedShader {
    std::uint32_t type = 0;
    std::shared_ptr<const std::vector<std::uint32_t>> code;
    std::shared_ptr<const std::vector<std::byte>> header;
    std::uint64_t headerAddress = 0;
    std::shared_ptr<std::atomic<std::uint8_t>> positionVectorUse = std::make_shared<std::atomic<std::uint8_t>>(0u);
};

struct Stats {
    std::atomic<std::uint64_t> draws{0}, dispatches{0}, submissions{0}, asyncSubmissions{0}, presents{0};
    std::chrono::steady_clock::time_point windowStart = std::chrono::steady_clock::now();
    double longestFrameMs = 0;
    std::chrono::steady_clock::time_point lastPresent{};
};

Stats& GetStats() {
    static Stats stats;
    return stats;
}

std::mutex& ShaderMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::uintptr_t, RecordedShader>& Shaders() {
    static std::map<std::uintptr_t, RecordedShader> shaders;
    return shaders;
}

void ReportStats(Stats& stats) {
    static const bool report = EnvSet("APS5_NATIVE_STATS");
    const auto now = std::chrono::steady_clock::now();
    if (stats.lastPresent != std::chrono::steady_clock::time_point{}) {
        const double frameMs = std::chrono::duration<double, std::milli>(now - stats.lastPresent).count();
        if (frameMs > stats.longestFrameMs) stats.longestFrameMs = frameMs;
    }
    stats.lastPresent = now;
    ++stats.presents;
    const double windowSeconds = std::chrono::duration<double>(now - stats.windowStart).count();
    if (!report || windowSeconds < 2.0) return;
    std::size_t shaderCount = 0;
    {
        std::lock_guard lock(ShaderMutex());
        shaderCount = Shaders().size();
    }
    std::fprintf(stderr, "[native] %.1f fps (longest frame %.1f ms) | per frame: %.0f draws, %.0f dispatches, %.1f graphics + %.1f async submissions skipped GPU work | %zu shaders recorded\n",
                 stats.presents / windowSeconds, stats.longestFrameMs,
                 stats.draws / static_cast<double>(stats.presents), stats.dispatches / static_cast<double>(stats.presents),
                 stats.submissions / static_cast<double>(stats.presents), stats.asyncSubmissions / static_cast<double>(stats.presents), shaderCount);
    stats.windowStart = now;
    stats.longestFrameMs = 0;
    stats.presents = 0;
    stats.draws = 0;
    stats.dispatches = 0;
    stats.submissions = 0;
    stats.asyncSubmissions = 0;
}

}

namespace {
// GT7's native PC graphics layer (gt7_render_test's libgt7native, GT7_NATIVE=<path to libgt7native.so>): loaded
// into the game process before its threads run; it patches GT7's graphics call sites and renders with its own
// Vulkan renderer.
void InstallPglHooks() {
    const char* path = std::getenv("GT7_NATIVE");
    if (!path || !*path || !std::strstr(program_invocation_short_name, "gt7ps5")) return;
    void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "[native] GT7_NATIVE: %s\n", dlerror()); std::abort(); }
    auto start = reinterpret_cast<int (*)()>(dlsym(library, "gt7native_start"));
    if (!start || !start()) { std::fprintf(stderr, "[native] GT7_NATIVE: gt7native_start failed\n"); std::abort(); }
}
}

bool Enabled() {
    static const bool enabled = !EnvSet("APS5_EMULATED_GPU");
    return enabled;
}

namespace {
// GT7's code is mapped before this library initializes, so the context hooks go in at load time,
// before any game thread runs.
void InstallPglHooks();
const bool hooksInstalled = [] {
    if (Enabled()) InstallGraphicsContextHooks();
    InstallPglHooks();
    return true;
}();
}

namespace {

struct GraphicsIssues {
    std::mutex mutex;
    std::map<std::string, std::uint64_t> counts;
    bool dirty = false;
    std::chrono::steady_clock::time_point written{};
};

GraphicsIssues& Issues() {
    static GraphicsIssues issues;
    return issues;
}

void WriteGraphicsIssues(GraphicsIssues& issues) {
    std::vector<std::pair<std::string, std::uint64_t>> sorted(issues.counts.begin(), issues.counts.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (std::FILE* file = std::fopen("graphics-issues.txt", "w")) {
        for (const auto& [issue, count] : sorted) std::fprintf(file, "%10llu  %s\n", static_cast<unsigned long long>(count), issue.c_str());
        std::fclose(file);
    }
    issues.dirty = false;
    issues.written = std::chrono::steady_clock::now();
}

}

void NotePoolUse(const char* use, std::uint64_t address) {
    static const char* path = std::getenv("APS5_POOL_AUDIT");
    if (path == nullptr || address == 0) return;
    static std::mutex mutex;
    static std::map<std::string, std::set<std::uint64_t>> pages;
    static auto written = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex);
    pages[use].insert(address >> 16u);
    const auto now = std::chrono::steady_clock::now();
    if (now - written < std::chrono::seconds(10)) return;
    written = now;
    if (std::FILE* file = std::fopen(path, "w")) {
        for (const auto& [name, set] : pages) {
            for (const auto page : set) std::fprintf(file, "%s %llx\n", name.c_str(), static_cast<unsigned long long>(page << 16u));
        }
        std::fclose(file);
    }
}

void TimelineEvent(const char* what, std::uint64_t a, std::uint64_t b) {
    static const double startAt = [] { const char* text = std::getenv("APS5_NATIVE_TIMELINE"); return text != nullptr ? std::atof(text) : -1.0; }();
    if (startAt < 0) return;
    struct Event {
        double ms;
        const char* what;
        std::uint64_t a, b;
        long thread;
    };
    static const auto started = std::chrono::steady_clock::now();
    static std::mutex mutex;
    static std::vector<Event> events;
    static bool written = false;
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (seconds < startAt || written) return;
    std::lock_guard lock(mutex);
    if (written) return;
    if (seconds < startAt + 2.0) {
        events.push_back({seconds * 1000.0, what, a, b, static_cast<long>(gettid())});
        return;
    }
    written = true;
    const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR");
    const auto path = std::filesystem::path(directory != nullptr ? directory : ".") / "native-timeline.txt";
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    for (const auto& event : events) {
        char line[160];
        std::snprintf(line, sizeof(line), "%10.3f t%ld %s 0x%llx 0x%llx\n", event.ms, event.thread, event.what, static_cast<unsigned long long>(event.a), static_cast<unsigned long long>(event.b));
        file << line;
    }
    std::fprintf(stderr, "[native] timeline: %zu events written to %s\n", events.size(), path.string().c_str());
}

void ReportGraphicsIssue(const std::string& category, const std::string& detail) {
    auto& issues = Issues();
    std::lock_guard lock(issues.mutex);
    const auto key = category + ": " + detail;
    if (issues.counts[key]++ == 0) {
        std::fprintf(stderr, "[graphics] %s\n", key.c_str());
        WriteGraphicsIssues(issues);
        return;
    }
    issues.dirty = true;
    if (std::chrono::steady_clock::now() - issues.written > std::chrono::seconds(10)) WriteGraphicsIssues(issues);
}

bool Logs() {
    static const bool logs = EnvSet("APS5_NATIVE_LOG") || EnvSet("APS5_NATIVE_STATS");
    return logs;
}

bool GpuNative() {
    static const bool native = Enabled() && !SkipsDraws();
    return native;
}

namespace {
struct Slots {
    std::mutex mutex;
    std::condition_variable freed;
    unsigned available = std::max(2u, std::thread::hardware_concurrency() / 2u);
};
Slots& BackgroundSlots() {
    static Slots slots;
    return slots;
}
}

BackgroundSlot::BackgroundSlot(bool urgent) : counted(!urgent) {
    if (urgent) return;
    auto& slots = BackgroundSlots();
    std::unique_lock lock(slots.mutex);
    slots.freed.wait(lock, [&] { return slots.available != 0; });
    --slots.available;
}

BackgroundSlot::~BackgroundSlot() {
    if (!counted) return;
    auto& slots = BackgroundSlots();
    {
        std::lock_guard lock(slots.mutex);
        ++slots.available;
    }
    slots.freed.notify_one();
}

void NotePresent() {
    ReportStats(GetStats());
}

bool SkipsDraws() {
    static const bool skip = Enabled() && EnvSet("APS5_NATIVE_SKIP_DRAWS");
    return skip;
}

bool FindRecordedShader(std::uint64_t address, RecordedProgram& program) {
    std::lock_guard lock(ShaderMutex());
    auto& shaders = Shaders();
    auto it = shaders.upper_bound(static_cast<std::uintptr_t>(address));
    if (it == shaders.begin()) return false;
    --it;
    const auto bytes = it->second.code->size() * sizeof(std::uint32_t);
    if (address >= it->first + bytes) return false;
    program.codeAddress = it->first;
    program.type = it->second.type;
    program.code = it->second.code;
    program.header = it->second.header;
    program.headerAddress = it->second.headerAddress;
    program.positionVectorUse = it->second.positionVectorUse;
    return true;
}

void RecordShader(const Shader* shader) {
    GuestMemory::CheckRange(shader, sizeof(Shader), 1);
    Shader fields;
    std::memcpy(&fields, static_cast<const void*>(shader), sizeof(Shader));
    if (fields.file_header != 0x34333231u || fields.header_size < sizeof(Shader) || fields.shader_size == 0 || (fields.shader_size & 3u) != 0) {
        throw std::runtime_error("native backend: invalid shader header");
    }
    const auto* code = const_cast<const void*>(fields.code);
    GuestMemory::CheckRange(code, fields.shader_size, 256);
    RecordedShader recorded;
    recorded.type = fields.type;
    auto words = std::make_shared<std::vector<std::uint32_t>>(fields.shader_size / sizeof(std::uint32_t));
    std::memcpy(words->data(), code, fields.shader_size);
    auto header = std::make_shared<std::vector<std::byte>>(fields.header_size);
    std::memcpy(header->data(), static_cast<const void*>(shader), fields.header_size);
    recorded.code = std::move(words);
    recorded.header = std::move(header);
    recorded.headerAddress = reinterpret_cast<std::uintptr_t>(shader);
    std::lock_guard lock(ShaderMutex());
    Shaders()[reinterpret_cast<std::uintptr_t>(code)] = std::move(recorded);
}

namespace {

struct ProgramCost {
    std::uint64_t count = 0, indirect = 0;
    double ms = 0;
    std::uint32_t queues = 0;
};

std::uintptr_t ExecutableBase() {
    static const std::uintptr_t base = [] {
        std::uintptr_t found = 0;
        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* data) {
            *static_cast<std::uintptr_t*>(data) = info->dlpi_addr;
            return 1;  // the first entry is the main executable
        }, &found);
        return found;
    }();
    return base;
}

}

bool ProfilesDispatches() {
    static const bool profile = Enabled() && EnvSet("APS5_NATIVE_PROFILE");
    return profile;
}

void ProfileDispatch(std::uint64_t programAddress, std::uint32_t queue, bool indirect, double ms) {
    static std::mutex mutex;
    static std::map<std::uint64_t, ProgramCost> costs;
    static auto windowStart = std::chrono::steady_clock::now();
    static std::uint64_t flips0 = 0;
    std::lock_guard lock(mutex);
    auto& cost = costs[programAddress];
    ++cost.count;
    if (indirect) ++cost.indirect;
    cost.ms += ms;
    cost.queues |= 1u << std::min<std::uint32_t>(queue & 0x1f, 31);
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - windowStart).count();
    if (seconds < 10.0) return;
    std::vector<std::pair<std::uint64_t, ProgramCost>> rows(costs.begin(), costs.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    double total = 0;
    std::uint64_t dispatches = 0;
    for (const auto& row : rows) {
        total += row.second.ms;
        dispatches += row.second.count;
    }
    if (std::FILE* file = std::fopen("native-dispatch-profile.txt", "a")) {
        const auto base = ExecutableBase();
        std::fprintf(file, "window %.1f s: %llu dispatches, %.1f ms driver time, %zu programs (image base 0x%llx)\n", seconds, static_cast<unsigned long long>(dispatches), total, rows.size(), static_cast<unsigned long long>(base));
        for (std::size_t i = 0; i < rows.size() && i < 60; ++i) {
            const auto& [address, cost] = rows[i];
            const auto imageOffset = address >= base ? address - base : 0;
            std::fprintf(file, "  program 0x%llx image+0x%llx: %llu dispatches (%llu indirect), %.1f ms, %.3f ms each, queues mask 0x%x\n",
                         static_cast<unsigned long long>(address), static_cast<unsigned long long>(imageOffset), static_cast<unsigned long long>(cost.count), static_cast<unsigned long long>(cost.indirect), cost.ms, cost.ms / cost.count, cost.queues);
        }
        std::fclose(file);
    }
    costs.clear();
    windowStart = now;
    (void)flips0;
}

void CountSkippedDraw() { ++GetStats().draws; }
void CountSkippedDispatch() { ++GetStats().dispatches; }

void CountSubmission(std::uint32_t queue) {
    if (queue == 0) ++GetStats().submissions;
    else ++GetStats().asyncSubmissions;
}

}
