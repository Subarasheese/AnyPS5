#include "prx/libSceAgcDriver/Native/include/GameHooks.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <link.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace AgcDriver::Native {

namespace {

constexpr std::size_t SlotBytes = 64;
constexpr std::size_t TrampolineOffset = 24;
constexpr std::size_t CounterOffset = 56;

// Executable memory within rel32 reach of the image, carved into per-hook slots:
// [0: thunk][24: trampoline = stolen prologue + jmp original+n][56: call counter].
struct NearPool {
    std::mutex mutex;
    std::uint8_t* begin = nullptr;
    std::size_t used = 0;
    std::size_t size = 0;
};

NearPool& Pool() {
    static NearPool pool;
    return pool;
}

std::uint8_t* AllocateSlot(std::uintptr_t near) {
    auto& pool = Pool();
    std::lock_guard lock(pool.mutex);
    if (pool.begin == nullptr) {
        const std::size_t size = 1u << 20u;
        const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
        for (std::uintptr_t distance = 16ull << 20u; distance < (1ull << 30u) && pool.begin == nullptr; distance += 16ull << 20u) {
            for (const std::intptr_t sign : {-1, 1}) {
                const auto hint = (near + sign * static_cast<std::intptr_t>(distance)) & ~(page - 1);
                void* mapped = mmap(reinterpret_cast<void*>(hint), size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
                if (mapped == MAP_FAILED) continue;
                pool.begin = static_cast<std::uint8_t*>(mapped);
                pool.size = size;
                break;
            }
        }
        if (pool.begin == nullptr) throw std::runtime_error("native hooks: no executable memory within 1 GiB of the game image");
    }
    if (pool.used + SlotBytes > pool.size) throw std::runtime_error("native hooks: hook pool exhausted");
    auto* slot = pool.begin + pool.used;
    pool.used += SlotBytes;
    std::memset(slot, 0xcc, SlotBytes);
    return slot;
}

void WriteAbsoluteJump(std::uint8_t* at, std::uintptr_t target) {
    at[0] = 0xff;  // jmp qword [rip+0]
    at[1] = 0x25;
    std::memset(at + 2, 0, 4);
    std::memcpy(at + 6, &target, sizeof(target));
}

// Length of the relocatable prologue to displace (whole instructions, at least 5 bytes):
//   55 48 89 e5 41 5x           push rbp; mov rbp, rsp; push r8..r15
//   55 48 89 e5 53 48 83 ec ib  push rbp; mov rbp, rsp; push rbx; sub rsp, imm8
std::size_t StolenLength(const std::uint8_t* code) {
    if (code[0] != 0x55 || code[1] != 0x48 || code[2] != 0x89 || code[3] != 0xe5) return 0;
    if (code[4] == 0x41 && (code[5] & 0xf8) == 0x50) return 6;
    if (code[4] == 0x53 && code[5] == 0x48 && code[6] == 0x83 && code[7] == 0xec) return 9;
    return 0;
}

std::uint8_t* CheckedTarget(const char* name, std::uintptr_t ebootAddress, std::size_t& stolen) {
    if (!GameCodeMapped(ebootAddress, 16u)) throw std::runtime_error(std::string("native hooks: target is outside the main executable code: ") + name);
    auto* target = reinterpret_cast<std::uint8_t*>(GameImageBase() + ebootAddress);
    stolen = StolenLength(target);
    if (stolen == 0) {
        char bytes[32] = {};
        for (std::size_t i = 0; i < 9; ++i) std::snprintf(bytes + 3 * i, 4, "%02x ", target[i]);
        throw std::runtime_error(std::string("native hooks: unexpected prologue at ") + name + ": " + bytes);
    }
    return target;
}

void PatchEntry(const char* name, std::uint8_t* target, std::size_t stolen, std::uint8_t* thunk) {
    const auto relative = reinterpret_cast<std::intptr_t>(thunk) - reinterpret_cast<std::intptr_t>(target + 5);
    if (relative < INT32_MIN || relative > INT32_MAX) throw std::runtime_error(std::string("native hooks: thunk out of rel32 range for ") + name);
    const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    const auto first = reinterpret_cast<std::uintptr_t>(target) & ~(page - 1);
    const auto last = (reinterpret_cast<std::uintptr_t>(target) + stolen - 1) & ~(page - 1);
    const auto span = last - first + page;
    if (mprotect(reinterpret_cast<void*>(first), span, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) throw std::runtime_error(std::string("native hooks: mprotect failed for ") + name);
    std::uint8_t patch[16];
    std::memset(patch, 0x90, sizeof(patch));
    patch[0] = 0xe9;
    const auto rel32 = static_cast<std::int32_t>(relative);
    std::memcpy(patch + 1, &rel32, sizeof(rel32));
    std::memcpy(target, patch, stolen);
    mprotect(reinterpret_cast<void*>(first), span, PROT_READ | PROT_EXEC);
}

std::uint8_t* BuildTrampoline(std::uint8_t* slot, std::uint8_t* target, std::size_t stolen) {
    auto* trampoline = slot + TrampolineOffset;
    std::memcpy(trampoline, target, stolen);
    WriteAbsoluteJump(trampoline + stolen, reinterpret_cast<std::uintptr_t>(target + stolen));
    return trampoline;
}

struct CountedHook {
    const char* name;
    std::uintptr_t ebootAddress;
    const std::uint64_t* counter;
    std::uint64_t last = 0;
};

std::mutex& CountedMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<CountedHook>& Counted() {
    static std::vector<CountedHook> hooks;
    return hooks;
}

}

std::uintptr_t GameImageBase() {
    static const std::uintptr_t base = [] {
        std::uintptr_t found = 0;
        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* data) {
            *static_cast<std::uintptr_t*>(data) = info->dlpi_addr;
            return 1;  // the first entry is the main executable (the relinked eboot)
        }, &found);
        return found;
    }();
    return base;
}

bool GameCodeMapped(std::uintptr_t ebootAddress, std::size_t bytes) {
    struct Probe { std::uintptr_t offset; std::size_t bytes; bool readable = false; } probe{ebootAddress, bytes};
    dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* data) {
        auto& probe = *static_cast<Probe*>(data);
        for (std::size_t i = 0; i < info->dlpi_phnum; ++i) {
            const auto& segment = info->dlpi_phdr[i];
            if (segment.p_type != PT_LOAD || (segment.p_flags & (PF_R | PF_X)) != (PF_R | PF_X)) continue;
            if (probe.offset >= segment.p_vaddr && probe.offset - segment.p_vaddr <= segment.p_memsz && segment.p_memsz - (probe.offset - segment.p_vaddr) >= probe.bytes) {
                probe.readable = true;
                break;
            }
        }
        return 1;
    }, &probe);
    return probe.readable;
}

void* InstallGameHook(const char* name, std::uintptr_t ebootAddress, void* handler) {
    std::size_t stolen = 0;
    auto* target = CheckedTarget(name, ebootAddress, stolen);
    auto* slot = AllocateSlot(reinterpret_cast<std::uintptr_t>(target));
    WriteAbsoluteJump(slot, reinterpret_cast<std::uintptr_t>(handler));
    auto* trampoline = BuildTrampoline(slot, target, stolen);
    PatchEntry(name, target, stolen, slot);
    return trampoline;
}

void InstallCountingHook(const char* name, std::uintptr_t ebootAddress, std::uint64_t** counter) {
    std::size_t stolen = 0;
    auto* target = CheckedTarget(name, ebootAddress, stolen);
    auto* slot = AllocateSlot(reinterpret_cast<std::uintptr_t>(target));
    auto* count = reinterpret_cast<std::uint64_t*>(slot + CounterOffset);
    *count = 0;
    // lock inc qword [rip+rel32] (f0 48 ff 05 rel32: 8 bytes); jmp [rip+0] trampoline (14 bytes)
    const auto rel = static_cast<std::int32_t>(reinterpret_cast<std::intptr_t>(count) - reinterpret_cast<std::intptr_t>(slot + 8));
    slot[0] = 0xf0;
    slot[1] = 0x48;
    slot[2] = 0xff;
    slot[3] = 0x05;
    std::memcpy(slot + 4, &rel, sizeof(rel));
    auto* trampoline = BuildTrampoline(slot, target, stolen);
    WriteAbsoluteJump(slot + 8, reinterpret_cast<std::uintptr_t>(trampoline));
    PatchEntry(name, target, stolen, slot);
    if (counter != nullptr) *counter = count;
    std::lock_guard lock(CountedMutex());
    Counted().push_back({name, ebootAddress, count});
}

void InstallGraphicsContextHooks() {
    static std::once_flag once;
    std::call_once(once, [] {
        // APS5_NATIVE_COUNT=va,va,...: counts calls of arbitrary eboot functions (reported with the rest).
        if (const char* list = std::getenv("APS5_NATIVE_COUNT")) {
            static std::deque<std::string> counted;
            std::string text(list);
            for (std::size_t at = 0; at < text.size();) {
                auto next = text.find(',', at);
                if (next == std::string::npos) next = text.size();
                const auto address = std::strtoull(text.substr(at, next - at).c_str(), nullptr, 16);
                char name[32];
                std::snprintf(name, sizeof(name), "fn_%llx", static_cast<unsigned long long>(address));
                counted.emplace_back(name);
                try {
                    InstallCountingHook(counted.back().c_str(), address, nullptr);
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "[native] %s\n", error.what());
                }
                at = next + 1;
            }
        }
        const bool context = std::getenv("APS5_NATIVE_TRACE_CONTEXT") != nullptr;
        if (!context && Counted().empty()) return;
        // The command-emitting methods of GT7's graphics context (*(fs-0xb8)): every function that
        // loads the context and writes its command stream (+0x788). See the GT7 decompilation.
        static constexpr std::uintptr_t Emitters[] = {
            0x1440, 0x5fb0, 0x9bb0, 0x18740, 0x188a0, 0x1a070, 0x2b5d0, 0x4b9f0, 0x53930, 0x982a0, 0x98be0, 0x98d60,
            0x9a9a0, 0xa1400, 0x10e810, 0x110a10, 0x16c970, 0x17e820, 0x188400, 0x188ad0, 0x188e40, 0x18b960,
            0x191e00, 0x195830, 0x19cff0, 0x19dfb0, 0x19e8c0, 0x1d0350, 0x274ff0, 0x2d0c50, 0x62b740, 0xa89520, 0xbc1580,
            0x1a34000, 0x1a38ac0, 0x1a39690, 0x1a3a890, 0x2bd8c00, 0x2bd95c0, 0x2d91500, 0x2dd0340, 0x2ddfc90, 0x2df1d50,
            0x2df4000, 0x2f17e00, 0x1a389c0, 0x1a38a40, 0x1a39550, 0x1a395f0, 0x1a3a7b0, 0x1d9bb0, 0x1a42560, 0x16370};
        static std::vector<std::string> names;
        names.reserve(std::size(Emitters));
        for (const auto address : Emitters) {
            if (!context) break;
            char text[32];
            std::snprintf(text, sizeof(text), "fn_%llx", static_cast<unsigned long long>(address));
            names.emplace_back(text);
            try {
                InstallCountingHook(names.back().c_str(), address, nullptr);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[native] %s\n", error.what());
            }
        }
        std::fprintf(stderr, "[native] traced %zu functions\n", Counted().size());
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                std::lock_guard lock(CountedMutex());
                std::string line = "[native] context calls/2s:";
                for (auto& hook : Counted()) {
                    const auto now = __atomic_load_n(hook.counter, __ATOMIC_RELAXED);
                    if (now != hook.last) line += " " + std::string(hook.name + 3) + "=" + std::to_string(now - hook.last);
                    hook.last = now;
                }
                std::fprintf(stderr, "%s\n", line.c_str());
            }
        }).detach();
    });
}

}
