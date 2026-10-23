// Native-execution implementation of the gt7_render_test Port API (include/port/X64.h).

#include "port/X64.h"

#include "prx/libSceAgcDriver/Native/include/GameHooks.hpp"

#include <link.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <map>
#include <vector>

namespace {

// The registers the entry / guest-call stubs exchange (offsets used by the assembly below).
struct Regs {
    uint64_t rdi, rsi, rdx, rcx, r8, r9;   // 0..40 (rax at 0 and rdx at 16 on return)
    uint64_t rsp;                          // 48: the guest's rsp at entry (its return address)
    uint64_t pad;                          // 56
    uint8_t xmm[8][16];                    // 64..191
};
static_assert(sizeof(Regs) == 192);

}  // namespace

extern "C" void PglPortEntry();   // r11 = the port (Port::Fn)
extern "C" void PglPortCallGuest(Regs* regs, uint64_t target, const uint64_t* stack, uint64_t count);

// PglPortEntry: the guest's call lands here (through the patched prologue and the hook slot, r11 = the port).
// PglPortCallGuest(regs, target, stack, count): calls `target` with the argument registers in `regs` and `count`
// stack arguments from `stack`; rax / rdx / xmm0 / xmm1 come back in regs (0 / 16 / 64 / 80).
asm(R"(
    .text
    .globl PglPortEntry
    .type PglPortEntry, @function
PglPortEntry:
    push %rbp
    mov %rsp, %rbp
    sub $0xc0, %rsp
    and $-32, %rsp
    mov %rdi, 0(%rsp)
    mov %rsi, 8(%rsp)
    mov %rdx, 16(%rsp)
    mov %rcx, 24(%rsp)
    mov %r8, 32(%rsp)
    mov %r9, 40(%rsp)
    lea 8(%rbp), %rax
    mov %rax, 48(%rsp)
    movdqu %xmm0, 64(%rsp)
    movdqu %xmm1, 80(%rsp)
    movdqu %xmm2, 96(%rsp)
    movdqu %xmm3, 112(%rsp)
    movdqu %xmm4, 128(%rsp)
    movdqu %xmm5, 144(%rsp)
    movdqu %xmm6, 160(%rsp)
    movdqu %xmm7, 176(%rsp)
    mov %rsp, %rdi
    mov %r11, %rsi
    call PglPortDispatch
    mov 0(%rsp), %rax
    mov 16(%rsp), %rdx
    movdqu 64(%rsp), %xmm0
    movdqu 80(%rsp), %xmm1
    leave
    ret

    .globl PglPortCallGuest
    .type PglPortCallGuest, @function
PglPortCallGuest:
    push %rbp
    mov %rsp, %rbp
    push %rbx
    push %r12
    mov %rdi, %rbx
    mov %rsi, %r12
    mov %rcx, %rax
    shl $3, %rax
    sub %rax, %rsp
    and $-16, %rsp
    mov %rsp, %rdi
    mov %rdx, %rsi
    cld
    rep movsq
    mov 8(%rbx), %rsi
    mov 16(%rbx), %rdx
    mov 24(%rbx), %rcx
    mov 32(%rbx), %r8
    mov 40(%rbx), %r9
    movdqu 64(%rbx), %xmm0
    movdqu 80(%rbx), %xmm1
    movdqu 96(%rbx), %xmm2
    movdqu 112(%rbx), %xmm3
    movdqu 128(%rbx), %xmm4
    movdqu 144(%rbx), %xmm5
    movdqu 160(%rbx), %xmm6
    movdqu 176(%rbx), %xmm7
    mov 0(%rbx), %rdi
    mov $8, %eax
    call *%r12
    mov %rax, 0(%rbx)
    mov %rdx, 16(%rbx)
    movdqu %xmm0, 64(%rbx)
    movdqu %xmm1, 80(%rbx)
    lea -16(%rbp), %rsp
    pop %r12
    pop %rbx
    pop %rbp
    ret
)");

namespace Port {
namespace {

constexpr uint64_t StackArguments = 32;   // stack argument slots copied in and out

struct Registry {
    std::mutex mutex;
    std::map<uint64_t, Fn> ports;           // by eboot address (node-stable: the hook stubs hold &ports[a])
    std::map<uint64_t, void*> originals;    // trampolines into GT7's own code
};
Registry& registry() {
    static Registry r;
    return r;
}

// The filter APS5_PGL_ONLY=addr,addr (hex): only those ports are installed (bring-up).
bool wanted(uint64_t address) {
    static const std::vector<uint64_t> only = [] {
        std::vector<uint64_t> list;
        if (const char* v = std::getenv("APS5_PGL_ONLY"))
            for (const char* p = v; *p;) {
                char* end = nullptr;
                const uint64_t a = std::strtoull(p, &end, 16);
                if (end == p) break;
                list.push_back(a);
                p = *end ? end + 1 : end;
            }
        return list;
    }();
    if (only.empty()) return true;
    for (const uint64_t a : only)
        if (a == address) return true;
    return false;
}

thread_local std::vector<Cpu*> t_free;   // a Cpu per nesting level, reused
Cpu* acquire() {
    if (t_free.empty()) return new Cpu;
    Cpu* c = t_free.back();
    t_free.pop_back();
    return c;
}
void release(Cpu* c) { t_free.push_back(c); }

uint64_t threadPointer() {
    uint64_t tp;
    asm volatile("mov %%fs:0, %0" : "=r"(tp));
    return tp;
}

void toRegs(const Cpu& c, Regs& r) {
    r.rdi = c.g[7], r.rsi = c.g[6], r.rdx = c.g[2], r.rcx = c.g[1], r.r8 = c.g[8], r.r9 = c.g[9];
    for (int k = 0; k < 8; ++k) std::memcpy(r.xmm[k], &c.y[k], 16);
}
void fromRegs(Cpu& c, const Regs& r) {
    c.g[0] = r.rdi;   // rax (the stub stores it at offset 0)
    c.g[2] = r.rdx;
    c.y[0] = _mm256_setzero_ps();
    c.y[1] = _mm256_setzero_ps();
    std::memcpy(&c.y[0], r.xmm[0], 16);
    std::memcpy(&c.y[1], r.xmm[1], 16);
}

// GT7's code at eboot address (or host address) `target`, called with the Cpu's registers; the stack arguments are
// the Cpu's stack from its current top (Port::call's caller pushed them; no return slot yet).
void callGuest(Cpu& c, uint64_t target) {
    Regs r{};
    toRegs(c, r);
    PglPortCallGuest(&r, host(target), reinterpret_cast<const uint64_t*>(c.g[4]), StackArguments);
    fromRegs(c, r);
}

}  // namespace

uint64_t imageBase() { return AgcDriver::Native::GameImageBase(); }

uint64_t imageLimit() {
    static const uint64_t limit = [] {
        uint64_t end = 0;
        dl_iterate_phdr([](dl_phdr_info* info, size_t, void* data) {
            auto& e = *static_cast<uint64_t*>(data);
            for (size_t i = 0; i < info->dlpi_phnum; ++i)
                if (info->dlpi_phdr[i].p_type == PT_LOAD) e = std::max<uint64_t>(e, info->dlpi_phdr[i].p_vaddr + info->dlpi_phdr[i].p_memsz);
            return 1;   // the first entry is the main executable
        }, &end);
        return end;
    }();
    return limit;
}

void registerFunction(uint64_t gt7, Fn fn) {
    if (!wanted(gt7)) return;
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    if (r.ports.count(gt7)) { r.ports[gt7] = fn; return; }   // re-registration: the hook stays, the port changes
    r.ports[gt7] = fn;
    char name[32];
    std::snprintf(name, sizeof name, "pgl %#llx", static_cast<unsigned long long>(gt7));
    r.originals[gt7] = AgcDriver::Native::InstallNativeHook(strdup(name), gt7, reinterpret_cast<void*>(&PglPortEntry), reinterpret_cast<void*>(&r.ports[gt7]));
}

Fn lookupFunction(uint64_t gt7) {
    auto& r = registry();
    std::lock_guard lock(r.mutex);
    const auto it = r.ports.find(gt7);
    return it == r.ports.end() ? nullptr : it->second;
}

void call(Cpu& c, uint64_t target) {
    const uint64_t address = target >= imageBase() && target - imageBase() < imageLimit() ? target - imageBase() : target;
    if (const Fn fn = lookupFunction(address)) {
        push(c, 0);
        fn(c);
        return;
    }
    callGuest(c, target);
}

void jump(Cpu& c, uint64_t target) {
    call(c, target);
    pop(c);   // the callee returns through the current slot
}

void callOriginal(Cpu& c, uint64_t gt7) {
    void* original = nullptr;
    {
        auto& r = registry();
        std::lock_guard lock(r.mutex);
        const auto it = r.originals.find(gt7);
        if (it != r.originals.end()) original = it->second;
    }
    if (!original) trap("no original body to call (non-standard prologue)", gt7);
    callGuest(c, reinterpret_cast<uint64_t>(original));
}

void trap(const char* what, uint64_t address) {
    std::fprintf(stderr, "pgl port: %s (%#llx)\n", what, static_cast<unsigned long long>(address));
    std::abort();
}

}  // namespace Port

// The entry stub's C side: the guest's registers and stack arguments into a Cpu, the port, its results back.
// `fn` points at the registry's slot for the port (so a re-registration takes effect).
extern "C" void PglPortDispatch(Regs* r, const Port::Fn* fn) {
    Port::Cpu* c = Port::acquire();
    std::memset(c->g, 0, sizeof c->g);
    c->g[7] = r->rdi, c->g[6] = r->rsi, c->g[2] = r->rdx, c->g[1] = r->rcx, c->g[8] = r->r8, c->g[9] = r->r9;
    for (int k = 0; k < 8; ++k) {
        c->y[k] = _mm256_setzero_ps();
        std::memcpy(&c->y[k], r->xmm[k], 16);
    }
    c->fs = Port::threadPointer();
    // The return slot and the stack arguments, at the top of the Cpu's private stack.
    uint64_t* top = c->stack + (sizeof c->stack / 8) - (Port::StackArguments + 1);
    std::memcpy(top, reinterpret_cast<const void*>(r->rsp), (Port::StackArguments + 1) * 8);
    c->g[4] = reinterpret_cast<uint64_t>(top);
    (*fn)(*c);
    r->rdi = c->g[0];   // rax (offset 0)
    r->rdx = c->g[2];
    std::memcpy(r->xmm[0], &c->y[0], 16);
    std::memcpy(r->xmm[1], &c->y[1], 16);
    Port::release(c);
}
