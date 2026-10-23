#pragma once
// The Port API of gt7_render_test (engine/port/X64.h) for native execution: GT7's PGL ports (gt7_render_test
// engine/pgl) run unchanged against the game's own code. There, every GT7 function is translated C++ over a register
// file; here GT7 runs natively, so
//   - a registered port replaces the function in the game image (a patched prologue, GameHooks);
//   - the port gets the guest's argument registers and stack arguments in a Cpu, and its rax / rdx / xmm0 / xmm1
//     become the function's results;
//   - Port::call / jump run either another port or GT7's code at that address (a real call with the Cpu's
//     registers and stack arguments);
//   - GT7 addresses inside the executable (code, rodata, data: eboot virtual addresses) are relocated by the image
//     base; every other address (heap, stack, mapped memory) is a host address already.

#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <array>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace Port {

using V = __m256;

struct Cpu {
    uint64_t g[16]{};   // rax rcx rdx rbx rsp rbp rsi rdi r8 .. r15
    V y[16]{};
    bool zf = false, sf = false, cf = false, of = false, pf = false;
    uint64_t fs = 0;    // the thread pointer (fs:0)
    uint64_t site = 0;
    uint64_t stack[512]{};   // a private stack for the port's pushes; the guest's arguments copied to its top
};
using Fn = void (*)(Cpu&);

// The relinked executable's base and the end of its eboot address range.
uint64_t imageBase();
uint64_t imageLimit();
inline uint64_t host(uint64_t a) { return a < imageLimit() ? a + imageBase() : a; }

void registerFunction(uint64_t gt7, Fn fn);
Fn lookupFunction(uint64_t gt7);   // the port registered at an address (null if none)
void call(Cpu& c, uint64_t target);
void jump(Cpu& c, uint64_t target);
void callOriginal(Cpu& c, uint64_t gt7);
[[noreturn]] void trap(const char* what, uint64_t address);

inline uint8_t* P(uint64_t a) { return reinterpret_cast<uint8_t*>(host(a)); }
inline uint8_t* G(uint64_t gt7) { return P(gt7); }
template <typename T> inline T rd(uint64_t a) { T v; std::memcpy(&v, P(a), sizeof(T)); return v; }
template <typename T> inline void wr(uint64_t a, T v) { std::memcpy(P(a), &v, sizeof(T)); }

inline void push(Cpu& c, uint64_t v) { c.g[4] -= 8; std::memcpy(reinterpret_cast<void*>(c.g[4]), &v, 8); }
inline uint64_t pop(Cpu& c) { uint64_t v; std::memcpy(&v, reinterpret_cast<const void*>(c.g[4]), 8); c.g[4] += 8; return v; }

}  // namespace Port
