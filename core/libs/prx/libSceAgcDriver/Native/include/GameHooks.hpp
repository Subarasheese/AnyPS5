#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_GAMEHOOKS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_GAMEHOOKS_HPP

// Redirects GT7 graphics-context methods to native implementations.
//
// The relinked executable keeps GT7's code at its original virtual addresses (host address =
// image base + eboot VA). Every hooked method starts with `push rbp; mov rbp, rsp; push r1x`
// (6 bytes). The hook replaces those bytes with a jump to a thunk next to the image. The thunk
// jumps to the native handler. The original method stays callable through a trampoline that
// runs the 6 displaced bytes and continues after them.

#include <cstdint>

namespace AgcDriver::Native {

// Installs one hook; returns the trampoline that runs the original method. Throws on mismatch.
void* InstallGameHook(const char* name, std::uintptr_t ebootAddress, void* handler);

// Observation hook: counts calls in `counter` (atomic increment in the thunk) and runs the original.
void InstallCountingHook(const char* name, std::uintptr_t ebootAddress, std::uint64_t** counter);

std::uintptr_t GameImageBase();

// Installs the GT7 graphics-context hooks (native backend start-up).
void InstallGraphicsContextHooks();

}

#endif
