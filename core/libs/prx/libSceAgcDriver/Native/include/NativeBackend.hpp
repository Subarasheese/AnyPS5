#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_NATIVEBACKEND_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_NATIVEBACKEND_HPP

// Native PC graphics backend for GT7.
//
// The game's own CPU code builds its frame (pipelines, passes, renderables); this backend renders
// it natively on the PC GPU (see NativeDevice.hpp) instead of emulating the PS5 GPU.
//
// It is the default. APS5_EMULATED_GPU=1 selects the emulating Vulkan backend instead.

#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include <cstdint>
#include <atomic>
#include <string>

namespace AgcDriver::Native {

bool Enabled();
// The default native renderer: GT7's command buffers run on the native
// executor (native compute and draws on the driver's Vulkan device, no PM4 emulation).
bool GpuNative();
// CPU-only profiling (APS5_NATIVE_SKIP_DRAWS=1): draw packets are counted and not rendered.
bool SkipsDraws();
// Background shader translations and pipeline builds hold one of these slots while they run (about
// half the CPU threads), so a burst of new shaders does not starve the game's own threads.
class BackgroundSlot {
public:
    BackgroundSlot();
    ~BackgroundSlot();
    BackgroundSlot(const BackgroundSlot&) = delete;
    BackgroundSlot& operator=(const BackgroundSlot&) = delete;
};
bool Logs();
void ReportGraphicsIssue(const std::string& category, const std::string& detail);
void TimelineEvent(const char* what, std::uint64_t a = 0, std::uint64_t b = 0);
// Debug aid: APS5_POOL_AUDIT=<file> lists, per 64 KiB page of GT7's memory, what uses it (command
// buffers, CPU descriptor walks, labels, vertex and index data, translated bases, textures).
void NotePoolUse(const char* use, std::uint64_t address);
// Per-frame statistics (APS5_NATIVE_STATS=1), counted at each presented frame.
void NotePresent();

// Keeps a copy of every registered shader program for native translation.
void RecordShader(const Shader* shader);
// The recorded program whose code contains `address`: its code (from `codeAddress`), header bytes
// and type. False when no registered program contains the address.
struct RecordedProgram {
    std::uint64_t codeAddress = 0;
    std::uint32_t type = 0;
    std::shared_ptr<const std::vector<std::uint32_t>> code;
    std::shared_ptr<const std::vector<std::byte>> header;
    std::uint64_t headerAddress = 0;
    std::shared_ptr<std::atomic<std::uint8_t>> positionVectorUse;
};
bool FindRecordedShader(std::uint64_t address, RecordedProgram& program);

// Per-frame counters reported by the presenter (APS5_NATIVE_STATS=1).
// Per-program compute profile (APS5_NATIVE_PROFILE=1): count and driver time per program, written
// every 10 s to native-dispatch-profile.txt with each program's offset in the executable image.
bool ProfilesDispatches();
void ProfileDispatch(std::uint64_t programAddress, std::uint32_t queue, bool indirect, double ms);

// Resource-creation audit (APS5_NATIVE_AUDIT=1, Native/src/ResourceAudit.cpp).
bool Audits();
void NoteSampledTexture(const std::uint32_t* words);

void CountSkippedDraw();
void CountSkippedDispatch();
void CountSubmission(std::uint32_t queue);

}

#endif
