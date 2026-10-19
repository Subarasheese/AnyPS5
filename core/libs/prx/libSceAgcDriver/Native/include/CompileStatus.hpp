#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_COMPILESTATUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_COMPILESTATUS_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace AgcDriver::Native {

class CompileJob {
public:
    CompileJob();
    ~CompileJob();
    void Compiling();
    CompileJob(const CompileJob&) = delete;
    CompileJob& operator=(const CompileJob&) = delete;

private:
    bool compiling = false;
};

// How long work waits for a translation or pipeline still compiling before it is skipped for this
// frame (APS5_COMPILE_WAIT_MS, default 2000; 0 waits until it completes): a shader the driver takes
// minutes to compile then costs missing work for those frames, not a frozen game.
std::chrono::milliseconds CompileWaitLimit();
// The wait to use now: CompileWaitLimit, or 1 ms for a few seconds after a wait timed out. A burst
// of new shaders (a race loading its materials) then costs skipped draws, not one limit per shader.
std::chrono::milliseconds CompileWaitNow();
void NoteCompileWaitTimedOut();

class CompileWait {
public:
    CompileWait();
    ~CompileWait();
    CompileWait(const CompileWait&) = delete;
    CompileWait& operator=(const CompileWait&) = delete;
};

struct CompileProgress {
    bool blocked = false;
    double blockedSeconds = 0;
    std::uint64_t finished = 0;
    std::uint64_t total = 0;
    bool compiling = false;
};

CompileProgress CurrentCompileProgress();
void RenderCompileStatus(std::uint32_t width, std::uint32_t height, const CompileProgress& progress, std::vector<std::byte>& pixels);

}

#endif
