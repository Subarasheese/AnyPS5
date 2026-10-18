#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_COMPILESTATUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_COMPILESTATUS_HPP

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
