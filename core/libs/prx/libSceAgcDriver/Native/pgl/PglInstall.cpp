// Installs GT7's PGL ports (gt7_render_test engine/pgl) into the running game (APS5_PGL=1; APS5_PGL_ONLY narrows
// the set during bring-up).

#include "pgl/Context.h"

#include "port/X64.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>

namespace pgl {
// gt7render's differential tester (Diff.cpp) compares ports with translated bodies; there are none here.
void captureTranslated() {}
void diffAtFlush(Port::Cpu&) {}
}  // namespace pgl

namespace AgcDriver::Native {

void InstallPgl() {
    static bool done = false;
    if (done) return;
    done = true;
    const char* on = std::getenv("APS5_PGL");
    if (!on || *on != '1') return;
    if (!std::strstr(program_invocation_short_name, "gt7ps5")) return;   // the game only (not the pipeline preparer)
    std::fprintf(stderr, "pgl: image base %#llx, eboot limit %#llx\n", static_cast<unsigned long long>(Port::imageBase()),
                 static_cast<unsigned long long>(Port::imageLimit()));
    pgl::install({});
    std::fprintf(stderr, "pgl: ports installed\n");
}

}  // namespace AgcDriver::Native
