// Placeholder module: the title lists this library as needed but imports no functions from it.
// The exports exist only so the library has a symbol table the NID patcher can rebuild.
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

APS5_EXPORT("AAAAAAAAAAA", PlaceholderExport0);
int APS5_VABI PlaceholderExport0(void) {
    NotImplemented_nid_no_patch("AAAAAAAAAAA");
    return 0;
}

APS5_EXPORT("AAAAAAAAAAB", PlaceholderExport1);
int APS5_VABI PlaceholderExport1(void) {
    NotImplemented_nid_no_patch("AAAAAAAAAAB");
    return 0;
}

}
