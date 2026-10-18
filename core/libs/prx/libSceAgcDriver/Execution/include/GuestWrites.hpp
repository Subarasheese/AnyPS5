#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GUESTWRITES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GUESTWRITES_HPP

#include <cstddef>

extern "C" void AgcDriverNoteGuestWrite_nid_postfix(const void* address, std::size_t bytes);

#endif
