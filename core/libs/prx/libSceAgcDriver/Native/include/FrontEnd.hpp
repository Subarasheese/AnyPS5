#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_FRONTEND_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_NATIVE_INCLUDE_FRONTEND_HPP

#include <cstdint>
#include <memory>

extern "C" {

void AgcFrontEndRegisters_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t opcode, const volatile void* registers, std::uint32_t count);
void AgcFrontEndRegisterRange_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t opcode, std::uint32_t offset, const std::uint32_t* values, std::uint32_t count);
void AgcFrontEndIndexState_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t field, std::uint64_t value);
void AgcFrontEndReset_nid_postfix(const void* buffer, const std::uint32_t* bottom);
std::uint32_t AgcFrontEndDraw_nid_postfix(const void* buffer, const std::uint32_t* bottom);
void AgcFrontEndPrepareDraw_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t id, const std::uint32_t* packet, std::uint32_t words);
void AgcFrontEndPrepareDispatch_nid_postfix(const void* buffer, const std::uint32_t* bottom, std::uint32_t id, const std::uint32_t* packet, std::uint32_t words, bool indirect);

}

namespace AgcDriver {
struct QueueState;
}

namespace AgcDriver::Native {

inline constexpr std::uint32_t FrontEndDrawMarker = 0x46454452u;

enum class FrontEndIndexField : std::uint32_t { Buffer, Size, Count, Instances, DrawIndirectBase, DispatchIndirectBase };

bool FrontEndValidating();
bool FrontEndDraws();
std::shared_ptr<void> TakeFrontEndDraw(std::uint32_t id);
void CompareFrontEndDraw(std::uint32_t id, const QueueState& queue);

}

#endif
