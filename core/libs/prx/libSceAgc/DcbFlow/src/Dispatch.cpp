#include "prx/libSceAgcDriver/Native/include/FrontEnd.hpp"
#include "prx/libSceAgc/DcbFlow/include/Dispatch.hpp"

#include "prx/libSceAgc/Command/include/Memory.hpp"
#include "prx/libSceAgc/Command/include/Packet.hpp"
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

uint32_t* APS5_VABI sceAgcDcbDispatchIndirect(CommandBuffer* buf, uint32_t data_offset_in_bytes, uint32_t flags) {
    // Graphics queues read the arguments at an offset from the base set by SetBaseDispatchIndirectArgs.
    Agc::Command::CheckBits(flags, 0xa038u | 0x41u, __func__);
    const auto id = AgcFrontEndDraw_nid_postfix(buf, buf->bottom);
    if (id != 0) {
        auto* marker = Agc::Command::WriteNop(buf, 3u, __func__);
        marker[1] = AgcDriver::Native::FrontEndDrawMarker;
        marker[2] = id;
    }
    auto* packet = Agc::Command::Emit(buf, 0x16u, {data_offset_in_bytes, flags | 0x41u}, __func__);
    AgcFrontEndPrepareDispatch_nid_postfix(buf, buf->bottom, id, packet, 3u, true);
    return packet;
}

std::uint32_t APS5_VABI sceAgcDcbDispatchIndirectGetSize() {
    return 12;
}

}
