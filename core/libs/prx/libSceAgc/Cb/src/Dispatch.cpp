#include "prx/libSceAgcDriver/Native/include/FrontEnd.hpp"
#include "prx/libSceAgc/Cb/include/Dispatch.hpp"

#include "prx/libSceAgc/Command/include/Packet.hpp"
#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

std::uint32_t* APS5_VABI sceAgcCbDispatch(CommandBuffer* buf, std::uint32_t threadGroupX, std::uint32_t threadGroupY, std::uint32_t threadGroupZ, std::uint32_t modifier) {
    Agc::Command::CheckBits(modifier, 0xa038u | 0x41u, __func__);
    const auto id = AgcFrontEndDraw_nid_postfix(buf, buf->bottom);
    if (id != 0) {
        auto* marker = Agc::Command::WriteNop(buf, 3u, __func__);
        marker[1] = AgcDriver::Native::FrontEndDrawMarker;
        marker[2] = id;
    }
    auto* packet = Agc::Command::Emit(buf, 0x15u, {threadGroupX, threadGroupY, threadGroupZ, modifier | 0x41u}, __func__);
    AgcFrontEndPrepareDispatch_nid_postfix(buf, buf->bottom, id, packet, 5u, false);
    return packet;
}

uint32_t APS5_VABI sceAgcCbDispatchGetSize(void) {
    return 20;
}

}
