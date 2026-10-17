#pragma once
#include <cstdint>

namespace AgcDriver::Graphics {
struct StageLibraryCaptureHeader {
    static constexpr std::uint32_t Magic = 0x314c5041u;
    std::uint32_t magic;
    std::uint32_t stage;
    std::uint32_t vertexWave;
    std::uint32_t fragmentWave;
    std::uint32_t pushOffset;
    std::uint32_t payloadBytes;
    std::uint64_t sourceVersion;
    std::uint64_t payloadHash;
};
static_assert(sizeof(StageLibraryCaptureHeader) == 40);
}
