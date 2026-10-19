#include "prx/libSceAgcDriver/Native/include/CompileStatus.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "stb_truetype.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace AgcDriver::Native {

namespace {

struct Counters {
    std::mutex mutex;
    std::uint64_t queued = 0;
    std::uint64_t finished = 0;
    std::uint64_t baseline = 0;
    std::uint64_t compiling = 0;
    int waiters = 0;
    std::chrono::steady_clock::time_point blockedSince{};
    std::chrono::steady_clock::time_point unblockedAt{};
};

Counters& State() {
    static Counters counters;
    return counters;
}

struct Font {
    std::vector<unsigned char> data;
    stbtt_fontinfo info{};
};

const Font* UiFont() {
    static const std::unique_ptr<Font> font = [] {
        auto made = std::make_unique<Font>();
        std::vector<std::string> candidates;
        if (const char* path = std::getenv("APS5_UI_FONT")) candidates.emplace_back(path);
        for (const char* path : {"/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
                 "/usr/share/fonts/google-noto/NotoSans-Regular.ttf", "/usr/share/fonts/noto-fonts/NotoSans-Regular.ttf",
                 "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"}) {
            candidates.emplace_back(path);
        }
        for (const auto& path : candidates) {
            std::ifstream file(path, std::ios::binary);
            if (!file) continue;
            made->data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            if (made->data.empty()) continue;
            const int offset = stbtt_GetFontOffsetForIndex(made->data.data(), 0);
            if (offset >= 0 && stbtt_InitFont(&made->info, made->data.data(), offset) != 0) return made;
        }
        return std::unique_ptr<Font>();
    }();
    return font.get();
}

struct Canvas {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::byte>& pixels;

    void blend(std::int64_t x, std::int64_t y, std::uint8_t level, std::uint8_t coverage) {
        if (x < 0 || y < 0 || x >= width || y >= height || coverage == 0) return;
        auto* pixel = pixels.data() + (static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)) * 4u;
        for (int channel = 0; channel < 3; ++channel) {
            const auto old = static_cast<unsigned>(pixel[channel]);
            pixel[channel] = static_cast<std::byte>((old * (255u - coverage) + static_cast<unsigned>(level) * coverage) / 255u);
        }
    }

    void fill(std::int64_t x, std::int64_t y, std::int64_t w, std::int64_t h, std::uint8_t level) {
        for (std::int64_t row = y; row < y + h; ++row) {
            for (std::int64_t column = x; column < x + w; ++column) blend(column, row, level, 255);
        }
    }
};

float TextWidth(const Font& font, const std::string& text, float pixelHeight) {
    const float scale = stbtt_ScaleForPixelHeight(&font.info, pixelHeight);
    float width = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        int advance = 0, bearing = 0;
        stbtt_GetCodepointHMetrics(&font.info, static_cast<unsigned char>(text[i]), &advance, &bearing);
        width += static_cast<float>(advance) * scale;
        if (i + 1 < text.size()) width += static_cast<float>(stbtt_GetCodepointKernAdvance(&font.info, static_cast<unsigned char>(text[i]), static_cast<unsigned char>(text[i + 1]))) * scale;
    }
    return width;
}

void DrawText(Canvas& canvas, const Font& font, const std::string& text, float x, float baseline, float pixelHeight, std::uint8_t level) {
    const float scale = stbtt_ScaleForPixelHeight(&font.info, pixelHeight);
    float pen = x;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const int codepoint = static_cast<unsigned char>(text[i]);
        int width = 0, height = 0, left = 0, top = 0;
        unsigned char* bitmap = stbtt_GetCodepointBitmapSubpixel(&font.info, scale, scale, pen - static_cast<float>(static_cast<int>(pen)), 0.0f, codepoint, &width, &height, &left, &top);
        for (int row = 0; row < height; ++row) {
            for (int column = 0; column < width; ++column) {
                canvas.blend(static_cast<std::int64_t>(pen) + left + column, static_cast<std::int64_t>(baseline) + top + row, level, bitmap[row * width + column]);
            }
        }
        stbtt_FreeBitmap(bitmap, nullptr);
        int advance = 0, bearing = 0;
        stbtt_GetCodepointHMetrics(&font.info, codepoint, &advance, &bearing);
        pen += static_cast<float>(advance) * scale;
        if (i + 1 < text.size()) pen += static_cast<float>(stbtt_GetCodepointKernAdvance(&font.info, codepoint, static_cast<unsigned char>(text[i + 1]))) * scale;
    }
}

}

CompileJob::CompileJob() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    ++state.queued;
}

CompileJob::~CompileJob() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    ++state.finished;
    if (compiling) --state.compiling;
}

void CompileJob::Compiling() {
    if (compiling) return;
    compiling = true;
    auto& state = State();
    std::lock_guard lock(state.mutex);
    ++state.compiling;
}

CompileWait::CompileWait() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (state.waiters++ == 0 && now - state.unblockedAt > std::chrono::milliseconds(500)) state.blockedSince = now;
}

CompileWait::~CompileWait() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (--state.waiters == 0) state.unblockedAt = std::chrono::steady_clock::now();
}

std::chrono::milliseconds CompileWaitLimit() {
    static const std::chrono::milliseconds limit{[] { const char* text = std::getenv("APS5_COMPILE_WAIT_MS"); return text != nullptr ? std::strtoll(text, nullptr, 10) : 2000ll; }()};
    return limit;
}

namespace {
std::atomic<std::int64_t> lastTimeout{std::numeric_limits<std::int64_t>::min()};
std::int64_t NowMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}

std::chrono::milliseconds CompileWaitNow() {
    const auto last = lastTimeout.load(std::memory_order_relaxed);
    if (last != std::numeric_limits<std::int64_t>::min() && NowMs() - last < 3000) return std::chrono::milliseconds{1};
    return CompileWaitLimit();
}

void NoteCompileWaitTimedOut() {
    lastTimeout.store(NowMs(), std::memory_order_relaxed);
}

CompileProgress CurrentCompileProgress() {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    CompileProgress progress;
    progress.blocked = state.waiters != 0 && state.queued != state.finished;
    if (state.waiters == 0 && state.queued == state.finished) state.baseline = state.finished;
    if (progress.blocked) progress.blockedSeconds = std::chrono::duration<double>(now - state.blockedSince).count();
    progress.finished = state.finished - state.baseline;
    progress.total = std::max(state.queued - state.baseline, progress.finished);
    progress.compiling = state.compiling != 0;
    return progress;
}

void RenderCompileStatus(std::uint32_t width, std::uint32_t height, const CompileProgress& progress, std::vector<std::byte>& pixels) {
    pixels.assign(static_cast<std::size_t>(width) * height * 4u, std::byte{0});
    for (std::size_t i = 3; i < pixels.size(); i += 4) pixels[i] = std::byte{255};
    Canvas canvas{width, height, pixels};
    const float unit = static_cast<float>(height) / 1080.0f;
    const auto margin = static_cast<std::int64_t>(120.0f * unit);
    const auto barY = static_cast<std::int64_t>(static_cast<float>(height) - 150.0f * unit);
    const auto barHeight = std::max<std::int64_t>(2, static_cast<std::int64_t>(4.0f * unit));
    const auto barWidth = static_cast<std::int64_t>(width) - 2 * margin;
    const double fraction = progress.total != 0 ? static_cast<double>(progress.finished) / static_cast<double>(progress.total) : 0.0;
    const auto percent = static_cast<unsigned>(std::min(100.0, fraction * 100.0));
    canvas.fill(margin, barY, barWidth, barHeight, 64);
    const auto* font = UiFont();
    const float title = 38.0f * unit;
    const float detail = 24.0f * unit;
    const float aboveBar = static_cast<float>(barY) - 22.0f * unit;
    if (!progress.blocked) {
        const double phase = std::fmod(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() / 1.6, 1.0);
        const auto segment = barWidth / 5;
        const auto start = static_cast<std::int64_t>(phase * static_cast<double>(barWidth + segment)) - segment;
        const auto left = std::max<std::int64_t>(start, 0);
        const auto right = std::min<std::int64_t>(start + segment, barWidth);
        if (right > left) canvas.fill(margin + left, barY, right - left, barHeight, 235);
        if (font != nullptr) DrawText(canvas, *font, "Loading", static_cast<float>(margin), aboveBar, title, 235);
        return;
    }
    canvas.fill(margin, barY, static_cast<std::int64_t>(static_cast<double>(barWidth) * std::min(1.0, fraction)), barHeight, 235);
    if (font == nullptr) return;
    DrawText(canvas, *font, progress.compiling ? "Compiling shaders" : "Loading shaders", static_cast<float>(margin), aboveBar, title, 235);
    const std::string share = std::to_string(percent) + "%";
    DrawText(canvas, *font, share, static_cast<float>(margin + barWidth) - TextWidth(*font, share, title), aboveBar, title, 235);
    char counts[96];
    std::snprintf(counts, sizeof(counts), "%llu of %llu", static_cast<unsigned long long>(progress.finished), static_cast<unsigned long long>(progress.total));
    const float belowBar = static_cast<float>(barY + barHeight) + 44.0f * unit;
    DrawText(canvas, *font, "Preparing graphics for your hardware", static_cast<float>(margin), belowBar, detail, 150);
    DrawText(canvas, *font, counts, static_cast<float>(margin + barWidth) - TextWidth(*font, counts, detail), belowBar, detail, 150);
}

}
