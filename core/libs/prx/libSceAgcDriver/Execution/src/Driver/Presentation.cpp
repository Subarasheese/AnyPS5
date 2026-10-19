#include "prx/libSceAgcDriver/Native/include/NativeDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "prx/libSceAgcDriver/Execution/include/TracyZone.hpp"
#include "prx/libSceAgcDriver/Native/include/NativeBackend.hpp"
#include "prx/libSceAgcDriver/Native/include/CompileStatus.hpp"
#include <thread>
#include <cstdlib>

namespace AgcDriver::DriverDetail {

std::uint32_t Driver::ReserveEopFlip(const FlipInfo& info) {
    std::lock_guard lock(mutex);
    rethrowFailure();
    checkStopping();
    require(nextEopToken <= EopFlipTokenLimit, "EOP flip token space exhausted");
    const auto output = outputs.find(info.handle);
    require(output != outputs.end(), "EOP flip references an unregistered video output");
    auto request = output->second->TryReserve(info);
    if (!request) return 0;
    const auto token = nextEopToken;
    eopFlips.emplace(token, EopFlip{info.handle, std::move(request)});
    ++nextEopToken;
    return token;
}

void Driver::RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    require(output != nullptr, "null video output");
    std::lock_guard lock(mutex);
    rethrowFailure();
    checkStopping();
    require(outputs.emplace(handle, output).second, "video output already registered");
}

void Driver::UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    std::lock_guard lock(mutex);
    const auto it = outputs.find(handle);
    require(it != outputs.end() && it->second == output, "video output registration mismatch");
    std::erase_if(eopFlips, [handle](const auto& entry) { return entry.second.handle == handle; });
    outputs.erase(it);
}

void Driver::AttachWindow(const PresentationWindow& window) {
    CheckFailure();
    std::unique_lock replacing(deviceReplacement);
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
    std::lock_guard lock(GuestMemory::GpuMutex());
    if (device) {
        require(device->Window() == nullptr, "a window is already attached to the device");
        device->PrepareForReplacement();
        replacedDevices.push_back(device);
    }
    device = std::make_shared<VulkanDevice>(&window);
    // The native renderer runs on this device.
    if (Native::GpuNative()) Native::NativeDevice::Create(device.Load()).ReplayPipelineJournal();
}

void Driver::Present(const PresentationWindow& window, const DisplayBuffer* buffer, bool opaque, void (*gpuReady)(void*), void* context) {
    if (Native::GpuNative()) Native::NotePresent();
    if (PerformanceTrace::Configured()) PerformanceTrace::Present(window.width, window.height, device.Load().get());
    PerformanceContext timingContext(window.timing.get());
    PerformanceTimer timing("Driver.Present");
    static thread_local bool pinned = false;
    if (!pinned) {
        pinned = true;
        PinWorkerThread("presenter");
        GuestMemory::MarkPresenterThread();
    }
    CheckFailure();
    require(gpuReady != nullptr && context != nullptr, "missing GPU completion callback");
    require(window.getDrawableSize != nullptr, "missing window drawable size query");
    std::unique_lock presentation(presentationMutex);
    if (Native::GpuNative()) std::call_once(compileStatusStarted, [this] { std::thread([this] { runCompileStatus(); }).detach(); });

    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr;
    static const std::size_t inFlight = VulkanDevice::FlipInFlight();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::shared_ptr<VulkanDevice> presenting;
    timing.Mark("validate");
    try {
        bool submitted = false;
        bool presentable = false;
        bool trailing = false;
        double waitedMs = 0;
        {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            timing.Mark("gpu_mutex_wait");
            require(device != nullptr && device->Window() == window.context, "presentation window is not attached to the device");
            presenting = device;
            timing.Mark("device_setup");
            std::uint32_t drawableWidth = 0;
            std::uint32_t drawableHeight = 0;
            if (window.drawableSizeKnown) {
                drawableWidth = window.drawableWidth;
                drawableHeight = window.drawableHeight;
            } else {
                window.getDrawableSize(window.context, &drawableWidth, &drawableHeight);
            }
            presenting->Resize(drawableWidth, drawableHeight);
            statusWindow = window;
            statusWindow->drawableSizeKnown = true;
            statusWindow->drawableWidth = drawableWidth;
            statusWindow->drawableHeight = drawableHeight;
            presenting->SelectDynamicRange(buffer);
            timing.Mark("resize");
            presentable = presenting->Presentable();
            if (buffer != nullptr) require(buffer->width == window.width && buffer->height == window.height, "display buffer extent differs from output");
        }

        if (presentable && !syncFlip) {
            if (inFlight != 0) {

                Native::TimelineEvent("present retire wait");
                waitedMs = presenting->RetirePresents(presenting->PresentWaitsForSlots(buffer) ? 0 : inFlight);
                Native::TimelineEvent("present retired");
                timing.Mark("inflight_wait");
            }
            presentable = presenting->AcquireImage();
            Native::TimelineEvent("present acquired");
            timing.Mark("acquire_image");
        }
        if (presentable) {
            // Native renderer: the display buffer is the image GT7 last rendered at its address
            // (looked up before the GPU mutex: the native batch takes it inside its own lock).
            VkImage nativeImage = VK_NULL_HANDLE;
            VkExtent2D nativeExtent{};
            VkFormat nativeFormat = VK_FORMAT_UNDEFINED;
            const bool nativeFrame = Native::GpuNative() && buffer != nullptr && Native::NativeDevice::Get() != nullptr && Native::NativeDevice::Get()->DisplayImage(buffer->address, nativeImage, nativeExtent, nativeFormat);
            static std::uint64_t nativePresents = 0;
            if (Native::GpuNative() && nativePresents++ % 300 == 0 && Native::Logs()) std::fprintf(stderr, "[native] present %llu: display buffer 0x%llx %ux%u format 0x%llx tiling %u: %s\n", static_cast<unsigned long long>(nativePresents), buffer != nullptr ? static_cast<unsigned long long>(buffer->address) : 0ull, buffer != nullptr ? buffer->width : 0u, buffer != nullptr ? buffer->height : 0u, buffer != nullptr ? static_cast<unsigned long long>(buffer->pixelFormat) : 0ull, buffer != nullptr ? buffer->tilingMode : 0u, nativeFrame ? "native image" : "no native image");
            // Debug aid: APS5_PRESENT_DUMP_AT=<seconds after the first present> saves the display
            // buffer's guest memory once (present-dump-<address>.bin in APS5_NATIVE_DUMP_DIR or .).
            {
                static const double dumpAt = [] { const char* text = std::getenv("APS5_PRESENT_DUMP_AT"); return text != nullptr ? std::atof(text) : -1.0; }();
                static const auto firstPresent = std::chrono::steady_clock::now();
                static bool dumped = false;
                if (dumpAt >= 0 && !dumped && buffer != nullptr && std::chrono::duration<double>(std::chrono::steady_clock::now() - firstPresent).count() >= dumpAt) {
                    dumped = true;
                    const char* directory = std::getenv("APS5_NATIVE_DUMP_DIR");
                    char name[64];
                    std::snprintf(name, sizeof(name), "present-dump-%llx.bin", static_cast<unsigned long long>(buffer->address));
                    const auto path = std::filesystem::path(directory != nullptr ? directory : ".") / name;
                    std::filesystem::create_directories(path.parent_path());
                    const auto bytes = static_cast<std::size_t>(buffer->width) * buffer->height * 4u;
                    if (GuestMemory::Accessible(reinterpret_cast<const void*>(buffer->address), bytes)) std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(buffer->address), static_cast<std::streamsize>(bytes));
                    std::fprintf(stderr, "[native] present dump: display buffer 0x%llx %ux%u tiling %u, %s, guest bytes %s\n", static_cast<unsigned long long>(buffer->address), buffer->width, buffer->height, buffer->tilingMode, nativeFrame ? "native image" : "no native image", path.string().c_str());
                }
            }
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            timing.Mark("gpu_mutex_wait");
            if (Native::GpuNative()) {
                submitted = nativeFrame ? presenting->PresentNativeImage(nativeImage, std::min(buffer->width, nativeExtent.width), std::min(buffer->height, nativeExtent.height)) : presenting->PresentClear(window.width, window.height, opaque);
            } else if (buffer != nullptr) {
                if (syncFlip) {
                    presenting->WaitIdle();
                    timing.Mark("device_idle_wait");
                }
                submitted = presenting->PresentDisplayBuffer(*buffer);
                timing.Mark("present_display_buffer");
            } else {
                submitted = presenting->PresentClear(window.width, window.height, opaque);
                timing.Mark("present_clear");
            }
            if (submitted && syncFlip) {
                waitedMs = presenting->FinishPresent();
                timing.Mark("render_fence_wait");
            }
            if (submitted && (syncFlip || inFlight != 0)) {

                presenting->QueuePresent();
                timing.Mark("queue_present");
                trailing = !syncFlip;
                submitted = false;
            }
        }
        if (trailing) {
            waitedMs += presenting->FinishPresent();
            timing.Mark("render_fence_wait");
        }
        if (submitted) {
            waitedMs = presenting->FinishPresent();
            timing.Mark("render_fence_wait");

            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            presenting->QueuePresent();
            timing.Mark("queue_present");
        }
        gpuReady(context);
        lastGamePresentNs.store(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(), std::memory_order_release);
        timing.Mark("release_and_callback");
        TracyZone::MarkFrame();
        if (profile) reportPresents(waitedMs, inFlight);
        CheckFailure();
    } catch (const ProcessShutdown&) {
        throw;
    } catch (...) {
        ReportFailure(std::current_exception());
        throw;
    }
}

void Driver::reportPresents(double waitedMs, std::size_t inFlight) {
    static std::vector<double> waits;
    static VulkanDevice::PresentStatistics previous{};
    static std::uint64_t previousFlips = 0, windowSerial = 0, previousUnsignaled = 0;
    static auto lastReport = std::chrono::steady_clock::now();
    waits.push_back(waitedMs);
    if (waits.size() == 1) windowSerial = flipSerial.load();
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::sort(waits.begin(), waits.end());
    const auto percentile = [&](double fraction) { return waits.empty() ? 0.0 : waits[std::min(waits.size() - 1, static_cast<std::size_t>(fraction * static_cast<double>(waits.size())))]; };
    const auto counts = VulkanDevice::PresentCounts();
    const auto presents = counts.presents - previous.presents;
    const double perPresent = presents != 0 ? 1.0 / static_cast<double>(presents) : 0.0;
    const auto dropped = counts.acquireDrops - previous.acquireDrops;
    const double perFrame = presents + dropped != 0 ? 1.0 / static_cast<double>(presents + dropped) : 0.0;
    const auto flips = flipsCounted.load() - previousFlips;
    const double perFlip = flips != 0 ? 1.0 / static_cast<double>(flips) : 0.0;
    const auto serial = flipSerial.load();
    AgcDriver::ProfilePrint_nid_no_patch( "[present] %llu presents over 10 s (%llu dropped: no free swapchain image; %zu may trail on the GPU); %s p50 %.2f ms, p90 %.2f ms, max %.2f ms over %zu; per frame, presented or dropped: GPU busy %.1f ms, idle gaps %.1f ms (%llu batches without a completion record); per flip: %.1f batches recorded (%.1f unsignaled at the flip); per present: %.1f batches ahead of the blit, %.1f after it; after the flip packet: last submit %.1f ms, blit submit %.1f ms; in-place reads overwritten before execution: %llu of %llu checked\n", static_cast<unsigned long long>(presents), static_cast<unsigned long long>(dropped), inFlight, inFlight != 0 ? "inflight_wait" : "render_fence_wait", percentile(0.5), percentile(0.9), waits.empty() ? 0.0 : waits.back(), waits.size(), (counts.gpuBusyMs - previous.gpuBusyMs) * perFrame, (counts.gpuGapMs - previous.gpuGapMs) * perFrame, static_cast<unsigned long long>(counts.gpuUnread - previous.gpuUnread), static_cast<double>(serial - windowSerial) * perFlip, static_cast<double>(flipBatchesUnsignaled.load() - previousUnsignaled) * perFlip, static_cast<double>(counts.batchesAheadOfBlit - previous.batchesAheadOfBlit) * perPresent, static_cast<double>(counts.batchesAfterFlip - previous.batchesAfterFlip) * perPresent, (counts.lastSubmitAfterFlipMs - previous.lastSubmitAfterFlipMs) * perPresent, (counts.blitSubmitAfterFlipMs - previous.blitSubmitAfterFlipMs) * perPresent, static_cast<unsigned long long>(counts.readsOverwritten - previous.readsOverwritten), static_cast<unsigned long long>(counts.readsChecked - previous.readsChecked));
    previous = counts;
    previousFlips = flipsCounted.load();
    previousUnsignaled = flipBatchesUnsignaled.load();
    waits.clear();
}

void Driver::runCompileStatus() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        if (stopping.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire)) return;
        const auto progress = Native::CurrentCompileProgress();
        const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        const auto last = lastGamePresentNs.load(std::memory_order_acquire);
        const auto quiet = now - last;
        const bool shaders = progress.blocked && progress.blockedSeconds >= 0.5 && quiet >= 500'000'000;
        const bool stalled = last != 0 && quiet >= 1'000'000'000;
        if (!shaders && !stalled) continue;
        try {
            presentCompileStatus();
        } catch (...) {
            return;
        }
    }
}

void Driver::presentCompileStatus() {
    constexpr std::uint32_t width = 1920;
    constexpr std::uint32_t height = 1080;
    std::lock_guard presentation(presentationMutex);
    if (!statusWindow.has_value()) return;
    const auto presenting = device.Load();
    if (presenting == nullptr) return;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
        std::lock_guard lock(GuestMemory::GpuMutex());
        if (presenting->Window() != statusWindow->context) return;
        presenting->Resize(statusWindow->drawableWidth, statusWindow->drawableHeight);
        if (!presenting->Presentable()) return;
    }
    static std::vector<std::byte> pixels;
    Native::RenderCompileStatus(width, height, Native::CurrentCompileProgress(), pixels);
    presenting->RetirePresents(0);
    if (!presenting->AcquireImage()) return;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
    std::lock_guard lock(GuestMemory::GpuMutex());
    presenting->PresentPixels(width, height, pixels);
}

void Driver::ReleaseWindow(void* window) {
    std::lock_guard presentation(presentationMutex);
    if (statusWindow.has_value() && statusWindow->context == window) statusWindow.reset();
    std::lock_guard lock(GuestMemory::GpuMutex());
    if (device && device->Window() == window) device.Reset();
}

}
