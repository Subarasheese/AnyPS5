#include "prx/libkernel/Pthread/include/Pthread.hpp"
#include "prx/libkernel/Pthread/include/Mutex.hpp"
#include "prx/libkernel/Pthread/include/Cond.hpp"
#include "prx/libkernel/Pthread/include/Cancel.hpp"
#include "prx/libkernel/Pthread/Posix/Common.hpp"
#include "prx/libkernel/KernelErrors.hpp"
#include "prx/libkernel/Time/include/Time.hpp"
#include "prx/libkernel/Time/include/TimedWait.hpp"
#include <atomic>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <map>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <cstring>
#ifdef __linux__
#include <csignal>
#include <dirent.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <cerrno>
#include <set>
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <unistd.h>
#endif

namespace {

constexpr int sceTimedOut = static_cast<int>(0x8002003cu);
std::mutex condInitializationMutex;

PthreadCond destroyedCond() {
    return reinterpret_cast<PthreadCond>(std::uintptr_t{2});
}

PthreadMutex destroyedMutex() {
    return reinterpret_cast<PthreadMutex>(std::uintptr_t{2});
}

PthreadMutex adaptiveInitializer() {
    return reinterpret_cast<PthreadMutex>(std::uintptr_t{1});
}

PthreadCond resolveCond(PthreadCond* cond) {
    if (!cond)
        throw std::invalid_argument("Condition variable pointer is null");
    std::atomic_ref<PthreadCond> slot(*cond);
    if (const auto current = slot.load(std::memory_order_acquire); current && current != destroyedCond())
        return current;
    std::lock_guard lock(condInitializationMutex);
    const auto current = slot.load(std::memory_order_acquire);
    if (current == destroyedCond())
        throw std::runtime_error("Condition variable has been destroyed");
    if (current)
        return current;
    auto* created = new PthreadCondPrivate();
    slot.store(created, std::memory_order_release);
    return created;
}

int mutexOwnership(PthreadMutex* mutex, PthreadMutex* owned) {
    if (!mutex)
        throw std::invalid_argument("Mutex pointer is null");
    const auto current = std::atomic_ref<PthreadMutex>(*mutex).load(std::memory_order_acquire);
    if (current == destroyedMutex())
        return SCE_KERNEL_ERROR_EINVAL;
    if (!current || current == adaptiveInitializer() || current->_owner.load(std::memory_order_acquire) != std::this_thread::get_id())
        return SCE_KERNEL_ERROR_EPERM;
    *owned = current;
    return 0;
}

template <class TNative>
struct WakingLock {
    PthreadMutex mutex;
    TNative& native;
    void lock() { native.lock(); }
    void unlock() {
        native.unlock();
#ifdef _WIN32
        TimedWait::Unpark(mutex);
#endif
    }
};

// Debug (APS5_WAIT_REPORT=1): untimed waits still blocked after 10 s are reported once each (name, condition, caller)
// by a watchdog thread; the waits themselves are unchanged.
struct WaitRecord { const PthreadCondPrivate* cond; const void* caller; std::chrono::steady_clock::time_point start; bool reported = false; std::uintptr_t chain[6] = {}; };
std::mutex g_waitRecordsMutex;
std::map<std::thread::id, WaitRecord> g_waitRecords;
const bool g_waitReport = [] {
    if (!std::getenv("APS5_WAIT_REPORT")) return false;
    std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            std::lock_guard lock(g_waitRecordsMutex);
            const auto now = std::chrono::steady_clock::now();
            for (auto& [thread, w] : g_waitRecords)
                if (!w.reported && now - w.start > std::chrono::seconds(10)) {
                    w.reported = true;
                    std::fprintf(stderr, "[wait-report] cond '%s' %p blocked > 10 s, caller %p; frames %lx %lx %lx %lx %lx %lx\n", w.cond->name.c_str(), static_cast<const void*>(w.cond), w.caller,
                                 static_cast<unsigned long>(w.chain[0]), static_cast<unsigned long>(w.chain[1]), static_cast<unsigned long>(w.chain[2]),
                                 static_cast<unsigned long>(w.chain[3]), static_cast<unsigned long>(w.chain[4]), static_cast<unsigned long>(w.chain[5]));
                }
        }
    }).detach();
    return true;
}();
#ifdef __linux__
// Debug (APS5_THREAD_SAMPLE=<file>): while <file> exists the watchdog samples every thread 20 times (50 ms apart)
// with a signal and prints, per thread, the distinct pc + frame-pointer chains it was found in; the file is then
// removed. For busy-loop freezes the wait report cannot see.
constexpr int SampleFrames = 6;
struct Sample { std::atomic<bool> done; std::uintptr_t frames[SampleFrames]; };
Sample* g_sample = nullptr;
std::uintptr_t readWord(std::uintptr_t address) {
    std::uintptr_t value = 0;
    iovec local{&value, sizeof value}, remote{reinterpret_cast<void*>(address), sizeof value};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == sizeof value ? value : 0;
}
void sampleHandler(int, siginfo_t*, void* context) {
    auto* sample = g_sample;
    if (!sample) return;
    const auto& registers = static_cast<ucontext_t*>(context)->uc_mcontext.gregs;
    sample->frames[0] = static_cast<std::uintptr_t>(registers[REG_RIP]);
    std::uintptr_t frame = static_cast<std::uintptr_t>(registers[REG_RBP]);
    for (int i = 1; i < SampleFrames; ++i) {
        sample->frames[i] = frame ? readWord(frame + 8) : 0;
        frame = frame ? readWord(frame) : 0;
    }
    sample->done.store(true, std::memory_order_release);
}
void sampleThreads() {
    static std::mutex sampling;
    std::lock_guard lock(sampling);
    struct sigaction action{};
    action.sa_sigaction = sampleHandler;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGRTMIN + 7, &action, nullptr);
    std::map<int, std::map<std::string, int>> chains;
    for (int round = 0; round < 20; ++round) {
        DIR* tasks = opendir("/proc/self/task");
        while (dirent* entry = tasks ? readdir(tasks) : nullptr) {
            const int tid = std::atoi(entry->d_name);
            if (tid <= 0 || tid == static_cast<int>(syscall(SYS_gettid))) continue;
            Sample sample{};
            g_sample = &sample;
            if (syscall(SYS_tgkill, getpid(), tid, SIGRTMIN + 7) == 0)
                for (int spin = 0; spin < 20000 && !sample.done.load(std::memory_order_acquire); ++spin) std::this_thread::yield();
            g_sample = nullptr;
            if (!sample.done.load(std::memory_order_acquire)) continue;
            std::string chain;
            for (auto f : sample.frames) { char text[24]; std::snprintf(text, sizeof text, " %lx", static_cast<unsigned long>(f)); chain += text; }
            ++chains[tid][chain];
        }
        if (tasks) closedir(tasks);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    for (auto& [tid, counts] : chains) {
        char path[64], name[32] = "";
        std::snprintf(path, sizeof path, "/proc/self/task/%d/comm", tid);
        if (FILE* comm = std::fopen(path, "r")) { if (std::fgets(name, sizeof name, comm)) name[std::strcspn(name, "\n")] = 0; std::fclose(comm); }
        for (auto& [chain, count] : counts)
            std::fprintf(stderr, "[thread-sample] %d '%s' x%d:%s\n", tid, name, count, chain.c_str());
    }
}
const bool g_threadSample = [] {
    const char* trigger = std::getenv("APS5_THREAD_SAMPLE");
    if (!trigger) return false;
    std::thread([path = std::string(trigger)] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (access(path.c_str(), F_OK) != 0) continue;
            sampleThreads();
            unlink(path.c_str());
        }
    }).detach();
    return true;
}();
// Debug (APS5_WATCH_WRITE=<hex address>): a hardware write breakpoint (8 bytes) on every thread; each hit prints the
// thread, the pc after the write + frame-pointer chain and the new value (first 64 hits).
void watchHandler(int, siginfo_t* info, void* context) {
    static std::atomic<int> hits{0};
    if (info->si_code != TRAP_PERF || hits.fetch_add(1) >= 64) return;
    const auto& registers = static_cast<ucontext_t*>(context)->uc_mcontext.gregs;
    std::uintptr_t frame = static_cast<std::uintptr_t>(registers[REG_RBP]);
    char text[256];
    int n = std::snprintf(text, sizeof text, "[watch-write] tid %ld value %016lx pc %lx:", static_cast<long>(syscall(SYS_gettid)),
                          static_cast<unsigned long>(readWord(reinterpret_cast<std::uintptr_t>(info->si_addr))), static_cast<unsigned long>(registers[REG_RIP]));
    for (int i = 0; i < 5 && frame && n < 220; ++i) {
        n += std::snprintf(text + n, sizeof text - n, " %lx", static_cast<unsigned long>(readWord(frame + 8)));
        frame = readWord(frame);
    }
    text[n++] = '\n';
    write(2, text, n);
}
const bool g_watchWrite = [] {
    const char* address = std::getenv("APS5_WATCH_WRITE");
    if (!address) return false;
    const std::uint64_t at = std::strtoull(address, nullptr, 16);
    struct sigaction action{};
    action.sa_sigaction = watchHandler;
    action.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGTRAP, &action, nullptr);
    std::thread([at] {
        std::set<int> watched;
        for (;;) {
            DIR* tasks = opendir("/proc/self/task");
            while (dirent* entry = tasks ? readdir(tasks) : nullptr) {
                const int tid = std::atoi(entry->d_name);
                if (tid <= 0 || !watched.insert(tid).second) continue;
                perf_event_attr attr{};
                attr.type = PERF_TYPE_BREAKPOINT;
                attr.size = sizeof attr;
                attr.bp_type = HW_BREAKPOINT_W;
                attr.bp_addr = at;
                attr.bp_len = HW_BREAKPOINT_LEN_8;
                attr.sample_period = 1;
                attr.exclude_kernel = 1;
                attr.exclude_hv = 1;
                attr.sigtrap = 1;
                attr.sig_data = at;
                attr.remove_on_exec = 1;
                if (syscall(SYS_perf_event_open, &attr, tid, -1, -1, PERF_FLAG_FD_CLOEXEC) < 0)
                    std::fprintf(stderr, "[watch-write] tid %d: perf_event_open failed (%s)\n", tid, std::strerror(errno));
            }
            if (tasks) closedir(tasks);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }).detach();
    std::fprintf(stderr, "[watch-write] watching %#llx\n", static_cast<unsigned long long>(at));
    return true;
}();
#endif
struct WaitRegistration {
    bool on;
    explicit WaitRegistration(bool registering, const PthreadCondPrivate* cond, const void* caller) : on(registering) {
        if (!on) return;
        WaitRecord record{cond, caller, std::chrono::steady_clock::now()};
#ifdef __linux__
        // The waiting code's return addresses (frame pointers; the guest keeps rbp frames), read safely.
        std::uintptr_t frame = reinterpret_cast<std::uintptr_t>(__builtin_frame_address(0));
        for (auto& address : record.chain) {
            if (!frame) break;
            address = readWord(frame + 8);
            frame = readWord(frame);
        }
#endif
        std::lock_guard lock(g_waitRecordsMutex);
        g_waitRecords[std::this_thread::get_id()] = record;
    }
    ~WaitRegistration() {
        if (!on) return;
        std::lock_guard lock(g_waitRecordsMutex);
        g_waitRecords.erase(std::this_thread::get_id());
    }
};

int waitUntil(PthreadCond* cond, PthreadMutex* mutex, std::optional<std::uint64_t> deadlineNanos, const void* caller) {
    auto* c = resolveCond(cond);
    const WaitRegistration registration(g_waitReport && !deadlineNanos, c, caller);
    PthreadMutex m = nullptr;
    if (const int error = mutexOwnership(mutex, &m))
        return error;
    bool timedOut = false;
    const auto waitStart = std::chrono::steady_clock::now();
    struct Trace {
        const void* caller; const bool& timedOut; std::chrono::steady_clock::time_point start;
        ~Trace() { KernelTraceWait_nid_postfix("cond", caller, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), timedOut); }
    } trace{caller, timedOut, waitStart};
    if (m->_type == MutexType::Recursive) {
        WakingLock<std::recursive_timed_mutex> waking{m, m->_rmtx};
        std::unique_lock lock(waking, std::adopt_lock);
        const auto previousCount = m->_count;
        m->_count = 0;
        m->_owner.store(std::thread::id{}, std::memory_order_release);
        for (int level = 1; level < previousCount; ++level)
            m->_rmtx.unlock();
        timedOut = !ThreadCancel::Wait(c->_cv, lock, deadlineNanos);
        for (int level = 1; level < previousCount; ++level)
            m->_rmtx.lock();
        m->_owner.store(std::this_thread::get_id(), std::memory_order_release);
        m->_count = previousCount;
        lock.release();
        ThreadCancel::Check();
        return timedOut ? sceTimedOut : 0;
    }
    WakingLock<std::timed_mutex> waking{m, m->_mtx};
    std::unique_lock lock(waking, std::adopt_lock);
    m->_owner.store(std::thread::id{}, std::memory_order_release);
    timedOut = !ThreadCancel::Wait(c->_cv, lock, deadlineNanos);
    m->_owner.store(std::this_thread::get_id(), std::memory_order_release);
    lock.release();
    ThreadCancel::Check();
    return timedOut ? sceTimedOut : 0;
}

}

int CondOperations::AbsoluteTimedwait(PthreadCond* cond, PthreadMutex* mutex, const KernelTimespec* abstime) {
    KernelUseconds usec = 0;
    if (!PosixThread::RelativeMicroseconds(resolveCond(cond)->_clockid, abstime, &usec))
        throw std::invalid_argument("Invalid absolute condition variable timeout");
    return waitUntil(cond, mutex, TimedWait::DeadlineNanos(usec), __builtin_return_address(0));
}

extern "C" {

int APS5_VABI scePthreadCondattrInit(PthreadCondattr* attr) {
    if (!attr)
        throw std::invalid_argument("Condition attribute pointer is null");
    *attr = new PthreadCondattrPrivate{0};
    return 0;
}

int APS5_VABI scePthreadCondattrDestroy(PthreadCondattr* attr) {
    if (!attr || !*attr)
        throw std::invalid_argument("Condition attributes are not initialized");
    delete *attr;
    *attr = nullptr;
    return 0;
}

int APS5_VABI scePthreadCondattrSetclock(PthreadCondattr* attr, KernelClockid clockId) {
    if (!attr || !*attr)
        throw std::invalid_argument("Condition attributes are not initialized");
    (*attr)->_clockid = static_cast<int>(clockId);
    return 0;
}

int APS5_VABI scePthreadCondInit(PthreadCond* cond, const PthreadCondattr* attr, const char* name) {
    if (!cond)
        throw std::invalid_argument("Condition variable pointer is null");
    if (attr && !*attr)
        throw std::invalid_argument("Condition attributes are not initialized");
    auto replacement = std::make_unique<PthreadCondPrivate>();
    if (attr)
        replacement->_clockid = (*attr)->_clockid;
    if (name) replacement->name = name;
    std::lock_guard lock(condInitializationMutex);
    std::atomic_ref<PthreadCond>(*cond).store(replacement.release(), std::memory_order_release);
    return 0;
}

int APS5_VABI scePthreadCondDestroy(PthreadCond* cond) {
    if (!cond)
        throw std::invalid_argument("Condition variable pointer is null");
    std::lock_guard lock(condInitializationMutex);
    if (*cond == destroyedCond())
        throw std::runtime_error("Condition variable has already been destroyed");
    delete *cond;
    std::atomic_ref<PthreadCond>(*cond).store(destroyedCond(), std::memory_order_release);
    return 0;
}

int APS5_VABI scePthreadCondSignal(PthreadCond* cond) {
    resolveCond(cond)->_cv.NotifyOne();
    return 0;
}

int APS5_VABI scePthreadCondBroadcast(PthreadCond* cond) {
    resolveCond(cond)->_cv.NotifyAll();
    return 0;
}

int APS5_VABI scePthreadCondSignalto(PthreadCond* cond, Pthread thread) {
    (void)thread;
    resolveCond(cond)->_cv.NotifyAll();
    return 0;
}

int APS5_VABI scePthreadCondWait(PthreadCond* cond, PthreadMutex* mutex) {
    return waitUntil(cond, mutex, std::nullopt, __builtin_return_address(0));
}

int APS5_VABI scePthreadCondTimedwait(PthreadCond* cond, PthreadMutex* mutex, KernelUseconds usec) {
    return waitUntil(cond, mutex, TimedWait::DeadlineNanos(usec), __builtin_return_address(0));
}

}
