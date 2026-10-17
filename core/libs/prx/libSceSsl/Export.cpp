#include <cstdint>
#include <cstddef>
#include <cstring>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

// No network is emulated: contexts, templates and requests can be created, but any request
// that would touch the network fails with the library's network error.
static constexpr int ERROR_NETWORK = static_cast<int>(0x80435001);
static std::atomic<int> g_nextHandle{1};
static std::mutex g_poolsMutex;
static std::map<int, uint64_t> g_pools;
struct Connection { int context; int socket; };
static std::map<int, Connection> g_connections;
static std::atomic<int> g_nextConnection{1};

namespace {

constexpr int ERROR_NOT_FOUND = static_cast<int>(0x8095F004);
constexpr int ERROR_INVALID_ARG = static_cast<int>(0x8095177A);
constexpr int ERROR_INVALID_ID = static_cast<int>(0x8095F006);
constexpr int ERROR_INVALID_VALUE = static_cast<int>(0x8095F007);

struct SslData {
    char* ptr;
    size_t size;
};

struct SslMemoryPoolStats {
    size_t pool_size;
    size_t max_inuse_size;
    size_t current_inuse_size;
    int32_t reserved;
};

struct SslCaCerts {
    SslData* certs;
    size_t num;
    void* pool;
};

struct SslCaList {
    void** certs;
    int num;
};
static_assert(sizeof(SslCaList) == 16);

void RequireContext(const char* function, int sslCtxId) {
    std::lock_guard lock(g_poolsMutex);
    if (!g_pools.contains(sslCtxId)) throw std::invalid_argument(std::string(function) + ": unknown context");
}

}

extern "C" {

int APS5_VABI sceSslFreeCaCerts(int ssl_ctx_id, void* ca_certs) {
    (void)ssl_ctx_id;
    if (!ca_certs) return ERROR_INVALID_ARG;
    *static_cast<SslCaCerts*>(ca_certs) = {};
    return 0;
}

int APS5_VABI sceSslGetCaCerts(int ssl_ctx_id, void* ca_certs) {
    (void)ssl_ctx_id;
    if (!ca_certs) return ERROR_INVALID_ARG;
    *static_cast<SslCaCerts*>(ca_certs) = {};
    return ERROR_NOT_FOUND;
}

int APS5_VABI sceSslGetCaList(int sslCtxId, SslCaList* caList) {
    if (!caList) return ERROR_INVALID_ARG;
    RequireContext(__func__, sslCtxId);
    *caList = {};
    return ERROR_NOT_FOUND;
}

int APS5_VABI sceSslFreeCaList(int sslCtxId, SslCaList* caList) {
    if (!caList) return ERROR_INVALID_ARG;
    RequireContext(__func__, sslCtxId);
    if (caList->certs || caList->num) throw std::invalid_argument("sceSslFreeCaList: list not returned by sceSslGetCaList");
    return 0;
}

int APS5_VABI sceSslInit_nid_postfix(uint64_t pool_size) {
    const int id = g_nextHandle.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(g_poolsMutex);
    g_pools[id] = pool_size;
    return id;
}

int APS5_VABI sceSslTerm_nid_postfix(int ssl_ctx_id) {
    std::lock_guard lock(g_poolsMutex);
    g_pools.erase(ssl_ctx_id);
    std::erase_if(g_connections, [=](const auto& entry) { return entry.second.context == ssl_ctx_id; });
    return 0;
}

int APS5_VABI sceSslClose(int context, int connection) {
    std::lock_guard lock(g_poolsMutex);
    const auto found = g_connections.find(connection);
    // A failed offline handshake has no session to shut down; keep the handle
    // until DeleteConnection, as the game closes and then deletes separately.
    return found != g_connections.end() && found->second.context == context ? 0 : ERROR_INVALID_ID;
}

int APS5_VABI sceSslGetSerialNumber() {
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI sceSslLoadCert() {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceSslUnloadCert(int sslCtxId) {
    RequireContext(__func__, sslCtxId);
    return 0;
}

int APS5_VABI sceSslGetMemoryPoolStats(int ssl_ctx_id, SslMemoryPoolStats* stats) {
    if (stats == nullptr) return ERROR_INVALID_VALUE;
    std::lock_guard lock(g_poolsMutex);
    const auto pool = g_pools.find(ssl_ctx_id);
    if (pool == g_pools.end()) {
        std::memset(stats, 0, sizeof(*stats));
        return ERROR_INVALID_ID;
    }
    *stats = {static_cast<size_t>(pool->second), 0, 0, 0};
    return 0;
}

int APS5_VABI sceSslFreeSslCertName(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceSslGetIssuerName(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceSslGetNameEntryCount(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceSslGetNameEntryInfo(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceSslGetPem(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

int APS5_VABI sceSslGetSubjectName(void) {
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

// Preserve the runtime's existing offline-network policy: creation/configuration are
// local operations; handshake and I/O report failure, never pretend TLS succeeded.
int APS5_VABI sceSslCreateConnection(int context, int socket) {
    std::lock_guard lock(g_poolsMutex);
    if (!g_pools.contains(context)) return ERROR_INVALID_ID;
    if (socket < 0) return ERROR_INVALID_ARG;
    const int id = g_nextConnection.fetch_add(1, std::memory_order_relaxed);
    g_connections.emplace(id, Connection{context, socket});
    return id;
}

int APS5_VABI sceSslDeleteConnection(int context, int connection) {
    std::lock_guard lock(g_poolsMutex);
    const auto found = g_connections.find(connection);
    if (found == g_connections.end() || found->second.context != context) return ERROR_INVALID_ID;
    g_connections.erase(found);
    return 0;
}

int APS5_VABI sceSslEnableVerifyOption(int context, int connection, uint64_t option) {
    (void)option;
    std::lock_guard lock(g_poolsMutex);
    const auto found = g_connections.find(connection);
    return found != g_connections.end() && found->second.context == context ? 0 : ERROR_INVALID_ID;
}

int APS5_VABI sceSslConnect(int context, int connection) {
    std::lock_guard lock(g_poolsMutex);
    const auto found = g_connections.find(connection);
    return found != g_connections.end() && found->second.context == context ? ERROR_NETWORK : ERROR_INVALID_ID;
}

int APS5_VABI sceSslRead(int context, int connection, void* buffer, int size) {
    if (!buffer || size < 0) return ERROR_INVALID_ARG;
    return sceSslConnect(context, connection);
}

int APS5_VABI sceSslWrite(int context, int connection, const void* buffer, int size) {
    if (!buffer || size < 0) return ERROR_INVALID_ARG;
    return sceSslConnect(context, connection);
}

}
