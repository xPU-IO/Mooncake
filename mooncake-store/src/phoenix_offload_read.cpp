// Phoenix offload direct read (same-host, cross-process). See header.

#include "phoenix_offload_read.h"

#ifdef USE_PHOENIX

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include <mutex>
#include <unordered_map>
#include <utility>

#include <glog/logging.h>

#include "phoenix_gds.h"

namespace mooncake {
namespace phoenix_offload {
namespace {

constexpr uint64_t kFileAlignment = 4096;

// path -> O_DIRECT fd. Data files are owned by co-located worker processes
// and live for the owner's lifetime; if an owner dies, the master drops its
// replicas, so no new descriptor will reference the path again.
int GetDirectFd(const std::string& path) {
    static std::mutex mu;
    static std::unordered_map<std::string, int> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        LOG(WARNING) << "Phoenix offload: open " << path
                     << " failed: " << strerror(errno);
        cache.emplace(path, -1);
        return -1;
    }
    LOG(INFO) << "Phoenix offload: direct-read fd opened for " << path;
    cache.emplace(path, fd);
    return fd;
}

// Eligibility gates mirror OffsetAllocatorStorageBackend::BatchReadPlan: a
// candidate failing any check is left to the caller's fallback path.
bool PlanOne(const DirectReadCandidate& cand, int& fd_out,
             uint64_t& value_file_offset_out) {
    if (cand.replica == nullptr || cand.slices == nullptr ||
        cand.total_size == 0) {
        return false;
    }
    if (!cand.replica->is_local_disk_replica()) return false;
    const auto& desc = cand.replica->get_local_disk_descriptor();
    if (desc.record_offset < 0 || desc.file_path.empty()) return false;
    if (desc.object_size != cand.total_size) return false;  // defensive

    // P0-2: reject cross-host replicas — the file path is local to the
    // owner's machine; on a different host it would be a different file (or
    // not exist at all), leading to wrong data. Compare the host portion
    // (ip:port → strip port) of the descriptor's endpoint vs the reader's.
    if (!cand.local_host.empty() && !desc.transport_endpoint.empty()) {
        const auto& ep = desc.transport_endpoint;
        auto colon = ep.rfind(':');
        std::string ep_host = (colon != std::string::npos)
                                  ? ep.substr(0, colon)
                                  : ep;
        if (ep_host != cand.local_host) return false;
    }

    int fd = GetDirectFd(desc.file_path);
    if (fd < 0) return false;

    const uint64_t value_offset =
        static_cast<uint64_t>(desc.record_offset) +
        OffsetAllocatorStorageBackend::RecordHeader::ValueOffsetInRecord(
            static_cast<uint32_t>(cand.record_key_len));
    if (value_offset % kFileAlignment != 0) return false;

    uint64_t cumulative = 0;
    for (const auto& s : *cand.slices) {
        if (s.size % 512 != 0) return false;
        if ((value_offset + cumulative) % kFileAlignment != 0) return false;
        cumulative += s.size;
    }
    if (cumulative != cand.total_size) return false;

    fd_out = fd;
    value_file_offset_out = value_offset;
    return true;
}

}  // namespace

void ServeDirectReads(const std::vector<DirectReadCandidate>& candidates,
                      std::vector<bool>& served) {
    served.assign(candidates.size(), false);
    if (candidates.empty()) return;

    // Guard: if the phxfs kernel module is not loaded, Phoenix DMA reads
    // cannot be served even though the descriptor carries file_path (the
    // file exists but there is no DMA engine to drive it). Fall back to the
    // legacy RPC path. This check is cached (module_present() probes once).
    if (!PhoenixCtx::Instance().module_present()) return;

    std::vector<DirectIoItem> items;
    // Per-candidate item range [begin, end) within items.
    std::vector<std::pair<size_t, size_t>> ranges(
        candidates.size(), {0, 0});

    for (size_t i = 0; i < candidates.size(); ++i) {
        int fd = -1;
        uint64_t value_offset = 0;
        // Initialize both ends before PlanOne: a failed candidate must have
        // first == second so the completion loop skips it (begin == end).
        ranges[i].first = items.size();
        ranges[i].second = items.size();
        if (!PlanOne(candidates[i], fd, value_offset)) continue;

        uint64_t cumulative = 0;
        for (const auto& s : *candidates[i].slices) {
            if (s.size == 0) continue;
            items.push_back(
                DirectIoItem{"", fd,
                             static_cast<off_t>(value_offset + cumulative),
                             s.size, s, nullptr});
            cumulative += s.size;
        }
        ranges[i].second = items.size();
    }

    if (items.empty()) return;

    std::vector<char> done(items.size(), 0);
    PhxReadBatch(items, done);

    for (size_t i = 0; i < candidates.size(); ++i) {
        const size_t begin = ranges[i].first;
        const size_t end = ranges[i].second;
        if (begin == end) continue;
        bool ok = true;
        for (size_t j = begin; j < end; ++j) ok = ok && done[j];
        if (!ok) {
            LOG(ERROR) << "Phoenix offload: direct read incomplete ("
                       << end - begin << " items) for "
                       << candidates[i]
                              .replica->get_local_disk_descriptor()
                              .file_path;
            continue;
        }
        served[i] = true;
    }
}

}  // namespace phoenix_offload
}  // namespace mooncake

#endif  // USE_PHOENIX
