#include "phoenix_gds.h"

#ifdef USE_PHOENIX

#include <glog/logging.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <unistd.h>

#include <phoenix.h>

#include "cuda_alike.h"

namespace mooncake {

namespace {
// libphoenix registration constraints.
constexpr uint64_t kRegAlign = 64 * 1024;             // addr/len alignment
constexpr uint64_t kMaxRegChunk = 32ULL * 1024 * 1024 * 1024;  // 32 GiB
}  // namespace

PhoenixCtx& PhoenixCtx::Instance() {
    static PhoenixCtx ctx;
    return ctx;
}

bool PhoenixCtx::module_present() const {
    std::call_once(module_check_flag_, [this] {
        // phxfs creates one char device per GPU: /dev/phxfs_dev<N>.
        module_present_ = (::access("/dev/phxfs_dev0", F_OK) == 0);
        if (!module_present_) {
            LOG(INFO) << "Phoenix: phxfs module not present "
                         "(/dev/phxfs_dev0 missing); direct read disabled";
        }
    });
    return module_present_;
}

bool PhoenixCtx::staging_mode() const {
    std::call_once(staging_check_flag_, [this] {
        if (!module_present()) return;
        int mode = phxfs_get_map_mode(0);
        staging_mode_ = (mode == 1);  // PHX_MAP_MODE_STAGING
        if (staging_mode_) {
            LOG(INFO) << "Phoenix: phxfs in STAGING mode; "
                         "GPU buffer registration not required";
        }
    });
    return staging_mode_;
}

bool PhoenixCtx::has_registrations(const std::vector<Slice>& slices) const {
    if (!module_present()) return false;
    if (staging_mode()) return true;
    std::shared_lock lock(mu_);
    if (regs_.empty()) return false;
    for (const auto& s : slices) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(s.ptr);
        // Binary search: find first reg with base > p, then check predecessor.
        auto it = std::lower_bound(
            regs_.begin(), regs_.end(), p,
            [](const Reg& r, uintptr_t a) { return r.base <= a; });
        uintptr_t base = 0;
        if (it == regs_.begin()) return false;  // p < first reg's base
        --it;
        if (!(p >= it->base && (p - it->base) <= it->len &&
              s.size <= it->len - (p - it->base))) {
            return false;
        }
        base = it->base;
        if (!Aligned(p - base, 0, s.size)) return false;
    }
    return true;
}

int PhoenixCtx::OpenDevice(int device_id) {
    auto it = dev_by_device_.find(device_id);
    if (it != dev_by_device_.end()) return it->second;

    int dev = phxfs_find_dev(device_id);
    if (dev < 0) {
        LOG(ERROR) << "Phoenix: phxfs_find_dev(" << device_id
                   << ") failed: " << dev;
        return dev;
    }
    int rc = phxfs_open(dev);
    if (rc != 0) {
        LOG(ERROR) << "Phoenix: phxfs_open(dev " << dev
                   << ") failed: " << rc;
        return rc;
    }
    dev_by_device_.emplace(device_id, dev);
    LOG(INFO) << "Phoenix: opened phxfs dev " << dev << " for accel device "
              << device_id;
    return dev;
}

int PhoenixCtx::ResolveDevice(uintptr_t addr, size_t len) {
    {
        // Fast path: [addr, addr+len) inside a range cached by an earlier
        // Register(). This is what the per-item hot path relies on — no
        // CUDA driver call. Entries are few (one per registered range), so
        // a linear scan is sufficient.
        std::shared_lock lock(mu_);
        for (const auto& e : dev_cache_) {
            if (addr >= e.base && (addr - e.base) <= e.len &&
                len <= e.len - (addr - e.base)) {
                return e.dev;
            }
        }
    }
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, reinterpret_cast<const void*>(addr))
            != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        cudaGetLastError();
        return -1;
    }
    return OpenDevice(attr.device);
}

int PhoenixCtx::Register(uintptr_t addr, size_t len) {
    if (!module_present()) return -ENODEV;
    if (staging_mode()) {
        // Staging mode: no registration needed, but open the device to
        // trigger staging pool setup on first use — and cache the extent so
        // per-item ResolveDevice() calls skip the CUDA driver call.
        const int dev = ResolveDevice(addr);
        if (dev < 0) return -ENODEV;
        std::unique_lock lock(mu_);
        bool present = false;
        for (const auto& e : dev_cache_) {
            if (e.base == addr && e.len == len) {
                present = true;
                break;
            }
        }
        if (!present) dev_cache_.push_back({addr, len, dev});
        return 0;
    }
    if (addr == 0 || (addr % kRegAlign) != 0) {
        LOG(ERROR) << "Phoenix: register addr " << std::hex << addr
                   << " not " << kRegAlign << "-byte aligned";
        return -EINVAL;
    }

    // Resolve device_id via cudaPointerGetAttributes (maps to
    // hipPointerGetAttributes on HIP, etc. via cuda_alike.h).
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, reinterpret_cast<const void*>(addr))
            != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        cudaGetLastError();
        return 0;  // not device memory — nothing to register
    }
    const int device_id = attr.device;

    len &= ~(kRegAlign - 1);  // round down; tail falls back to legacy
    if (len == 0) {
        LOG(ERROR) << "Phoenix: register len too small (< 64 KiB)";
        return -EINVAL;
    }

    std::unique_lock lock(mu_);

    // Re-registering an already tracked range: bump refcounts on every chunk
    // it covers (libphoenix refcounts the exact-duplicate regmem calls).
    // regs_ is kept sorted by base for O(log R) lookup.
    bool any_new = false;
    for (size_t off = 0; off < len; off += kMaxRegChunk) {
        const uintptr_t chunk_addr = addr + off;
        const size_t chunk_len =
            std::min<size_t>(kMaxRegChunk, len - off);

        auto it = std::lower_bound(
            regs_.begin(), regs_.end(), chunk_addr,
            [](const Reg& r, uintptr_t a) { return r.base < a; });
        if (it != regs_.end() && it->base == chunk_addr) {
            if (it->len != chunk_len) {
                LOG(ERROR) << "Phoenix: conflicting re-registration at "
                           << std::hex << chunk_addr;
                return -EINVAL;
            }
        } else {
            any_new = true;
        }
    }

    int dev = OpenDevice(device_id);
    if (dev < 0) return dev;

    for (size_t off = 0; off < len; off += kMaxRegChunk) {
        const uintptr_t chunk_addr = addr + off;
        const size_t chunk_len = std::min<size_t>(kMaxRegChunk, len - off);

        void* target = nullptr;
        int rc = phxfs_regmem(dev, reinterpret_cast<const void*>(chunk_addr),
                              chunk_len, &target);
        if (rc != 0) {
            LOG(ERROR) << "Phoenix: phxfs_regmem(dev " << dev << ", addr "
                       << std::hex << chunk_addr << ", len " << std::dec
                       << chunk_len << ") failed: " << rc;
            return rc;
        }

        auto it = std::lower_bound(
            regs_.begin(), regs_.end(), chunk_addr,
            [](const Reg& r, uintptr_t a) { return r.base < a; });
        if (it != regs_.end() && it->base == chunk_addr) {
            it->refs++;
        } else {
            regs_.insert(it, Reg{chunk_addr, chunk_len, dev, 1});
        }
    }
    if (any_new) {
        LOG(INFO) << "Phoenix: registered GPU range [" << std::hex << addr
                  << ", " << (addr + len) << ") on dev " << dev;
    }
    return 0;
}

int PhoenixCtx::Unregister(uintptr_t addr, size_t len) {
    if (staging_mode()) {
        // Drop the cached resolution for this exact range.
        std::unique_lock lock(mu_);
        for (auto it = dev_cache_.begin(); it != dev_cache_.end(); ++it) {
            if (it->base == addr && it->len == len) {
                dev_cache_.erase(it);
                break;
            }
        }
        return 0;
    }
    len &= ~(kRegAlign - 1);
    std::unique_lock lock(mu_);
    for (size_t off = 0; off < len; off += kMaxRegChunk) {
        const uintptr_t chunk_addr = addr + off;
        auto it = std::lower_bound(
            regs_.begin(), regs_.end(), chunk_addr,
            [](const Reg& r, uintptr_t a) { return r.base < a; });
        if (it == regs_.end() || it->base != chunk_addr) {
            LOG(WARNING) << "Phoenix: unregister of unknown range "
                         << std::hex << chunk_addr;
            continue;
        }
        if (--it->refs == 0) {
            phxfs_deregmem(it->dev, reinterpret_cast<const void*>(it->base),
                           it->len);
            regs_.erase(it);
        }
    }
    return 0;
}

int PhoenixCtx::FindRegistration(uintptr_t addr, size_t len,
                                 uintptr_t* reg_base) const {
    std::shared_lock lock(mu_);
    if (regs_.empty()) return -1;
    // Find the first reg with base > addr; the candidate is the one before it.
    auto it = std::lower_bound(
        regs_.begin(), regs_.end(), addr,
        [](const Reg& r, uintptr_t a) { return r.base <= a; });
    if (it != regs_.begin()) {
        --it;
        if (addr >= it->base && (addr - it->base) <= it->len &&
            len <= it->len - (addr - it->base)) {
            if (reg_base) *reg_base = it->base;
            return it->dev;
        }
    }
    return -1;
}

namespace {

/// Staging mode: no registration lookup; resolve device from the GPU buffer
/// address directly (cached per registered range after Register()). 
/// libphoenix routes reads through its staging pool.
static void BuildStagingReqs(const std::vector<DirectIoItem>& items,
                             std::vector<phxfs_io_req_t>& reqs,
                             std::vector<size_t>& idx) {
    auto& phx = PhoenixCtx::Instance();
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        const uintptr_t buf = reinterpret_cast<uintptr_t>(it.buf.ptr);
        const int dev = phx.ResolveDevice(buf, it.nbytes);
        if (dev < 0) continue;
        phxfs_io_req_t r{};
        r.fd = it.fd;
        r.device_id = dev;
        r.buf = it.buf.ptr;
        r.buf_offset = 0;
        r.nbytes = it.nbytes;
        r.f_offset = it.f_offset;
        reqs.push_back(r);
        idx.push_back(i);
    }
}

/// Filter items through PhoenixCtx (registration + alignment), build reqs.
static void BuildDirectReqs(const std::vector<DirectIoItem>& items,
                            std::vector<phxfs_io_req_t>& reqs,
                            std::vector<size_t>& idx) {
    auto& phx = PhoenixCtx::Instance();
    reqs.clear();
    idx.clear();
    reqs.reserve(items.size());
    idx.reserve(items.size());

    if (phx.staging_mode()) {
        BuildStagingReqs(items, reqs, idx);
        return;
    }

    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        const uintptr_t buf = reinterpret_cast<uintptr_t>(it.buf.ptr);
        uintptr_t base = 0;
        const int dev = phx.FindRegistration(buf, it.nbytes, &base);
        if (dev < 0) continue;
        if (!PhoenixCtx::Aligned(buf - base,
                                 static_cast<uint64_t>(it.f_offset),
                                 it.nbytes)) {
            continue;
        }
        phxfs_io_req_t r{};
        r.fd = it.fd;
        r.device_id = dev;
        r.buf = it.buf.ptr;
        r.buf_offset = 0;
        r.nbytes = it.nbytes;
        r.f_offset = it.f_offset;
        reqs.push_back(r);
        idx.push_back(i);
    }
}

static void FillDone(const std::vector<DirectIoItem>& items,
                     const std::vector<phxfs_io_req_t>& reqs,
                     const std::vector<size_t>& idx,
                     std::vector<char>& done, const char* op) {
    for (size_t k = 0; k < reqs.size(); ++k) {
        if (reqs[k].result == (ssize_t)reqs[k].nbytes) {
            done[idx[k]] = 1;
        } else {
            LOG(WARNING) << "Phoenix: short/failed " << op
                         << " for key " << items[idx[k]].key
                         << ": result=" << reqs[k].result
                         << " expect=" << reqs[k].nbytes;
        }
    }
}

}  // namespace

int PhxReadBatch(const std::vector<DirectIoItem>& items,
                 std::vector<char>& done) {
    done.assign(items.size(), 0);
    if (items.empty()) return 0;
    if (!PhoenixCtx::Instance().module_present()) return -ENODEV;

    std::vector<phxfs_io_req_t> reqs;
    std::vector<size_t> idx;
    BuildDirectReqs(items, reqs, idx);
    if (reqs.empty()) return 0;

    const int read_result = phxfs_read_batch(reqs.data(), (int)reqs.size());
    if (read_result < 0) {
        LOG(ERROR) << "Phoenix: phxfs_read_batch submission failed: "
                   << read_result;
        return read_result;
    }
    FillDone(items, reqs, idx, done, "read");
    return 0;
}

struct PhxAsyncReadBatch {
    phxfs_batch_t* h = nullptr;
    std::vector<phxfs_io_req_t> reqs;
    std::vector<size_t> idx;
    std::vector<DirectIoItem> items;  // pins alive until wait
};

PhxAsyncReadBatch* PhxAsyncReadBatchSubmit(
    const std::vector<DirectIoItem>& items) {
    if (items.empty()) return nullptr;
    if (!PhoenixCtx::Instance().module_present()) return nullptr;

    auto* batch = new PhxAsyncReadBatch();
    BuildDirectReqs(items, batch->reqs, batch->idx);
    if (batch->reqs.empty()) {
        delete batch;
        return nullptr;
    }
    batch->items = items;  // keep pins (allocation guards) alive
    batch->h = phxfs_batch_submit_read(batch->reqs.data(),
                                       (int)batch->reqs.size());
    if (batch->h == nullptr) {
        delete batch;
        return nullptr;
    }
    return batch;
}

int PhxAsyncReadBatchWait(PhxAsyncReadBatch* batch, std::vector<char>& done) {
    if (batch == nullptr) return -EINVAL;
    done.assign(batch->items.size(), 0);
    const int wait_result = phxfs_batch_wait(batch->h);
    FillDone(batch->items, batch->reqs, batch->idx, done, "read");
    delete batch;
    return wait_result < 0 ? wait_result : 0;
}

void PhxAsyncReadBatchCancel(PhxAsyncReadBatch* batch) {
    if (batch == nullptr) return;
    phxfs_batch_destroy(batch->h);
    delete batch;
}

// ---- Write batch ----

int PhxWriteBatch(const std::vector<DirectIoItem>& items,
                  std::vector<char>& done) {
    done.assign(items.size(), 0);
    if (items.empty()) return 0;
    if (!PhoenixCtx::Instance().module_present()) return -ENODEV;

    std::vector<phxfs_io_req_t> reqs;
    std::vector<size_t> idx;
    BuildDirectReqs(items, reqs, idx);
    if (reqs.empty()) return 0;

    const int write_result = phxfs_write_batch(reqs.data(), (int)reqs.size());
    if (write_result < 0) {
        LOG(ERROR) << "Phoenix: phxfs_write_batch submission failed: "
                   << write_result;
        return write_result;
    }
    FillDone(items, reqs, idx, done, "write");
    return 0;
}

struct PhxAsyncWriteBatch {
    phxfs_batch_t* h = nullptr;
    std::vector<phxfs_io_req_t> reqs;
    std::vector<size_t> idx;
    std::vector<DirectIoItem> items;
};

PhxAsyncWriteBatch* PhxAsyncWriteBatchSubmit(
    const std::vector<DirectIoItem>& items) {
    if (items.empty()) return nullptr;
    if (!PhoenixCtx::Instance().module_present()) return nullptr;

    auto* batch = new PhxAsyncWriteBatch();
    BuildDirectReqs(items, batch->reqs, batch->idx);
    if (batch->reqs.empty()) {
        delete batch;
        return nullptr;
    }
    batch->items = items;
    batch->h = phxfs_batch_submit_write(batch->reqs.data(),
                                         (int)batch->reqs.size());
    if (batch->h == nullptr) {
        delete batch;
        return nullptr;
    }
    return batch;
}

int PhxAsyncWriteBatchWait(PhxAsyncWriteBatch* batch,
                            std::vector<char>& done) {
    if (batch == nullptr) return -EINVAL;
    done.assign(batch->items.size(), 0);
    const int wait_result = phxfs_batch_wait(batch->h);
    FillDone(batch->items, batch->reqs, batch->idx, done, "write");
    delete batch;
    return wait_result < 0 ? wait_result : 0;
}

void PhxAsyncWriteBatchCancel(PhxAsyncWriteBatch* batch) {
    if (batch == nullptr) return;
    phxfs_batch_destroy(batch->h);
    delete batch;
}

}  // namespace mooncake

#endif  // USE_PHOENIX
