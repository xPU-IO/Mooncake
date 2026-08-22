#pragma once

#ifdef USE_PHOENIX

// Phoenix GDS direct-read support (STORE_USE_PHOENIX).
//
// PhoenixCtx owns the phxfs device handles and the registry of GPU buffers
// that direct DMA reads may target. It is deliberately storage-agnostic: the
// storage backends hand us (fd, file offset) via DirectIoItem and Phoenix
// turns them into phxfs_io_req_t batch reads straight into registered GPU
// memory. No file management (layout/eviction/metadata) happens here.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "storage_backend.h"

namespace mooncake {

class PhoenixCtx {
   public:
    static PhoenixCtx& Instance();

    // True when the phxfs kernel module is present (fast guard; cached at
    // first call). Direct-read candidates still must pass
    // FindRegistration()+alignment.
    bool module_present() const;

    // True when phxfs is in STAGING mode (cached at first call). In staging
    // mode libphoenix routes reads through an internal staging pool
    // (SSD->staging->D2D->user buffer); user buffers are not registered,
    // so Register/Unregister are no-ops and has_registrations returns true.
    bool staging_mode() const;

    // True when every slice lies inside a Phoenix-registered GPU range and
    // satisfies the DMA alignment rules (4 KiB offset within the registration,
    // 512 B multiple length).  Returns false when module is absent or no
    // buffer has been registered.  In staging mode returns true immediately.
    bool has_registrations(const std::vector<Slice>& slices) const;

    // Register an accelerator buffer range for Phoenix DMA.
    //  - addr must be 64 KiB aligned (device allocations are);
    //  - len is rounded DOWN to a 64 KiB multiple — the uncovered tail falls
    //    back to the legacy path per slice;
    //  - internally chunked at <= 32 GiB per phxfs_regmem call;
    //  - device_id is resolved automatically via the accelerator registry
    //    (cudaPointerGetAttributes on CUDA, hipPointerGetAttributes on HIP,
    //    etc.). Non-device pointers are silently skipped (no-op, returns 0).
    // Re-registering the exact same range bumps a reference count (mirrors
    // libphoenix exact-duplicate refcounting); Unregister must be called the
    // same number of times.
    // Returns 0 on success (including no-op for non-device memory),
    // negative errno otherwise.
    int Register(uintptr_t addr, size_t len);
    int Unregister(uintptr_t addr, size_t len);

    // If [addr, addr+len) is fully inside a live registration, returns the
    // phxfs device id (>= 0) and (when reg_base != nullptr) sets *reg_base to
    // the registration base address. Otherwise returns -1.
    int FindRegistration(uintptr_t addr, size_t len,
                         uintptr_t* reg_base = nullptr) const;

    // O_DIRECT alignment requirement for one direct read:
    // gpu_delta (= dest - registration base) and f_offset 4 KiB aligned,
    // nbytes a 512 B multiple.
    static bool Aligned(uint64_t gpu_delta, uint64_t f_offset,
                        uint64_t nbytes) {
        return (gpu_delta % 4096 == 0) && (f_offset % 4096 == 0) &&
               (nbytes % 512 == 0);
    }

    // Resolve (and open on first use) the phxfs device for the GPU buffer
    // range [addr, addr+len) via cudaPointerGetAttributes.  The result is
    // cached per registered range, so after Register() has warmed the cache
    // the steady-state per-item lookup costs no CUDA driver call.  Returns
    // phxfs device id or -1.  Used in staging mode where no registration
    // lookup is needed.
    int ResolveDevice(uintptr_t addr, size_t len = 1);

   private:
    PhoenixCtx() = default;

    struct Reg {
        uintptr_t base;
        size_t len;  // aligned length actually registered
        int dev;     // phxfs device id
        int refs;    // Register/Unregister balance
    };

    // Resolve (and open on first use) the phxfs device for a vendor device.
    // Returns phxfs device id or negative errno.
    int OpenDevice(int device_id);

    mutable std::shared_mutex mu_;
    std::vector<Reg> regs_;
    // Staging-mode device resolution cache: [base, base+len) -> phxfs dev.
    // Populated/invalidated by Register()/Unregister() (which know the full
    // extent) and consulted by ResolveDevice(), so the per-item hot path in
    // staging mode avoids a cudaPointerGetAttributes call.  Guarded by mu_
    // alongside regs_.
    struct DevReg {
        uintptr_t base;
        size_t len;
        int dev;
    };
    std::vector<DevReg> dev_cache_;
    std::unordered_map<int, int> dev_by_device_;  // vendor device id -> phxfs dev
    mutable std::once_flag module_check_flag_;
    mutable bool module_present_ = false;
    mutable std::once_flag staging_check_flag_;
    mutable bool staging_mode_ = false;
};

// Execute one synchronous direct-read batch: filters items through PhoenixCtx
// (registration + alignment), issues a single phxfs_read_batch for the
// eligible ones, and marks per-item outcomes in done (done.size() ==
// items.size(); entries for ineligible or failed items are false).
// Returns 0 when the batch ran (inspect done for per-key results), negative
// errno on submission-level failure (nothing was read).
int PhxReadBatch(const std::vector<DirectIoItem>& items,
                 std::vector<char>& done);

// Async variant for batch pipelining (submit several batches ahead, wait in
// order). PhxAsyncReadBatch owns the request array; callers must not touch
// it between Async and Wait/Cancel. Async applies the same filtering as
// the sync path and returns nullptr on submission-level failure (including
// EBUSY when libphoenix's queue is full — wait the oldest batch and retry).
struct PhxAsyncReadBatch;
PhxAsyncReadBatch* PhxAsyncReadBatchSubmit(
    const std::vector<DirectIoItem>& items);
// Waits for completion, marks per-item outcomes in done (same contract as
// the sync path) and frees the batch. Returns 0 on success, negative errno
// on failure.
int PhxAsyncReadBatchWait(PhxAsyncReadBatch* batch, std::vector<char>& done);
// Abandons a submitted batch without reading results (frees it).
void PhxAsyncReadBatchCancel(PhxAsyncReadBatch* batch);

// ---- Write batch (GPU → SSD) ----
// Same filtering/alignment contract as the read batch; the GPU buffer is
// the source of the transfer.

int PhxWriteBatch(const std::vector<DirectIoItem>& items,
                  std::vector<char>& done);

struct PhxAsyncWriteBatch;
PhxAsyncWriteBatch* PhxAsyncWriteBatchSubmit(
    const std::vector<DirectIoItem>& items);
int PhxAsyncWriteBatchWait(PhxAsyncWriteBatch* batch,
                            std::vector<char>& done);
void PhxAsyncWriteBatchCancel(PhxAsyncWriteBatch* batch);

}  // namespace mooncake

#endif  // USE_PHOENIX
