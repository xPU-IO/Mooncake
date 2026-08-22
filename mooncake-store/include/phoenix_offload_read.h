#pragma once

// Phoenix offload direct read (same-host, cross-process).
//
// A LOCAL_DISK replica descriptor carries the record's file location
// (record_offset + file_path, populated by the master from the offload
// metadata). When the data file is reachable from the reading process — the
// common case for co-located workers on one host — the read can bypass the
// owner-RPC staging path entirely and DMA straight from the local NVMe into
// the user's GPU buffers via Phoenix (staging mode).
//
// Everything here is additive: candidates that cannot be served are reported
// false and the caller falls back to the original endpoint-grouped RPC path.

#ifdef USE_PHOENIX

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "replica.h"
#include "storage_backend.h"

namespace mooncake {
namespace phoenix_offload {

// One direct-read candidate, prepared by the caller from a LOCAL_DISK replica.
struct DirectReadCandidate {
    const Replica::Descriptor* replica = nullptr;  // location info source
    size_t record_key_len = 0;      // scoped-key length (record layout)
    const std::vector<Slice>* slices = nullptr;  // GPU destinations
    uint64_t total_size = 0;        // expected bytes (sum of slice sizes)
    std::string local_host;         // reader's host (ip:port → host part),
                                     // empty = skip host check (single-node)
};

// Plans and executes Phoenix DMA reads for all servable candidates in a single
// phxfs batch. `served` is resized to candidates.size() and filled per entry:
//   true  — all bytes landed in slices (caller must still verify the object
//           checksum before marking success)
//   false — not servable or incomplete read; no partial state is visible to
//           the caller beyond the destination buffers, fall back as usual
void ServeDirectReads(const std::vector<DirectReadCandidate>& candidates,
                      std::vector<bool>& served);

}  // namespace phoenix_offload
}  // namespace mooncake

#endif  // USE_PHOENIX
