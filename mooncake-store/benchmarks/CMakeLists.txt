# Add allocator benchmark executable
add_executable(allocator_bench allocator_bench.cpp)
target_link_libraries(allocator_bench PRIVATE cachelib_memory_allocator
                                              mooncake_store)

# Add focused OffsetAllocator concurrency benchmark
add_executable(offset_allocator_concurrency_bench
               offset_allocator_concurrency_bench.cpp)
target_link_libraries(offset_allocator_concurrency_bench
                      PRIVATE mooncake_store gflags::gflags pthread)

# Add master benchmark executable
add_executable(master_bench master_bench.cpp)
target_link_libraries(master_bench PRIVATE cachelib_memory_allocator
                                           mooncake_store)

# Add BatchGetReplicaList latency benchmark executable
add_executable(batch_get_replica_bench batch_get_replica_bench.cpp)
target_link_libraries(batch_get_replica_bench PRIVATE cachelib_memory_allocator
                                                       mooncake_store)

# Add file interface benchmark executable
add_executable(file_interface_bench file_interface_bench.cpp)
target_link_libraries(file_interface_bench PRIVATE mooncake_store glog::glog)

# Add storage backend benchmark executable
# This benchmark tests all storage backends (OffsetAllocator, Bucket, FilePerKey)
# with realistic KV cache workloads for LLM inference scenarios
add_executable(storage_backend_bench storage_backend_bench.cpp)
target_link_libraries(
  storage_backend_bench PRIVATE mooncake_store cachelib_memory_allocator
                                gflags::gflags glog::glog pthread)

# Add allocation strategy benchmark executable
# This benchmark tests AllocationStrategy performance with configurable
# segment counts, allocation sizes, replica counts, and workload patterns
add_executable(allocation_strategy_bench allocation_strategy_bench.cpp)
target_link_libraries(
  allocation_strategy_bench PRIVATE mooncake_store cachelib_memory_allocator
                                    gflags::gflags glog::glog pthread)

# Add NoF worker pool benchmark executable
if(USE_NOF)
  add_executable(nof_worker_pool_bench nof_worker_pool_bench.cpp)
  target_link_libraries(nof_worker_pool_bench PRIVATE mooncake_store glog::glog
                                                       gflags::gflags)
endif()

# Add BatchEvict benchmark executable
add_executable(batch_evict_bench batch_evict_bench.cpp)
target_link_libraries(
  batch_evict_bench PRIVATE mooncake_store cachelib_memory_allocator
                            gflags::gflags glog::glog pthread)

if(STORE_USE_ETCD)
  add_executable(oplog_batch_bench oplog_batch_bench.cpp)
  target_link_libraries(
    oplog_batch_bench PRIVATE mooncake_store glog::glog gflags::gflags
                              JsonCpp::JsonCpp)
endif()

add_executable(uring_batch_read_bench uring_batch_read_bench.cpp)
target_link_libraries(uring_batch_read_bench PRIVATE mooncake_store)

# --phoenix mode of storage_backend_bench: a vendor runtime is needed only
# to allocate the GPU destination buffer. Detect the toolkit locally
# (find_package results do not cross sibling directory scopes), same QUIET
# pattern as mooncake_store's D2H staging; without it the mode stays
# unavailable.
if(STORE_USE_PHOENIX)
  # The bench is the --phoenix A/B harness and keeps its own #ifdef
  # USE_PHOENIX (it is the one target that legitimately must rebuild when the
  # flag flips). mooncake_store no longer propagates the definition or the
  # libphoenix include dir to its consumers, so they are set here instead.
  target_compile_definitions(storage_backend_bench PRIVATE USE_PHOENIX)
  target_include_directories(storage_backend_bench
                             PRIVATE ${PHOENIX_INCLUDE_DIR})
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_link_libraries(storage_backend_bench PRIVATE CUDA::cudart)
    target_include_directories(storage_backend_bench
                               PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(storage_backend_bench PRIVATE USE_CUDA)
  else()
    message(STATUS "storage_backend_bench: CUDAToolkit not found, "
                   "--phoenix mode unavailable")
  endif()
endif()
