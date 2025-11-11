add_executable(infeng_bench_stub bench_stub.cc)
target_link_libraries(infeng_bench_stub PRIVATE infeng_core)

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.4
)
FetchContent_MakeAvailable(benchmark)

add_executable(infeng_bench_tokenizer_gb bench_tokenizer_gb.cc)
target_link_libraries(infeng_bench_tokenizer_gb PRIVATE infeng_core benchmark::benchmark)

add_executable(infeng_bench_tokenizer bench_tokenizer.cc)
target_link_libraries(infeng_bench_tokenizer PRIVATE infeng_core)

add_executable(infeng_bench_gemm_lora bench_gemm_lora.cc)
target_link_libraries(infeng_bench_gemm_lora PRIVATE infeng_core benchmark::benchmark)

add_executable(infeng_bench_attn_prefill bench_attn_prefill.cc)
target_link_libraries(infeng_bench_attn_prefill PRIVATE infeng_core benchmark::benchmark)

add_executable(infeng_bench_attn_decode bench_attn_decode.cc)
target_link_libraries(infeng_bench_attn_decode PRIVATE infeng_core benchmark::benchmark)

add_executable(infeng_bench_scheduler_zipf bench_scheduler_zipf.cc ../perf/sched_zipf_sim.cc)
target_include_directories(infeng_bench_scheduler_zipf PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(infeng_bench_scheduler_zipf PRIVATE infeng_core)
