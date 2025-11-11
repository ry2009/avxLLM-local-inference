set(INFENG_ENGINE_SOURCES
    runtime/runtime_init.cc
    runtime/cpu_lora_engine.cc
    runtime/microbatch.cc
    runtime/scheduler.cc
    runtime/affinity.cc
    mem/arena.cc
    mem/kv_paged.cc
    telemetry/counters.cc
    adapters/lora_loader.cc
    kernels/gemm/pack.cc
    kernels/ref/attention_ref.cpp
    kernels/ref/gemm_ref.cpp
    kernels/gemm/fused_base_lora_avx512.cc
    kernels/attn/rope_fht_kv_avx512.cc
    kernels/attn/decode_int8_qk_avx512.cc
    kernels/attn/kv_dequant_blocked.cc
    kernels/attn/softmax_mx.cc
    quant/loaders/infq_reader.cc
    quant/dequantize.cc
    quant/packers/int8_rowwise.cc
    quant/packers/int4_rowwise.cc
    tokenizer/tokenizer.cpp)

add_library(infeng_core STATIC ${INFENG_ENGINE_SOURCES})
target_include_directories(infeng_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${INFENG_GENERATED_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/tokenizer)

if (INFENG_AVX2)
    target_compile_options(infeng_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
endif ()

if (INFENG_AVX512)
    target_compile_options(infeng_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni>)
    target_compile_definitions(infeng_core PRIVATE INFENG_ENABLE_AVX512=1)
else ()
    target_compile_definitions(infeng_core PRIVATE INFENG_ENABLE_AVX512=0)
endif ()

if (INFENG_AMX)
    target_compile_options(infeng_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mamx-int8 -mamx-bf16>)
endif ()

target_compile_options(infeng_core
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffp-contract=fast -fno-rtti -fno-omit-frame-pointer>)

find_program(CARGO_EXECUTABLE cargo)
if (NOT CARGO_BIN)
    message(FATAL_ERROR "cargo executable not found. Install Rust toolchain.")
endif ()

set(FASTTOK_DEFAULT_RUSTFLAGS "-C target-cpu=native -C codegen-units=1")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "[Xx]86_64|[Aa][Mm][Dd]64")
    set(FASTTOK_DEFAULT_RUSTFLAGS "${FASTTOK_DEFAULT_RUSTFLAGS} -C target-feature=+avx2,+fma")
endif ()
if (DEFINED FASTTOK_RUSTFLAGS AND FASTTOK_RUSTFLAGS STREQUAL "-C target-cpu=native -C codegen-units=1 -C embed-bitcode=no")
    unset(FASTTOK_RUSTFLAGS CACHE)
endif ()
if (NOT DEFINED FASTTOK_RUSTFLAGS OR FASTTOK_RUSTFLAGS STREQUAL "")
    set(FASTTOK_RUSTFLAGS "${FASTTOK_DEFAULT_RUSTFLAGS}")
endif ()
set(FASTTOK_RUSTFLAGS "${FASTTOK_RUSTFLAGS}" CACHE STRING "Additional RUSTFLAGS applied when building fasttok" FORCE)
set(_effective_rustflags "$ENV{RUSTFLAGS}")
if (_effective_rustflags)
    set(_effective_rustflags "${_effective_rustflags} ${FASTTOK_RUSTFLAGS}")
else ()
    set(_effective_rustflags "${FASTTOK_RUSTFLAGS}")
endif ()

set(FASTTOK_TARGET_DIR ${CMAKE_BINARY_DIR}/tokenizer)
set(FASTTOK_LIB ${FASTTOK_TARGET_DIR}/release/libfasttok.a)

add_custom_command(
    OUTPUT ${FASTTOK_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=${_effective_rustflags}" ${CARGO_BIN} build --release --target-dir ${FASTTOK_TARGET_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/tokenizer
    COMMENT "Building fasttok tokenizer runtime"
    BYPRODUCTS ${FASTTOK_LIB}
    VERBATIM)

add_custom_target(fasttok_build DEPENDS ${FASTTOK_LIB})
add_library(fasttok STATIC IMPORTED GLOBAL)
set_target_properties(fasttok PROPERTIES IMPORTED_LOCATION ${FASTTOK_LIB})
add_dependencies(fasttok fasttok_build)
add_dependencies(infeng_core fasttok)
target_link_libraries(infeng_core PRIVATE fasttok pthread)

set(INFENG_FASTTOK_LIB ${FASTTOK_LIB} PARENT_SCOPE)

if (UNIX)
    target_link_options(infeng_core PRIVATE -Wl,--as-needed)
endif ()

install(TARGETS infeng_core ARCHIVE DESTINATION lib)
install(FILES ${FASTTOK_LIB} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/engine/include/infeng/ DESTINATION include/infeng FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_SOURCE_DIR}/engine/tokenizer/c_api.h DESTINATION include/infeng/tokenizer)
