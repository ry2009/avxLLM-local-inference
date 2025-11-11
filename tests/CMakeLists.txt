add_executable(test_kv_paged test_kv_paged.cc)
target_link_libraries(test_kv_paged PRIVATE infeng_core)
add_test(NAME kv_paged COMMAND test_kv_paged)

add_executable(test_tokenizer_golden test_tokenizer_golden.cc)
target_link_libraries(test_tokenizer_golden PRIVATE infeng_core)
target_include_directories(test_tokenizer_golden PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME tokenizer_golden COMMAND test_tokenizer_golden)
set_tests_properties(tokenizer_golden PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_fused_gemm test_fused_gemm.cc)
target_link_libraries(test_fused_gemm PRIVATE infeng_core)
add_test(NAME fused_gemm COMMAND test_fused_gemm)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(INFENG_NUMPY_CHECK_ENV
    "OPENBLAS_NUM_THREADS=1"
    "OMP_NUM_THREADS=1"
    "VECLIB_MAXIMUM_THREADS=1"
    "OPENBLAS_CORETYPE=Haswell"
    "MKL_DEBUG_CPU_TYPE=5")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env ${INFENG_NUMPY_CHECK_ENV} ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/py/check_numpy.py
  RESULT_VARIABLE NUMPY_IMPORT_RESULT
  OUTPUT_VARIABLE NUMPY_CHECK_OUTPUT
  ERROR_VARIABLE NUMPY_CHECK_ERROR)
if (NUMPY_IMPORT_RESULT EQUAL 0)
  add_test(NAME infq_converter COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/infq_converter_test.py)
  set_tests_properties(infq_converter PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;OPENBLAS_CORETYPE=Haswell;MKL_DEBUG_CPU_TYPE=5")
else ()
  message(WARNING "Skipping infq_converter test (numpy unavailable)")
endif ()

add_executable(test_scheduler_basic test_scheduler_basic.cc)
target_link_libraries(test_scheduler_basic PRIVATE infeng_core)
add_test(NAME scheduler_basic COMMAND test_scheduler_basic)

add_executable(test_scheduler_smoke test_scheduler_smoke.cc)
target_link_libraries(test_scheduler_smoke PRIVATE infeng_core)
add_test(NAME scheduler_smoke COMMAND test_scheduler_smoke)

add_executable(test_cpu_lora_engine test_cpu_lora_engine.cc)
target_link_libraries(test_cpu_lora_engine PRIVATE infeng_core)
add_test(NAME cpu_lora_engine COMMAND test_cpu_lora_engine)

add_executable(test_rope_fht test_rope_fht.cc)
target_link_libraries(test_rope_fht PRIVATE infeng_core)
add_test(NAME rope_fht COMMAND test_rope_fht)

add_executable(test_scheduler_zipf test_scheduler_zipf.cc)
target_link_libraries(test_scheduler_zipf PRIVATE infeng_core)
add_test(NAME scheduler_zipf COMMAND test_scheduler_zipf)

add_executable(test_attn_decode test_attn_decode.cc)
target_link_libraries(test_attn_decode PRIVATE infeng_core)
add_test(NAME attn_decode COMMAND test_attn_decode)

add_executable(test_scheduler_metrics test_scheduler_metrics.cc)
target_link_libraries(test_scheduler_metrics PRIVATE infeng_core)
add_test(NAME test_scheduler_metrics COMMAND test_scheduler_metrics)
