# Test-only wrapper: builds tests/codegen/standalone_driver.cpp against a
# prebuilt ggml (GGML_BUILD_DIR, produced by the pytest session fixture).
#
# All machine specifics (compiler, threads, OpenMP, math libs) come from
# ggml's own exported package and CMake's toolchain detection. This file
# contains no toolchain flags by design; porting to a new platform must not
# require editing test code.
cmake_minimum_required(VERSION 3.18)
project(ggmlc_standalone_test LANGUAGES C CXX)

# Provided via -D: GGML_BUILD_DIR, MODEL_PROJ_DIR, RT_INCLUDE_DIR,
# TEST_DRIVER_SRC, TEST_MODEL.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ggml CONFIG REQUIRED PATHS "${GGML_BUILD_DIR}" NO_DEFAULT_PATH)

add_executable(standalone_test "${TEST_DRIVER_SRC}")
target_compile_definitions(standalone_test PRIVATE
    TEST_MODEL=${TEST_MODEL}
    TEST_HEADER=${TEST_MODEL}.h
)
target_include_directories(standalone_test PRIVATE
    "${MODEL_PROJ_DIR}"
    "${RT_INCLUDE_DIR}"
)
target_link_libraries(standalone_test PRIVATE ggml::ggml)
