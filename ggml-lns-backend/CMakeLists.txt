cmake_minimum_required(VERSION 3.14)
project(ggml-lns-backend C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---- Fetch ggml ----
include(FetchContent)
FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

# Only build ggml core, skip examples/tests
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(ggml)

# ---- ggml-lns static library ----
add_library(ggml-lns STATIC src/ggml-lns.cpp)
target_include_directories(ggml-lns PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../xlnscpp
)
# ggml-backend-impl.h is an internal header in ggml's src/ directory
target_include_directories(ggml-lns PRIVATE
    ${ggml_SOURCE_DIR}/src
)
target_link_libraries(ggml-lns PUBLIC ggml)

# Use math.h-based ideal sb/db functions (not lookup tables)
target_compile_definitions(ggml-lns PRIVATE xlns16_ideal)

# ---- Demo executables ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(lns_matmul examples/lns_matmul.cpp)
target_link_libraries(lns_matmul PRIVATE ggml-lns ggml)

add_executable(mini_transformer examples/mini_transformer.cpp)
target_link_libraries(mini_transformer PRIVATE ggml-lns ggml)
