cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

find_program(CCACHE_PROGRAM ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

# only for Ampere architecture (RTX A6000, RTX 3090)
set(CMAKE_CUDA_ARCHITECTURES "80")

# ------------- configure rapids-cmake --------------#
include(./cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#
rapids_cuda_init_architectures(test_ffanns)

# Set RAFT logging level to DEBUG
option(RAPIDS_LOGGING_LEVEL "Set the logging level for RAFT" "DEBUG")
# Set RAFT_LOG_LEVEL environment variable to DEBUG
set(ENV{RAFT_LOG_LEVEL} "DEBUG")

project(test_ffanns LANGUAGES CXX CUDA)
find_package(Threads)

# ------------- configure dependencies -----------------#
rapids_cpm_init()

# Find installed ffanns package
find_package(ffanns REQUIRED)

# 使用CPM获取yaml-cpp
rapids_cpm_find(yaml-cpp 0.7.0
  GITHUB_REPOSITORY jbeder/yaml-cpp
  GIT_TAG yaml-cpp-0.7.0
  OPTIONS
    "YAML_CPP_BUILD_TESTS OFF"
    "YAML_CPP_BUILD_TOOLS OFF"
)

# -------------- compile examples ----------------- #
add_executable(benchmark src/workload_manager.cu src/benchmark.cu)

target_include_directories(benchmark
    PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src/include
)

# Link libraries
target_link_libraries(benchmark           # 为benchmark添加库链接
    PRIVATE 
    ffanns::ffanns
    yaml-cpp
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

# Set properties
foreach(target benchmark)
    set_target_properties(${target}
        PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endforeach()