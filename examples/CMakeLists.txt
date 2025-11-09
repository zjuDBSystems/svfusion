cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# find_program(CCACHE_PROGRAM ccache)
# set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

# only for Ampere architecture (RTX A6000, RTX 3090)
set(CMAKE_CUDA_ARCHITECTURES "90")

# 优先使用conda环境的CUDA，避免与系统CUDA冲突
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "$ENV{CONDA_PREFIX}")
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CONDA_PREFIX}")
# 确保运行时也使用conda的CUDA库
set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")

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

# 使用CPM获取yaml-cpp (使用0.8.0版本，兼容新版CMake)
CPMAddPackage(
  NAME yaml-cpp
  GITHUB_REPOSITORY jbeder/yaml-cpp
  GIT_TAG 0.8.0
  VERSION 0.8.0
  OPTIONS
    "YAML_CPP_BUILD_TESTS OFF"
    "YAML_CPP_BUILD_TOOLS OFF"
    "YAML_CPP_BUILD_CONTRIB OFF"
    "YAML_CPP_INSTALL OFF"
    "YAML_BUILD_SHARED_LIBS OFF"
)

# ------------- find Protobuf and Kafka -----------------#
# Protobuf应该通过conda install -c conda-forge protobuf libprotobuf安装
# 强制使用conda环境中的protoc以避免版本不匹配
set(Protobuf_PROTOC_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/protoc")
find_package(Protobuf REQUIRED)
find_library(RDKAFKA_LIBRARY rdkafka REQUIRED)
find_path(RDKAFKA_INCLUDE_DIR librdkafka/rdkafka.h REQUIRED)

# ------------- compile protobuf messages -----------------#
set(PROTO_FILES proto/streaming_messages.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# -------------- compile examples ----------------- #
# Original benchmark executable (保留原有的benchmark)
add_executable(benchmark src/workload_manager.cu src/benchmark.cu)

# Streaming producer executable
add_executable(streaming_producer 
    src/streaming_producer_main.cu
    ${PROTO_SRCS}
)

# Streaming consumer executable  
add_executable(streaming_consumer
    src/streaming_consumer_main.cu
    ${PROTO_SRCS}
)

# Include directories for all targets
target_include_directories(benchmark
    PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src/include
)

target_include_directories(streaming_producer
    PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src/include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
    ${PROTOBUF_INCLUDE_DIRS}
    ${RDKAFKA_INCLUDE_DIR}
)

target_include_directories(streaming_consumer
    PRIVATE
    ${CMAKE_BINARY_DIR}/_deps/yaml-cpp-src/include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
    ${PROTOBUF_INCLUDE_DIRS}
    ${RDKAFKA_INCLUDE_DIR}
)

# Link libraries for benchmark (原有的)
target_link_libraries(benchmark
    PRIVATE 
    ffanns::ffanns
    yaml-cpp
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

# Link libraries for streaming_producer
target_link_libraries(streaming_producer
    PRIVATE 
    ffanns::ffanns
    yaml-cpp
    ${PROTOBUF_LIBRARIES}
    ${RDKAFKA_LIBRARY}
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

# Link libraries for streaming_consumer
target_link_libraries(streaming_consumer
    PRIVATE 
    ffanns::ffanns
    yaml-cpp
    ${PROTOBUF_LIBRARIES}
    ${RDKAFKA_LIBRARY}
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

# Set properties for all targets
foreach(target benchmark streaming_producer streaming_consumer)
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