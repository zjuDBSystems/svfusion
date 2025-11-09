cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
include(../rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

set(lang_list "CXX")

# find_program(CCACHE_PROGRAM ccache)
# set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
# only for Ampere architecture (RTX A6000, RTX 3090)
set(CMAKE_CUDA_ARCHITECTURES "90")

include(rapids-cuda)
rapids_cuda_init_architectures(FFANNS)
list(APPEND lang_list "CUDA")

project(
  FFANNS
  VERSION "0.0.1"
  LANGUAGES ${lang_list}
)
set(CMAKE_INSTALL_MESSAGE LAZY)

# Write the version header
rapids_cmake_write_version_file(include/ffanns/version_config.hpp)

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)
# rapids_cmake_build_type(Debug)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build ffanns shared libraries" ON)
# option(BUILD_TESTS "Build FFANNS unit-tests" ON)
# option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
# option(CUDA_ENABLE_LINEINFO
#        "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
# )
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(CUDA_STATIC_MATH_LIBRARIES "Statically link the CUDA math libraries" OFF)
# option(FFANNS_USE_RAFT_STATIC "Build and statically link the RAFT libraries" OFF)
# option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
# option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
# option(FFANNS_NVTX "Enable nvtx markers" OFF)
# option(FFANNS_RAFT_CLONE_ON_PIN "Explicitly clone RAFT branch when pinned to non-feature branch" OFF)
option(FFANNS_MISS_LOG "Enable miss ratio logging" OFF)
option(FFANNS_TIME_LOG "Enable time ratio logging" OFF)
option(FFANNS_DEBUG_LOG "Debug Logging" OFF)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)

# message(VERBOSE "FFANNS: Build FFANNS unit-tests: ${BUILD_TESTS}")
message(VERBOSE "FFANNS: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
# message(VERBOSE "FFANNS: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "FFANNS: Disable OpenMP: ${DISABLE_OPENMP}")
# message(VERBOSE "FFANNS: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
# message(VERBOSE "FFANNS: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
# message(VERBOSE "FFANNS: Enable nvtx markers: ${FFANNS_NVTX}")
message(VERBOSE "FFANNS: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "FFANNS: Statically link the CUDA math libraries: ${CUDA_STATIC_MATH_LIBRARIES}")
# message(VERBOSE "FFANNS: Build and statically link RAFT libraries: ${FFANNS_USE_RAFT_STATIC}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "FFANNS: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# ##################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "FFANNS: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}"
    )
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options ----------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_MATH_LIBRARIES)
  set(_ctk_static_suffix "_static")
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})
# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET ffanns-exports
  INSTALL_EXPORT_SET ffanns-exports
)

if(NOT DISABLE_OPENMP)
  rapids_find_package(
    OpenMP REQUIRED
    BUILD_EXPORT_SET ffanns-exports
    INSTALL_EXPORT_SET ffanns-exports
  )
  if(OPENMP_FOUND)
    message(VERBOSE "FFANNS: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

include(cmake/thirdparty/get_raft.cmake)
include(cmake/thirdparty/get_cutlass.cmake)

# if(BUILD_TESTS)
#   include(${rapids-cmake-dir}/cpm/gtest.cmake)
#   rapids_cpm_gtest(BUILD_STATIC)
# endif()

# ##################################################################################################
# * ffanns ---------------------------------------------------------------------

# search properties and targets
if(BUILD_SHARED_LIBS)
  
  add_library(
      ffanns-cagra-search STATIC
      src/neighbors/cagra_search_float.cu
      src/neighbors/cagra_search_uint8.cu
      src/neighbors/detail/cagra/compute_distance.cu
      src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_float_uint32_dim128_t8.cu
      src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_float_uint32_dim256_t16.cu
      src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_float_uint32_dim512_t32.cu
      src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_uint8_uint32_dim128_t8.cu
      src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_uint8_uint32_dim256_t16.cu
      src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_uint8_uint32_dim512_t32.cu
      src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_float_uint32_dim128_t8.cu
      src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_float_uint32_dim256_t16.cu
      src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_float_uint32_dim512_t32.cu
      src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_uint8_uint32_dim128_t8.cu
      src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_uint8_uint32_dim256_t16.cu
      src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_uint8_uint32_dim512_t32.cu
  )

  file(GLOB_RECURSE compute_distance_sources "src/neighbors/detail/cagra/compute_distance_*.cu")
  set_source_files_properties(${compute_distance_sources} PROPERTIES COMPILE_FLAGS -maxrregcount=64)

  target_compile_definitions(
    ffanns-cagra-search PUBLIC 
    $<$<BOOL:${FFANNS_MISS_LOG}>:FFANNS_MISS_LOG>
    $<$<BOOL:${FFANNS_DEBUG_LOG}>:FFANNS_DEBUG_LOG>
  )

  set_target_properties(
    ffanns-cagra-search
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               CUDA_SEPARABLE_COMPILATION ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(ffanns-cagra-search PRIVATE raft::raft)
  target_include_directories(
    ffanns-cagra-search PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )
  target_compile_options(
    ffanns-cagra-search PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FFANNS_CXX_FLAGS}>"
                              "$<$<COMPILE_LANGUAGE:CUDA>:${FFANNS_CUDA_FLAGS}>"
  )

  add_library(
    ffanns_objs OBJECT
    src/cluster/kmeans_balanced_fit_float.cu
    src/core/bitset.cu
    src/distance/detail/pairwise_matrix/dispatch_l2_expanded_float_float_float_int64_t.cu
    src/distance/detail/pairwise_matrix/dispatch_l2_expanded_float_float_float_int.cu
    src/distance/detail/fused_distance_nn.cu
    src/distance/distance.cu
    src/distance/pairwise_distance.cu
    src/neighbors/brute_force.cu
    src/neighbors/cagra_build_float.cu 
    src/neighbors/cagra_build_uint8.cu 
    src/neighbors/cagra_extend_float.cu
    src/neighbors/cagra_extend_uint8.cu  
    src/neighbors/cagra_serialize_float.cu
    src/neighbors/cagra_delete_float.cu
    src/neighbors/cagra_delete_uint8.cu
    src/neighbors/sample_filter.cu
    src/neighbors/detail/cagra/topk_for_cagra/topk.cu
    src/selection/select_k_float_int64_t.cu
    src/selection/select_k_float_int32_t.cu
    src/selection/select_k_float_uint32_t.cu
    src/selection/select_k_half_uint32_t.cu
  )

  target_compile_definitions(
    ffanns_objs PUBLIC 
    $<$<BOOL:${FFANNS_TIME_LOG}>:FFANNS_TIME_LOG>
  )

  set_target_properties(
    ffanns_objs
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options(
    ffanns_objs PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FFANNS_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${FFANNS_CUDA_FLAGS} -Wno-sign-compare>"
  )
  target_link_libraries(
    ffanns_objs PUBLIC raft::raft rmm::rmm ${FFANNS_CTK_MATH_DEPENDENCIES}
                     $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  )

  add_library(ffanns SHARED $<TARGET_OBJECTS:ffanns_objs>)
  add_library(ffanns_static STATIC $<TARGET_OBJECTS:ffanns_objs>)

  target_compile_options(
    ffanns INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                   --expt-relaxed-constexpr>
  )

  add_library(ffanns::ffanns ALIAS ffanns)
  add_library(ffanns::ffanns_static ALIAS ffanns_static)

  # 在这里添加优化配置 ⬇️⬇️⬇️
  # set_target_properties(ffanns PROPERTIES 
  #   LINK_DEPENDS_NO_SHARED TRUE
  #   INTERPROCEDURAL_OPTIMIZATION TRUE
  #   CUDA_SEPARABLE_COMPILATION OFF
  # )
  # ⬆️⬆️⬆️ 配置结束

  set_target_properties(
    ffanns_static
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               EXPORT_NAME ffanns_static
  )

  target_compile_options(ffanns_static PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FFANNS_CXX_FLAGS}>")

  target_include_directories(
    ffanns_objs
    PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    INTERFACE "$<INSTALL_INTERFACE:include>"
  )

  target_include_directories(
    ffanns_static
    PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
    INTERFACE "$<INSTALL_INTERFACE:include>"
  )

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(ffanns_static PRIVATE $<HOST_LINK:${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld>)

  target_include_directories(
    ffanns_static PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                       "$<INSTALL_INTERFACE:include>"
  )

  target_include_directories(
    ffanns PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                "$<INSTALL_INTERFACE:include>"
  )

  rapids_find_package(
    OpenMP REQUIRED
    BUILD_EXPORT_SET ffanns-exports
    INSTALL_EXPORT_SET ffanns-exports
  )
  
  # ensure NOT BUILD_CPU_ONLY
  set(FFANNS_CUSOLVER_DEPENDENCY CUDA::cusolver${_ctk_static_suffix})
  set(FFANNS_CUBLAS_DEPENDENCY CUDA::cublas${_ctk_static_suffix})
  set(FFANNS_CURAND_DEPENDENCY CUDA::curand${_ctk_static_suffix})
  set(FFANNS_CUSPARSE_DEPENDENCY CUDA::cusparse${_ctk_static_suffix})

  set(FFANNS_CTK_MATH_DEPENDENCIES ${FFANNS_CUBLAS_DEPENDENCY} ${FFANNS_CUSOLVER_DEPENDENCY}
                                  ${FFANNS_CUSPARSE_DEPENDENCY} ${FFANNS_CURAND_DEPENDENCY}
  )

  # Keep cuVS as lightweight as possible. Only CUDA libs and rmm should be used in global target.
  target_link_libraries(
    ffanns
    PUBLIC rmm::rmm raft::raft ${FFANNS_CTK_MATH_DEPENDENCIES}
    PRIVATE nvidia::cutlass::cutlass $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            ffanns-cagra-search  
  )

  target_link_libraries(
    ffanns_static
    PUBLIC rmm::rmm raft::raft ${FFANNS_CTK_MATH_DEPENDENCIES}
    PRIVATE nvidia::cutlass::cutlass $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  )
  # end if NOT BUILD_CPU_ONLY

  # Endian detection
  include(TestBigEndian)
  test_big_endian(BIG_ENDIAN)
  if(BIG_ENDIAN)
    target_compile_definitions(ffanns PRIVATE FFANNS_SYSTEM_LITTLE_ENDIAN=0)
  else()
    target_compile_definitions(ffanns PRIVATE FFANNS_SYSTEM_LITTLE_ENDIAN=1)
  endif()

  file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
    [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
  )

# ################################################################################################
  # * NVTX support in ffanns -----------------------------------------------------

  if(FFANNS_NVTX)
    # This enables NVTX within the project with no option to disable it downstream.
    target_link_libraries(ffanns PUBLIC CUDA::nvtx3)
    target_compile_definitions(ffanns PUBLIC NVTX_ENABLED)

    target_link_libraries(ffanns-cagra-search PUBLIC CUDA::nvtx3)
    target_compile_definitions(ffanns-cagra-search PUBLIC NVTX_ENABLED)
  else()
    # Allow enable NVTX downstream if not set here. This creates a new option at build/install time,
    # which is set by default to OFF, but can be enabled in the dependent project.
    get_property(
      nvtx_option_help_string
      CACHE FFANNS_NVTX
      PROPERTY HELPSTRING
    )
    string(
      CONCAT
        nvtx_export_string
        "option(FFANNS_NVTX \""
        ${nvtx_option_help_string}
        "\" OFF)"
        [=[

target_link_libraries(ffanns::ffanns INTERFACE $<$<BOOL:${FFANNS_NVTX}>:CUDA::nvtx3>)
target_compile_definitions(ffanns::ffanns INTERFACE $<$<BOOL:${FFANNS_NVTX}>:NVTX_ENABLED>)

  ]=]
    )
  endif()

  set_target_properties(
    ffanns
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(
    ffanns PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FFANNS_CXX_FLAGS}>"
                 "$<$<COMPILE_LANGUAGE:CUDA>:${FFANNS_CUDA_FLAGS}>"
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target ffanns ffanns_static ffanns-cagra-search ffanns_objs)
      target_compile_options(${target} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -Ofast -mtune=native>)
    endforeach()
    # -Ofast -march=native -mtune=native

    foreach(target ffanns ffanns_static ffanns-cagra-search ffanns_objs)
      target_compile_options(${target} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -Xcompiler=-mavx2,-mfma,-mavx512f,-Ofast,-mtune=native>)
    endforeach()

    # target_compile_options(ffanns PRIVATE -mavx2 -mfma -mavx512f)
    # target_compile_options(ffanns_static PRIVATE -mavx2 -mfma -mavx512f)
    # target_compile_options(ffanns-cagra-search PRIVATE -mavx2 -mfma -mavx512f)
    # target_compile_options(ffanns_objs PRIVATE -mavx2 -mfma -mavx512f -Ofast -march=native)
  elseif(MSVC)
    target_compile_options(ffanns PRIVATE /arch:AVX2)
    target_compile_options(ffanns_static PRIVATE /arch:AVX2)
    target_compile_options(ffanns-cagra-search PRIVATE /arch:AVX2)
    target_compile_options(ffanns_objs PRIVATE /arch:AVX2)
  endif()

  # 同时定义 USE_AVX2 宏
  target_compile_definitions(ffanns PRIVATE USE_AVX2)
  target_compile_definitions(ffanns_static PRIVATE USE_AVX2)
  target_compile_definitions(ffanns-cagra-search PRIVATE USE_AVX2)
  target_compile_definitions(ffanns_objs PRIVATE USE_AVX2)

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(ffanns PRIVATE $<HOST_LINK:${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld>)

  # ################################################################################################
  # * install targets-----------------------------------------------------------
  rapids_cmake_install_lib_dir(lib_dir)
  include(GNUInstallDirs)
  include(CPack)

  install(
    TARGETS ffanns ffanns_static
    DESTINATION ${lib_dir}
    COMPONENT ffanns
    EXPORT ffanns-exports
  )

  install(
    DIRECTORY include/ffanns
    COMPONENT ffanns
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/include/ffanns/version_config.hpp
    COMPONENT ffanns
    DESTINATION include/ffanns
  )

  # Use `rapids_export` for 22.04 as it will have COMPONENT support
  rapids_export(
    INSTALL ffanns
    EXPORT_SET ffanns-exports
    NAMESPACE ffanns::
  )

  # ################################################################################################
  # * build export -------------------------------------------------------------
  rapids_export(
    BUILD ffanns
    EXPORT_SET ffanns-exports
    NAMESPACE ffanns::
  )
endif()
