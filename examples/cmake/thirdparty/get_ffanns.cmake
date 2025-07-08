# Use RAPIDS_VERSION from cmake/thirdparty/fetch_rapids.cmake
set(FFANNS_VERSION "0.0.1")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG ENABLE_NVTX BUILD_CUVS_C_LIBRARY)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )


    set(CUVS_COMPONENTS "")
    if(PKG_BUILD_CUVS_C_LIBRARY)
        string(APPEND CUVS_COMPONENTS " c_api")
    endif()
    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-examples-exports
            INSTALL_EXPORT_SET  cuvs-examples-exports
            COMPONENTS ${CUVS_COMPONENTS}
            CPM_ARGS
            GIT_REPOSITORY https://github.com/${PKG_FORK}/cuvs.git
            GIT_TAG        ${PKG_PINNED_TAG}
            SOURCE_SUBDIR  cpp
            OPTIONS
            "BUILD_C_LIBRARY ${PKG_BUILD_CUVS_C_LIBRARY}"
            "BUILD_TESTS OFF"
            "CUVS_NVTX ${PKG_ENABLE_NVTX}"
            )
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different CUVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}.00
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        ENABLE_NVTX              OFF
        BUILD_CUVS_C_LIBRARY     ${BUILD_CUVS_C_LIBRARY}
)
