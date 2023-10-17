# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

# We rely on the MKL shim library.
find_package(MKLShim REQUIRED)

# To aid portability between our implementation and
# the ROCm hipFFT implementation, we use the ROCm
# hipFFT library's hipfft.h header.
include(ExternalProject)
option(H4I_ROCM_HIPFFT_TAG "Tag to use from the ROCm hipFFT repository when obtaining its hipfft.h header" "rocm-5.4.0")
ExternalProject_Add(ROCmHipfft
    GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/hipfft
    GIT_TAG ${H4I_ROCM_HIPFFT_TAG}

    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND sh ${CMAKE_SOURCE_DIR}/Scripts/install-hipfft-header.sh <SOURCE_DIR> ${CMAKE_BINARY_DIR} ${CMAKE_INSTALL_PREFIX}
)

# Product the hipfft version header required by hipfft.h in recent ROCm versions.
configure_file(
    ${CMAKE_SOURCE_DIR}/include/hipfft/hipfft-version.h.in
    ${CMAKE_BINARY_DIR}/include/hipfft/hipfft-version.h
    @ONLY)

# Specify how to build the library.
add_library(HipFFT SHARED
    hipfft.cpp)
target_include_directories(HipFFT
	PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include;${CMAKE_BINARY_DIR}/include>"
		$<INSTALL_INTERFACE:include>)

target_link_libraries(HipFFT
	PRIVATE
        HipFFTCommonConfig
        H4I::MKLShim
    PUBLIC
		hip::host
    )

# Specify how to install the library.
include (GNUInstallDirs)
install(TARGETS HipFFT
		EXPORT HipFFT
)

# install the primary headers
install(FILES
        ${CMAKE_BINARY_DIR}/include/hipfft/hipfft.h
        ${CMAKE_BINARY_DIR}/include/hipfft/hipfft-version.h
        ${CMAKE_SOURCE_DIR}/include/hipfft/hipfft-export.h
        ${CMAKE_SOURCE_DIR}/include/hipfft/hipfftXt.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hipfft
)

# install the redirection headers for backwards compatibility
# with previous versions of hipfft
install(FILES
	${CMAKE_SOURCE_DIR}/include/hipfft.h
	${CMAKE_SOURCE_DIR}/include/hipfft-version.h
	${CMAKE_SOURCE_DIR}/include/hipfft-export.h
        ${CMAKE_SOURCE_DIR}/include/hipfftXt.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

install(EXPORT HipFFT
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HipFFT
	NAMESPACE H4I::
)

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_SOURCE_DIR}/CMake/HipFFTConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/HipFFTConfig.cmake
	PATH_VARS CMAKE_INSTALL_INCLUDEDIR
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HipFFT
)

install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/HipFFTConfig.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HipFFT
)

