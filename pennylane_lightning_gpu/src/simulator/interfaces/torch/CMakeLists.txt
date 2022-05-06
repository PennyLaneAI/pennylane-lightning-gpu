project(lightning_gpu_interface_torch LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(FetchContent)

ExternalProject_Add( torch_release
    URL https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
    SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/torch_release
    BUILD_IN_SOURCE 0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
add_library(lightning_gpu_interface_torch SHARED SVCudaTorch.hpp)
add_dependencies(lightning_gpu_interface_torch torch_release)

find_file(  TORCH_EXT
    NAMES   extension.h
    HINTS   ${FETCHCONTENT_BASE_DIR}/torch_release/include
            ${FETCHCONTENT_BASE_DIR}/torch_release/include/torch
            /usr/include
            /usr/local/include
            /opt
            include
            ENV CPATH
)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

target_include_directories(lightning_gpu_interface_torch PRIVATE ${FETCHCONTENT_BASE_DIR}/torch_release/include/ ${FETCHCONTENT_BASE_DIR}/torch_release/include/torch/csrc/api/include/)

target_link_libraries(lightning_gpu_interface_torch INTERFACE CUDA::cudart ${PYTHON_LIBRARIES})
set_target_properties(lightning_gpu_interface_torch PROPERTIES LINKER_LANGUAGE CXX)