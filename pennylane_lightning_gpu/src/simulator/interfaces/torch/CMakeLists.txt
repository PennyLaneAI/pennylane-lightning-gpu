project(lightning_gpu_interface_torch LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(FetchContent)

FetchContent_Declare(
    torch_release
    URL      https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
)
FetchContent_MakeAvailable(torch_release)

add_library(lightning_gpu_interface_torch SHARED SVCudaTorch.hpp SVCudaTorch.cpp)

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

message("${FETCHCONTENT_BASE_DIR}/torch_release-src/include")
message("${FETCHCONTENT_BASE_DIR}/torch_release/include/torch")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

target_include_directories(lightning_gpu_interface_torch PRIVATE ${FETCHCONTENT_BASE_DIR}/torch_release-src/include ${FETCHCONTENT_BASE_DIR}/torch_release-src/include/torch/csrc/api/include/ ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(lightning_gpu_interface_torch INTERFACE CUDA::cudart ${PYTHON_LIBRARIES} lightning_gpu_simulator)
set_target_properties(lightning_gpu_interface_torch PROPERTIES LINKER_LANGUAGE CXX)