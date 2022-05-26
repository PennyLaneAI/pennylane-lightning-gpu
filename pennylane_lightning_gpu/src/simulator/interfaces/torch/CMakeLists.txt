project(lightning_gpu_interface_torch LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(FetchContent)

FetchContent_Declare(
    torch_release
    URL      https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip
)
FetchContent_MakeAvailable(torch_release)

add_library(lightning_gpu_interface_torch STATIC SVCudaTorch.hpp SVCudaTorch.cpp)

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

find_package(Python ${PY_VERSION} 
                    EXACT 
                    REQUIRED
)

# Due to the sheer difficulty of getting CMake to find the Python include header path on a manylinux image, defining a custom command here makes the most sense
execute_process(COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'), end='')" OUTPUT_VARIABLE PY_INCLUDE_DIR)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

get_filename_component(CUSTATEVEC_INC_DIR ${CUSTATEVEC_INC} DIRECTORY)

target_include_directories(lightning_gpu_interface_torch PUBLIC 
                                ${FETCHCONTENT_BASE_DIR}/torch_release-src/include 
                                ${FETCHCONTENT_BASE_DIR}/torch_release-src/include/torch/csrc/api/include/
                                ${CMAKE_CURRENT_SOURCE_DIR}
                                ${CUDAToolkit_INCLUDE_DIRS} 
                                ${CUSTATEVEC_INC_DIR}
                                ${PY_INCLUDE_DIR} )

target_link_libraries(lightning_gpu_interface_torch PUBLIC CUDA::cudart lightning_gpu_simulator lightning_gpu_utils lightning_simulator)
set_target_properties(lightning_gpu_interface_torch PROPERTIES LINKER_LANGUAGE CXX)

pybind11_add_module(lightning_gpu_bindings_torch     "Bindings.cpp")
target_link_libraries(lightning_gpu_bindings_torch PRIVATE lightning_gpu_interface_torch)
