project(lightning_gpu_interface_torch LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# Add Torch repo
include(FetchContent)
FetchContent_Declare(
    torch_python
    GIT_REPOSITORY https://github.com/pytorch/pytorch.git
    GIT_TAG        v1.11.0
)
FetchContent_MakeAvailable(torch_python)


set(TORCH_INTERFACE_FILES LightningGPU_Torch.cpp Tensor_Interop.hpp CACHE INTERNAL "" FORCE)
add_library(lightning_gpu_algorithms STATIC ${TORCH_INTERFACE_FILES})


add_library(lightning_gpu_interface_torch INTERFACE)
target_include_directories(lightning_gpu_interface_torch 
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(lightning_gpu_utils INTERFACE CUDA::cudart torch_python)
