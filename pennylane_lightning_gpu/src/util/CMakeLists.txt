project(lightning_gpu_utils LANGUAGES CUDA CXX)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


set(DATA_BUFFER_FILES DataBuffer.cu CACHE INTERNAL "" FORCE)

add_library(lightning_gpu_utils STATIC ${DATA_BUFFER_FILES})

set_target_properties( lightning_gpu_utils
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties( lightning_gpu_utils
                       PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_target_properties(lightning_gpu_utils PROPERTIES CUDA_ARCHITECTURES "80;86")

target_include_directories(lightning_gpu_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)


get_filename_component(CUSTATEVEC_INC_DIR ${CUSTATEVEC_INC} DIRECTORY)
target_include_directories(lightning_gpu_utils PUBLIC ${CUDA_INCLUDE_DIRS} ${CUSTATEVEC_INC_DIR})

target_link_libraries(lightning_gpu_utils PUBLIC ${CUDA_SHARED_RT} ${CUSTATEVEC_LIB})
target_link_libraries(lightning_gpu_utils INTERFACE CUDA::cublas CUDA::cusparse)
target_link_libraries(lightning_gpu_utils PUBLIC lightning_utils)

set_property(TARGET lightning_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

