project(lightning_gpu_utils LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

add_library(lightning_gpu_utils INTERFACE)
target_include_directories(lightning_gpu_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(lightning_gpu_utils INTERFACE CUDA::cublas)
