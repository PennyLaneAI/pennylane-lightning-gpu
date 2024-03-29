project(lightning_gpu_simulator)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(SIMULATOR_FILES StateVectorCudaBase.hpp StateVectorCudaManaged.hpp cuGateCache.hpp cuGates_host.hpp initSV.cu CACHE INTERNAL "" FORCE)

if(PLGPU_ENABLE_MPI)
    list(APPEND SIMULATOR_FILES StateVectorCudaMPI.hpp)
endif()

add_library(lightning_gpu_simulator STATIC ${SIMULATOR_FILES})

get_filename_component(CUSTATEVEC_INC_DIR ${CUSTATEVEC_INC} DIRECTORY)
target_include_directories(lightning_gpu_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${CUSTATEVEC_INC_DIR})

target_link_libraries(lightning_gpu_simulator PUBLIC lightning_qubit lightning_gates lightning_utils lightning_gpu_utils)
target_link_libraries(lightning_gpu_simulator PUBLIC ${CUDA_SHARED_RT} ${CUSTATEVEC_LIB})
set_target_properties(lightning_gpu_simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(lightning_gpu_simulator PROPERTIES LINKER_LANGUAGE CXX)

if(ENABLE_NATIVE)
    target_compile_options(lightning_gpu_simulator PRIVATE -march=native)
endif()
