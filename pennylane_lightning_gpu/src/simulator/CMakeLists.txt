project(lightning_gpu_simulator)
set(CMAKE_CXX_STANDARD 20)
enable_language(CXX CUDA)

set(SIMULATOR_FILES StateVectorCudaBase.hpp StateVectorCudaManaged.hpp cuGateCache.hpp cuGates_host.hpp CACHE INTERNAL "" FORCE)
add_library(lightning_gpu_simulator STATIC ${SIMULATOR_FILES})

get_filename_component(CUSTATEVEC_INC_DIR ${CUSTATEVEC_INC} DIRECTORY)
target_include_directories(lightning_gpu_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} ${CUSTATEVEC_INC_DIR})

target_link_libraries(lightning_gpu_simulator PUBLIC lightning_simulator lightning_gates lightning_utils lightning_gpu_utils )
target_link_libraries(lightning_gpu_simulator PUBLIC ${CUDA_SHARED_RT} ${CUSTATEVEC_LIB})
set_target_properties(lightning_gpu_simulator PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(lightning_gpu_simulator PROPERTIES LINKER_LANGUAGE CXX)

if(ENABLE_NATIVE)
    target_compile_options(lightning_gpu_simulator PRIVATE -march=native)
endif()