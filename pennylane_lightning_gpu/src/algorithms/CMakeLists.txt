project(lightning_gpu_algorithms LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(GPU_ALGORITHM_FILES AdjointDiffGPU.hpp AdjointDiffGPU.cpp GateGenerators.hpp ObservablesGPU.hpp CACHE INTERNAL "" FORCE)

if(PLGPU_ENABLE_MPI)
    list(APPEND SIMULATOR_FILES AdjointDiffGPUMPI.hpp AdjointDiffGPUMPI.cpp ObservablesGPUMPI.hpp)
endif()

add_library(lightning_gpu_algorithms STATIC ${GPU_ALGORITHM_FILES})

target_include_directories(lightning_gpu_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(lightning_gpu_algorithms PUBLIC pennylane_lightning lightning_algorithms lightning_qubit lightning_gpu_simulator lightning_gpu_utils) 
set_property(TARGET lightning_gpu_algorithms PROPERTY POSITION_INDEPENDENT_CODE ON)