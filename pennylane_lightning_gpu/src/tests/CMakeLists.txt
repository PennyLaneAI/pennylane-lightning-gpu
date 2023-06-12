cmake_minimum_required(VERSION 3.21)

project(pennylane_lightning_tests)

set(CMAKE_CXX_STANDARD 20)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(PLLGPU_ENABLE_NATIVE "Enable native CPU build tuning" OFF)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.7
)

FetchContent_MakeAvailable(Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

add_executable(runner_gpu runner_main.cpp)
if(PLLGPU_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
	target_link_libraries(runner_gpu INTERFACE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(runner_gpu PUBLIC pennylane_lightning_gpu Catch2::Catch2 lightning_simulator)

target_sources(runner_gpu PRIVATE Test_StateVectorCudaManaged_NonParam.cpp
	                      Test_StateVectorCudaManaged_Param.cpp
	                      Test_AdjointDiffGPU.cpp
	                      Test_ObservablesGPU.cpp
	                      Test_GateCache.cpp
	                      Test_Generators.cpp
	                      Test_DataBuffer.cpp
	                      TestHelpers.hpp
)

target_compile_options(runner_gpu PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")

if(PLLGPU_ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(runner_gpu PRIVATE -march=native)
endif()

catch_discover_tests(runner_gpu)

if(PLLGPU_ENABLE_MPI)
    add_executable(mpi_runner ./mpi/mpi_runner_main.cpp)
    find_package(MPI REQUIRED)
    target_link_libraries(mpi_runner PUBLIC MPI::MPI_CXX)
    target_link_libraries(mpi_runner PUBLIC pennylane_lightning_gpu Catch2::Catch2 lightning_simulator)
    target_sources(mpi_runner PRIVATE ./mpi/Test_MPI_StateVectorCudaManaged_Param.cpp 
                                      ./mpi/Test_MPI_StateVectorCudaManaged_NonParam.cpp
                                      ./mpi/Test_MPIManager.cpp
    )
    target_compile_options(mpi_runner PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")

    if(PLLGPU_ENABLE_NATIVE)
        message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
        target_compile_options(mpi_runner PRIVATE -march=native)
    endif()
    catch_discover_tests(mpi_runner)
endif()
