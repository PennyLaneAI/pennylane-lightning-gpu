cmake_minimum_required(VERSION 3.21)

project(pennylane_lightning_tests)

set(CMAKE_CXX_STANDARD 20)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)

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
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
	  target_link_libraries(runner INTERFACE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(runner_gpu PUBLIC pennylane_lightning_gpu Catch2::Catch2 lightning_simulator)

target_sources(runner_gpu PRIVATE Test_StateVectorCudaManaged_NonParam.cpp 
                              Test_StateVectorCudaManaged_Param.cpp 
                              Test_AdjointDiffGPU.cpp 
                              Test_GateCache.cpp 
                              TestHelpers.hpp 
)

target_compile_options(runner_gpu PRIVATE "$<$<CONFIG:DEBUG>:-Wall>")

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Use -march=native for cpptests.")
    target_compile_options(runner_gpu PRIVATE -march=native)
endif()

catch_discover_tests(runner_gpu)
