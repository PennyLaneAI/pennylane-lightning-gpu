##########################
## Set Project version
##########################
cmake_minimum_required(VERSION 3.21)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█▀▀░█▀█░█░█
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█░█░█▀▀░█░█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀▀▀░▀░░░▀▀▀
]=])
message(${LOGO})

project(pennylane_lightning_gpu
    DESCRIPTION "Lightning-GPU bindings for PennyLane. Backed by NVIDIA cuQuantum SDK."
    LANGUAGES CXX C CUDA
)

##########################
## Utility methods
##########################

# Read and set pennylane_lightning_gpu version
function(set_pennylane_lightning_gpu_version VERSION_FILE_PATH)
    file(STRINGS ${VERSION_FILE_PATH} VERSION_FILE_STR)
    foreach (LINE IN LISTS VERSION_FILE_STR)
        if("${LINE}" MATCHES "__version__.*")
            set(VERSION_LINE_STR "${LINE}")
        endif()
    endforeach()
    string(REGEX REPLACE "__version__ = \"(.*)\"" "\\1" VERSION_STRING ${VERSION_LINE_STR})
    set(VERSION_STRING ${VERSION_STRING} PARENT_SCOPE)
endfunction()

set_pennylane_lightning_gpu_version(${PROJECT_SOURCE_DIR}/pennylane_lightning_gpu/_version.py)

message(STATUS "pennylane_lightning_gpu version ${VERSION_STRING}")
set(PROJECT_VERSION ${VERSION_STRING})

##########################
## Enfore C++ Standard
##########################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##########################
## Set Default Options
##########################
# Compiler options
option(ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(BUILD_TESTS "Build cpp tests" OFF)
option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)
option(DISABLE_CUDA_SAFETY "Build without CUDA call safety checks" OFF)
option(ENABLE_PYTHON "Build Python bindings" ON)
option(ENABLE_SANITIZER "Enable address sanitizer" OFF)

# Build options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_RPATH "$ORIGIN/../cuquantum/lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../cuquantum/lib:$ORIGIN/")

if(ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++17;
    )
endif()

if(ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address -static-libasan)
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

##########################
## Fetch dependencies
##########################
# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.9.1
)
FetchContent_MakeAvailable(pybind11)

# Add PennyLane-Lightning
set(ENABLE_OPENMP ON)
FetchContent_Declare(
    pennylane_lightning
    GIT_REPOSITORY https://github.com/PennyLaneAI/pennylane-lightning.git
    GIT_TAG       latest
)
FetchContent_MakeAvailable(pennylane_lightning)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

find_package (Python COMPONENTS Interpreter Development)

find_library(CUSTATEVEC_LIB
    NAMES   libcustatevec.so.1 custatevec.so.1
    HINTS   /usr/lib
            /usr/local/cuda
            /usr/local/lib
            /opt
            /opt/cuda
            lib
            lib64
            ${CUQUANTUM_SDK}/lib
            ${CUQUANTUM_SDK}/lib64
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${Python_SITELIB}/cuquantum/lib
            ENV LD_LIBRARY_PATH
)

find_file( CUSTATEVEC_INC
    NAMES   custatevec.h
    HINTS   /usr/include
            /usr/local/cuda
            /usr/local/include
            /opt
            /opt/cuda
            include
            ${CUQUANTUM_SDK}/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CUDA_TOOLKIT_ROOT_DIR}/include
            ${Python_SITELIB}/cuquantum/include
            ENV CPATH
)

if(NOT CUSTATEVEC_LIB OR NOT CUSTATEVEC_INC)
    message(FATAL_ERROR "\nUnable to find cuQuantum SDK installation. Please ensure it is correctly installed and available on path.")
endif()

##########################
## Compile options
##########################
set(CUDA_SEPARABLE_COMPILATION ON)

############################
## Create libraries
############################
add_subdirectory(pennylane_lightning_gpu/src)
add_library(pennylane_lightning_gpu INTERFACE)
target_link_libraries(pennylane_lightning_gpu INTERFACE pennylane_lightning
                                                        lightning_gpu_utils
                                                        lightning_gpu_simulator
                                                        lightning_gpu_algorithms)
target_include_directories(pennylane_lightning_gpu INTERFACE "pennylane_lightning_gpu/src")
target_link_libraries(pennylane_lightning_gpu INTERFACE ${CUSTATEVEC_LIB} ${CUDA_SHARED_RT})

# Create binding module
if(ENABLE_PYTHON)
    pybind11_add_module(lightning_gpu_qubit_ops     "pennylane_lightning_gpu/src/bindings/Bindings.cpp")
    target_link_libraries(lightning_gpu_qubit_ops PRIVATE pennylane_lightning_gpu)
    set_target_properties(lightning_gpu_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(lightning_gpu_qubit_ops PROPERTIES INSTALL_RPATH "$ORIGIN/../cuquantum/lib:$ORIGIN/")
    target_compile_options(lightning_gpu_qubit_ops PRIVATE "$<$<CONFIG:RELEASE>:-W>")
    target_compile_definitions(lightning_gpu_qubit_ops PRIVATE VERSION_INFO=${VERSION_STRING})
    target_include_directories(lightning_gpu_qubit_ops PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
    target_link_libraries(lightning_gpu_qubit_ops PRIVATE CUDA::cudart)
endif()

# To avoid DSO errors on platforms preferring static linkage, uncomment the following line:
# string(REPLACE "libcudart_static.a" "libcudart.so" CUDA_SHARED_RT "${CUDA_LIBRARIES}")
target_include_directories(pennylane_lightning_gpu INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(pennylane_lightning_gpu INTERFACE CUDA::cudart)
set_target_properties(pennylane_lightning_gpu PROPERTIES INSTALL_RPATH "$ORIGIN/../cuquantum/lib:$ORIGIN/")

if(ENABLE_OPENMP)
	target_link_libraries(pennylane_lightning_gpu INTERFACE OpenMP::OpenMP_CXX)
endif()
if(DISABLE_CUDA_SAFETY)
    target_compile_options(pennylane_lightning_gpu INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUDA_UNSAFE>)
endif()

if(ENABLE_WARNINGS)
    target_compile_options(pennylane_lightning_gpu INTERFACE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
    )
endif()

if(ENABLE_NATIVE)
    message(STATUS "ENABLE_NATIVE is ON. Using -march=native")
    add_compile_options(-march=native)
    target_compile_options(pennylane_lightning_gpu INTERFACE -march=native)
    target_compile_options(lightning_gpu_qubit_ops PRIVATE -march=native)
endif()

if (BUILD_TESTS)
    enable_testing()
endif()
