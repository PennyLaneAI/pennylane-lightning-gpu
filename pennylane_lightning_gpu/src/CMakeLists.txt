project(lightning_gpu_components LANGUAGES CXX CUDA)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS      algorithms;
                           simulator;
                           util;
)
foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (PLLGPU_BUILD_TESTS)
    add_subdirectory("tests" "tests")
endif()