project(lightning_gpu_interfaces LANGUAGES CXX CUDA)

###############################################################################
# Include all nested sources directories
###############################################################################
if (ENABLE_INTERFACES)
    add_subdirectory(torch)
endif()