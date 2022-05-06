project(lightning_gpu_interfaces LANGUAGES CXX CUDA)

###############################################################################
# Include all nested sources directories
###############################################################################
if (ENABLE_INTERFACE_TORCH)
    add_subdirectory(torch)
endif()