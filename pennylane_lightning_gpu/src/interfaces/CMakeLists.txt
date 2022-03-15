project(lightning_gpu_interfaces LANGUAGES CXX CUDA)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS      torch)
foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()