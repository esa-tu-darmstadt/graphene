add_subdirectory(ipu)

add_library(GrapheneCodelet "Codelet.cpp")
target_link_libraries(GrapheneCodelet PRIVATE popops )
target_link_libraries(GrapheneCodelet PUBLIC GrapheneHeaders)

# FIXME: Fix for installation
target_compile_definitions(GrapheneCodelet PRIVATE GRAPHENE_PRECOMPILED_CODELETS_PATH="${CMAKE_CURRENT_BINARY_DIR}/ipu/GrapheneCodeletsIPU.gp")

add_library(Graphene::Codelet ALIAS GrapheneCodelet)

add_dependencies(GrapheneCodelet GrapheneCodeletsIPU)