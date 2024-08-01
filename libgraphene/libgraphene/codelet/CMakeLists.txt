add_subdirectory(ipu)

add_library(GrapheneCodelet SHARED "Codelet.cpp")
target_link_libraries(GrapheneCodelet PRIVATE popops )

add_dependencies(GrapheneCodelet GrapheneCodeletsIPU)