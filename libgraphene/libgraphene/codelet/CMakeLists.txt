add_subdirectory(ipu)

add_library(GrapheneCodelet SHARED "Codelet.cpp")
target_link_libraries(GrapheneCodelet PRIVATE popops )
target_link_libraries(GrapheneCodelet PUBLIC GrapheneHeaders)

add_library(Graphene::Codelet ALIAS GrapheneCodelet)

add_dependencies(GrapheneCodelet GrapheneCodeletsIPU)