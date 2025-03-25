add_library(GrapheneCommon SHARED "Type.cpp")
target_link_libraries(GrapheneCommon PUBLIC GrapheneHeaders)

add_library(Graphene::Common ALIAS GrapheneCommon)

add_subdirectory(tests)