add_library(GrapheneCommon SHARED "Type.cpp")
target_link_libraries(GrapheneCommon PUBLIC GrapheneHeaders)

add_subdirectory(tests)