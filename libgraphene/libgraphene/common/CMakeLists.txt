add_library(GrapheneCommon "Type.cpp")
target_link_libraries(GrapheneCommon PUBLIC GrapheneHeaders spdlog::spdlog twofloat::twofloat fmt::fmt)

add_library(graphene::common ALIAS GrapheneCommon)

add_subdirectory(tests)