include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

link_libraries(twofloat poplar)

add_subdirectory(codelet)
add_subdirectory(common)
add_subdirectory(matrix)
add_subdirectory(dsl)
add_subdirectory(util)

add_library(Graphene INTERFACE)
target_link_libraries(Graphene INTERFACE GrapheneCommon GrapheneUtil GrapheneTensorDSL GrapheneCodeDSL GrapheneMatrix popops nlohmann_json::nlohmann_json metis)
target_include_directories(Graphene INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)