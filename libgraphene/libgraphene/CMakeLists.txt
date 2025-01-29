include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

INCLUDE(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG "v1.14.0"
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

link_libraries(twofloat poplar)

add_subdirectory(codelet)
add_subdirectory(common)
add_subdirectory(matrix)
add_subdirectory(dsl)
add_subdirectory(util)

add_library(Graphene INTERFACE)
target_link_libraries(Graphene INTERFACE GrapheneCommon GrapheneUtil GrapheneTensorDSL GrapheneCodeDSL GrapheneMatrix nlohmann_json::nlohmann_json metis)
target_include_directories(Graphene INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add a convenience target to build all the tests
add_custom_target(GrapheneTests)
add_dependencies(GrapheneTests GrapheneCommonTests GrapheneDSLTests)