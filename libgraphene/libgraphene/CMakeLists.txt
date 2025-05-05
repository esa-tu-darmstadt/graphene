include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

find_package(GTest REQUIRED)
include(GoogleTest)

link_libraries(poplar)

add_subdirectory(codelet)
add_subdirectory(common)
add_subdirectory(matrix)
add_subdirectory(dsl)
add_subdirectory(util)

add_library(GrapheneHeaders INTERFACE)
target_include_directories(GrapheneHeaders INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
# Collect all public headers
file(GLOB_RECURSE GRAPHENE_PUBLIC_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)
target_include_directories(GrapheneHeaders INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_sources(GrapheneHeaders PUBLIC FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
    FILES
        ${GRAPHENE_PUBLIC_HEADERS}
)

add_library(Graphene INTERFACE)
target_link_libraries(Graphene INTERFACE GrapheneCommon GrapheneUtil GrapheneTensorDSL GrapheneCodeDSL GrapheneMatrix)

# Add a convenience target to build all the tests
add_custom_target(GrapheneTests)
add_dependencies(GrapheneTests GrapheneCommonTests GrapheneDSLTests)

# Install library targets
install(TARGETS GrapheneCommon 
  GrapheneUtil 
  GrapheneTensorDSL 
  GrapheneCodeDSL 
  GrapheneDSLCommon 
  GrapheneMatrix 
  GrapheneHeaders 
  GrapheneCodelet
  FILE_SET HEADERS
)