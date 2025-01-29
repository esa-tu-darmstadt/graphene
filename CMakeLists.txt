cmake_minimum_required(VERSION 3.26)
project("graphene" LANGUAGES CXX)
include(CTest)

find_package(poplar REQUIRED)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include(AddGPLibrary)

add_compile_options(-fno-limit-debug-info)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_subdirectory(libgraphene)
add_subdirectory(applications)