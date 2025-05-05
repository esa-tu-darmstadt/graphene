cmake_minimum_required(VERSION 3.25)
project("graphene" LANGUAGES CXX)
include(CTest)

find_package(poplar REQUIRED)
get_target_property(POPC_EXECUTABLE popc_bin LOCATION)

find_package(nlohmann_json REQUIRED)
find_package(twofloat CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(metis CONFIG REQUIRED)
find_package(fast_matrix_market CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include(AddGPLibrary)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fno-limit-debug-info" COMPILER_SUPPORTS_NO_LIMIT_DEBUG_INFO)
if(COMPILER_SUPPORTS_NO_LIMIT_DEBUG_INFO)
  add_compile_options(-fno-limit-debug-info)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

add_subdirectory(libgraphene)
add_subdirectory(applications)