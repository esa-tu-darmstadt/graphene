add_subdirectory(libs/json)
add_subdirectory(libs/twofloat)
add_subdirectory(libs/fast_matrix_market)

add_compile_definitions(LIBTWOFLOAT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/libs/twofloat/include")

add_subdirectory(libgraphene)