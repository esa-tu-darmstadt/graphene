
# Use Conan-provided CLI11 instead of embedded library
find_package(CLI11 REQUIRED)

add_subdirectory(playground)
add_subdirectory(benchmark)