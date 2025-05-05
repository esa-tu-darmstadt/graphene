add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark Graphene spdlog::spdlog CLI11::CLI11)