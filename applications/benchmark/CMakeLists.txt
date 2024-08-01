add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark Graphene spdlog::spdlog_header_only CLI11::CLI11)