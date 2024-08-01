add_executable(playground playground.cpp)

target_link_libraries(playground Graphene spdlog::spdlog_header_only)