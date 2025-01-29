add_subdirectory(common)
add_subdirectory(code)
add_subdirectory(tensor)

# Add tests
add_executable(GrapheneDSLTests tensor/tests/TensorTests.cpp)
target_link_libraries(GrapheneDSLTests GrapheneTensorDSL GrapheneUtil GrapheneCommon gtest_main)
gtest_discover_tests(GrapheneDSLTests)