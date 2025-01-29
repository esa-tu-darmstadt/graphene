add_library(GrapheneUtil SHARED "Tracepoint.cpp" "DebugInfo.cpp" "Context.cpp" "Runtime.cpp" "PoplarHelpers.cpp") 
target_link_libraries(GrapheneUtil PRIVATE GrapheneCommon GrapheneCodelet)