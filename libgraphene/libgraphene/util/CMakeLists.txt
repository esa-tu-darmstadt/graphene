add_library(GrapheneUtil "Tracepoint.cpp" "DebugInfo.cpp" "Context.cpp" "Runtime.cpp" "VertexCache.cpp" "PoplarHelpers.cpp") 
target_link_libraries(GrapheneUtil PUBLIC GrapheneCommon GrapheneCodelet pvti)
target_link_libraries(GrapheneUtil PRIVATE poputil)

add_library(graphene::util ALIAS GrapheneUtil)