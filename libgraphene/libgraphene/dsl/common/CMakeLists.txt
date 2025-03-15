add_library(GrapheneDSLCommon SHARED 
    "details/Expressions.cpp"
)
target_link_libraries(GrapheneDSLCommon PUBLIC GrapheneCommon)