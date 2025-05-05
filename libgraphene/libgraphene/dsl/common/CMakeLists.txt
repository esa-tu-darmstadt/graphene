add_library(GrapheneDSLCommon 
    "details/Expressions.cpp"
)
target_link_libraries(GrapheneDSLCommon PUBLIC GrapheneCommon)

add_library(graphene::dsl-common ALIAS GrapheneDSLCommon)