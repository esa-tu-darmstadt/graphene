add_library(GrapheneDSLCommon SHARED 
    "details/Expressions.cpp"
)
target_link_libraries(GrapheneDSLCommon PUBLIC GrapheneCommon)

add_library(Graphene::DSL::Common ALIAS GrapheneDSLCommon)