add_library(GrapheneCodeDSL 
    "CodeGen.cpp" 
    "Operators.cpp" 
    "VertexTypes.cpp"
    "Value.cpp"
    "ControlFlow.cpp"
    "Execute.cpp"
    "Vertex.cpp"
    "Function.cpp"
    "types/ArrayType.cpp"
)
target_link_libraries(GrapheneCodeDSL PUBLIC GrapheneCommon GrapheneDSLCommon GrapheneUtil)
target_link_libraries(GrapheneCodeDSL PRIVATE fmt::fmt)

add_library(graphene::code-dsl ALIAS GrapheneCodeDSL)