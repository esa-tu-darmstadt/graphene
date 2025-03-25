add_library(GrapheneCodeDSL SHARED 
    "CodeGen.cpp" 
    "Operators.cpp" 
    "VertexTypes.cpp"
    "Value.cpp"
    "ControlFlow.cpp"
    "Execute.cpp"
    "Vertex.cpp"
    "Function.cpp"
)
target_link_libraries(GrapheneCodeDSL PUBLIC GrapheneCommon GrapheneDSLCommon GrapheneUtil)

add_library(Graphene::DSL::Code ALIAS GrapheneCodeDSL)