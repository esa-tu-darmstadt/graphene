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
target_link_libraries(GrapheneCodeDSL PRIVATE GrapheneCommon GrapheneDSLCommon GrapheneUtil popops)