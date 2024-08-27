add_library(GrapheneCodeDSL SHARED 
    "CodeGen.cpp" 
    "Operators.cpp" 
    "VertexTypes.cpp"
    "Value.cpp"
    "ControlFlow.cpp"
)
target_link_libraries(GrapheneCodeDSL PRIVATE GrapheneCommon GrapheneUtil popops)