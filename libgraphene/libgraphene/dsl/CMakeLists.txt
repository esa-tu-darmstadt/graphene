add_library(GrapheneDSL SHARED 
    "RemoteTensor.cpp" 
    "Tensor.cpp" 
    "Operators.cpp" 
    "Expression.cpp" 
    "HostTensor.cpp" 
    "ControlFlow.cpp" 
    "codelet/CodeGen.cpp" 
    "codelet/Operators.cpp" 
    "codelet/VertexTypes.cpp"
    "codelet/Value.cpp"
    "codelet/ControlFlow.cpp"
)
target_link_libraries(GrapheneDSL PRIVATE GrapheneCommon GrapheneUtil popops)