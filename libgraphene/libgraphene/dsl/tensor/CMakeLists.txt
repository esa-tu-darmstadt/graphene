add_library(GrapheneTensorDSL SHARED "RemoteTensor.cpp" 
    "Tensor.cpp" 
    "Operators.cpp" 
    "Expression.cpp" 
    "HostTensor.cpp" 
    "ControlFlow.cpp" 
    "details/Expressions.cpp"
)
target_link_libraries(GrapheneTensorDSL PRIVATE GrapheneCommon GrapheneDSLCommon GrapheneUtil GrapheneCodeDSL popops)

# Contains functions inlined into the operators
target_link_libraries(GrapheneTensorDSL PUBLIC GrapheneDSLCommon)