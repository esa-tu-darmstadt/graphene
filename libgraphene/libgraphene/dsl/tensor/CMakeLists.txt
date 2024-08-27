add_library(GrapheneTensorDSL SHARED "RemoteTensor.cpp" 
    "Tensor.cpp" 
    "Operators.cpp" 
    "Expression.cpp" 
    "HostTensor.cpp" 
    "ControlFlow.cpp" 
)
target_link_libraries(GrapheneTensorDSL PRIVATE GrapheneCommon GrapheneUtil GrapheneCodeDSL popops)