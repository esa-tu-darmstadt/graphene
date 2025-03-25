add_library(GrapheneTensorDSL SHARED "RemoteTensor.cpp" 
    "Tensor.cpp" 
    "Operators.cpp" 
    "Expression.cpp" 
    "HostTensor.cpp" 
    "ControlFlow.cpp" 
    "details/Expressions.cpp"
)
target_link_libraries(GrapheneTensorDSL PUBLIC GrapheneCommon GrapheneUtil GrapheneCodeDSL popops)

add_library(Graphene::DSL::Tensor ALIAS GrapheneTensorDSL)

# Contains functions inlined into the operators
target_link_libraries(GrapheneTensorDSL PUBLIC GrapheneDSLCommon)