add_library(GrapheneMatrix SHARED 
    "Norm.cpp"
    "Matrix.cpp"
    "details/crs/CRSMatrix.cpp"
    "details/ldu/LDUMatrix.cpp"
    "details/MatrixBase.cpp"
    "host/HostMatrix.cpp"
    "host/DistributedTileLayout.cpp"
    "host/details/CoordinateFormat.cpp"
    "host/details/Poisson.cpp"
    "host/details/MatrixMarket.cpp"
    "host/details/HostMatrixBase.cpp"
    "host/details/crs/CRSHostMatrix.cpp"
    "solver/Solver.cpp"
    "solver/Configuration.cpp"
    "solver/SolverStats.cpp"
    "solver/gauss-seidel/Solver.cpp"
    "solver/gauss-seidel/Configuration.cpp"
    "solver/ilu/Solver.cpp"
    "solver/ilu/Configuration.cpp"
    "solver/iterative-refinement/Solver.cpp"
    "solver/iterative-refinement/Configuration.cpp"
    "solver/pbicgstab/Solver.cpp"
    "solver/pbicgstab/Configuration.cpp"
    "solver/restarter/Solver.cpp"
    "solver/restarter/Configuration.cpp")
target_link_libraries(GrapheneMatrix PRIVATE fast_matrix_market::fast_matrix_market)
target_link_libraries(GrapheneMatrix PUBLIC GrapheneCommon GrapheneUtil GrapheneTensorDSL nlohmann_json::nlohmann_json)

add_library(Graphene::Matrix ALIAS GrapheneMatrix)