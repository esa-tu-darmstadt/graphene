#include "libgraphene/matrix/solver/restarter/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/restarter/Configuration.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::restarter {
template <DataType Type>
Solver<Type>::Solver(const Matrix<Type>& matrix,
                     std::shared_ptr<Configuration> config)
    : solver::Solver<Type>(matrix),
      config_(std::move(config)),
      innerSolver_(solver::Solver<Type>::createSolver(this->matrix(),
                                                      config_->innerSolver)) {}

template <DataType Type>
SolverStats Solver<Type>::solve(Tensor<Type>& x, Tensor<Type>& b) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RestartSolver");

  spdlog::trace("Building restarter for solver {} with {} restarts",
                config_->innerSolver->solverName(), config_->maxRestarts);

  Tensor<int> iterations(1);
  SolverStats stats = innerSolver_->solve(x, b);

  cf::While(iterations < config_->maxRestarts && !stats.converged, [&]() {
    iterations = iterations + 1;
    stats = innerSolver_->solve(x, b);
  });

  return stats;
}

template class Solver<float>;
}  // namespace graphene::matrix::solver::restarter