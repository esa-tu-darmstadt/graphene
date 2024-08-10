#include "libgraphene/matrix/solver/restarter/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/ControlFlow.hpp"
#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/restarter/Configuration.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::restarter {

template <DataType Type>
SolverStats Solver<Type>::solve(Value<Type>& x, Value<Type>& b) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RestartSolver");

  spdlog::trace("Building restarter for solver {} with {} restarts",
                config_->innerSolver->solverName(), config_->maxRestarts);

  if (!innerSolver_)
    innerSolver_ = solver::Solver<Type>::createSolver(this->matrix(),
                                                      config_->innerSolver);

  Value<int> iterations(1);
  SolverStats stats = innerSolver_->solve(x, b);

  cf::While(iterations < config_->maxRestarts && !stats.converged, [&]() {
    iterations = iterations + 1;
    stats = innerSolver_->solve(x, b);
  });

  return stats;
}

template class Solver<float>;
}  // namespace graphene::matrix::solver::restarter