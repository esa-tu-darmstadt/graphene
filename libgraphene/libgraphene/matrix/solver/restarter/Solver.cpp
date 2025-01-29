#include "libgraphene/matrix/solver/restarter/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/restarter/Configuration.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::restarter {
Solver::Solver(const Matrix& matrix, std::shared_ptr<Configuration> config)
    : solver::Solver(matrix),
      config_(std::move(config)),
      innerSolver_(
          solver::Solver::createSolver(this->matrix(), config_->innerSolver)) {}

SolverStats Solver::solve(Tensor& x, Tensor& b) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RestartSolver");

  spdlog::trace("Building restarter for solver {} with {} restarts",
                config_->innerSolver->solverName(), config_->maxRestarts);

  Tensor iterations = Tensor::withInitialValue((uint32_t)0);
  SolverStats stats = innerSolver_->solve(x, b);

  cf::While(iterations < config_->maxRestarts && !stats.converged, [&]() {
    iterations = iterations + 1;
    stats = innerSolver_->solve(x, b);
  });

  return stats;
}

}  // namespace graphene::matrix::solver::restarter