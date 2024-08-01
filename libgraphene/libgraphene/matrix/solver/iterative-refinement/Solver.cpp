#include "libgraphene/matrix/solver/iterative-refinement/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/ControlFlow.hpp"
#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::iterativerefinement {

template <DataType Type>
SolverStats Solver<Type>::solveMixedPrecision(Value<Type>& x,
                                              Value<Type>& b) const {
  GRAPHENE_TRACEPOINT();
  spdlog::debug("Solving with mixed precision iterative refinement");
  DebugInfo di("GaussSeidelSolver");
  auto& program = Context::program();

  const Matrix<Type>& A = this->matrix();
  SolverStats stats(name(), config_->norm, A.numTiles());

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, b);

  // Calculate the initial residual
  Value<Type> residual = A.residual(x, b);
  stats.initialResidual = A.vectorNorm(config_->norm, residual);
  stats.finalResidual = stats.initialResidual;

  // Create a double precision vector
  Value<double> xDouble = x.template cast<double>();

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations);

  cf::While(!terminate, [&]() {
    stats.iterations = stats.iterations + 1;

    // Reset the initial guess to zero if the inner solver uses it
    if (innerSolver_->usesInitialGuess()) x = 0;

    // Solve the correction
    innerSolver_->solve(x, residual);

    // Update the solution
    xDouble = ops::Add(xDouble, x);

    // Recalculate the residual
    residual = A.residual(xDouble, b);

    // Check for convergence
    stats.finalResidual = A.vectorNorm(config_->norm, residual);
    stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                           config_->relResidual);

    if (config_->printPerformanceEachIteration) stats.print();
  });
  x = xDouble.cast<float>();

  if (config_->printPerformanceAfterSolve) stats.print();

  return stats;
}

template <DataType Type>
SolverStats Solver<Type>::solve(Value<Type>& x, Value<Type>& b) {
  GRAPHENE_TRACEPOINT();

  if (!innerSolver_)
    innerSolver_ = solver::Solver<Type>::createSolver(this->matrix(),
                                                      config_->innerSolver);

  const Matrix<Type>& A = this->matrix();
  if (!A.isVectorCompatible(x, true, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false, true))
    throw std::runtime_error("b must be vector without halo cells.");

  if (config_->mixedPrecision) return solveMixedPrecision(x, b);

  throw std::runtime_error("Only mixed precision is currently supported.");
}

template class Solver<float>;
}  // namespace graphene::matrix::solver::iterativerefinement