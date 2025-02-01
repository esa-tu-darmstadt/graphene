#include "libgraphene/matrix/solver/iterative-refinement/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace {
template <typename T>
struct double_of {};
template <>
struct double_of<float> {
  using type = double;
};
template <typename T>
using double_of_v = typename double_of<T>::type;
}  // namespace

namespace graphene::matrix::solver::iterativerefinement {

SolverStats Solver::solve(Tensor& x, Tensor& b) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("IterativeRefinementSolver");

  const Matrix& A = this->matrix();
  if (!A.isVectorCompatible(x, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false))
    throw std::runtime_error("b must be vector without halo cells.");

  // Extended precision can be specified, working precision is always the
  // precision of x
  TypeRef extendedPrecision = config_->extendedPrecisionType
                                  ? config_->extendedPrecisionType
                                  : x.type();
  TypeRef workingPrecision = x.type();

  spdlog::debug(
      "Mixed-Precision Iterative Refinement Solver. Working precision: {}, "
      "Extended precision: {}",
      workingPrecision->str(), extendedPrecision->str());

  SolverStats stats(name(), config_->norm, A.numTiles());

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, b);

  // Calculate the initial residual in extended precision and store in working
  // precision
  Tensor residual = A.residual(x, b, workingPrecision, extendedPrecision);
  stats.initialResidual = A.vectorNorm(config_->norm, residual);
  stats.finalResidual = stats.initialResidual;

  // Allocate a vector of the extended precision type
  Tensor phi = x.cast(extendedPrecision);

  // Reuse the input vector as the correction
  Tensor& correction = x;

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations);

  cf::While(!terminate, [&]() {
    stats.iterations = stats.iterations + 1;

    // Reset the initial guess to zero if the inner solver uses it
    if (innerSolver_->usesInitialGuess()) correction = 0;

    // Solve the correction
    innerSolver_->solve(correction, residual);

    // Update the solution
    phi = phi + correction.cast(extendedPrecision);

    // Recalculate the residual in extended precision and store in working
    // precision
    residual = A.residual(phi, b, workingPrecision, extendedPrecision);

    // Check for convergence
    stats.finalResidual = A.vectorNorm(config_->norm, residual);
    stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                           config_->relResidual);

    if (config_->printPerformanceEachIteration) stats.print();
  });

  // Cast the result back to the working precision
  x = phi.cast(workingPrecision);

  if (config_->printPerformanceAfterSolve) stats.print();

  return stats;
}
Solver::Solver(const Matrix& matrix, std::shared_ptr<Configuration> config)
    : solver::Solver(matrix),
      config_(std::move(config)),
      innerSolver_(
          solver::Solver::createSolver(this->matrix(), config_->innerSolver)) {}

}  // namespace graphene::matrix::solver::iterativerefinement