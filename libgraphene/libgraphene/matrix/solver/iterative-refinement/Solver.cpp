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

template <DataType Type>
template <DataType ExtendedPrecisionType>
SolverStats Solver<Type>::solveMixedPrecision(Tensor<Type>& x,
                                              Tensor<Type>& b) const {
  GRAPHENE_TRACEPOINT();
  spdlog::debug("Solving with mixed precision iterative refinement");
  DebugInfo di("IterativeRefinementSolver");
  auto& program = Context::program();

  const Matrix<Type>& A = this->matrix();
  SolverStats stats(name(), config_->norm, A.numTiles());

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, b);

  // Calculate the initial residual
  Tensor<Type> residual = A.residual(x, b);
  stats.initialResidual = A.vectorNorm(config_->norm, residual);
  stats.finalResidual = stats.initialResidual;

  // Create a double precision vector
  Tensor<ExtendedPrecisionType> xDouble =
      x.template cast<ExtendedPrecisionType>();

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations);

  cf::While(!terminate, [&]() {
    stats.iterations = stats.iterations + 1;

    // Reset the initial guess to zero if the inner solver uses it
    if (innerSolver_->usesInitialGuess()) {
      // x = 0;
      cf::Time([&]() { x = 0; }, 0).print("MPIR Reset Initial Guess");
    }

    // Solve the correction
    // innerSolver_->solve(x, residual);
    cf::Time([&]() { innerSolver_->solve(x, residual); }, 0)
        .print("MPIR Inner Solver Cycles");

    // Update the solution
    // xDouble = ops::Add(xDouble, x);
    cf::Time([&]() { xDouble = ops::Add(xDouble, x); }, 0)
        .print("MPIR Add Cycles");

    // Recalculate the mixed-precision residual
    // residual = A.residual(xDouble, b);
    cf::Time([&]() { residual = A.residual(xDouble, b); }, 0)
        .print("MPIR Residual Cylces");

    // Check for convergence
    cf::Time(
        [&]() {
          stats.finalResidual = A.vectorNorm(config_->norm, residual);
          stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                                 config_->relResidual);
        },
        0)
        .print("MPIR Check Convergence Cycles");

    if (config_->printPerformanceEachIteration) stats.print();
  });
  x = xDouble.template cast<Type>();

  if (config_->printPerformanceAfterSolve) stats.print();

  return stats;
}

template <DataType Type>
SolverStats Solver<Type>::solveSinglePrecision(Tensor<Type>& x,
                                               Tensor<Type>& b) const {
  GRAPHENE_TRACEPOINT();
  spdlog::debug("Solving with single precision iterative refinement");
  DebugInfo di("IterativeRefinementSolver");
  auto& program = Context::program();

  const Matrix<Type>& A = this->matrix();
  SolverStats stats(name(), config_->norm, A.numTiles());

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, b);

  // Calculate the initial residual
  Tensor<Type> residual = A.residual(x, b);
  stats.initialResidual = A.vectorNorm(config_->norm, residual);
  stats.finalResidual = stats.initialResidual;

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations);

  cf::While(!terminate, [&]() {
    stats.iterations = stats.iterations + 1;

    Tensor<Type> correction = A.template createUninitializedVector<Type>(true);
    // Reset the initial guess to zero if the inner solver uses it
    if (innerSolver_->usesInitialGuess()) correction = 0;

    // Solve the correction
    innerSolver_->solve(correction, residual);

    // Update the solution
    x = x + correction;

    // Recalculate the residual
    residual = A.residual(x, b);

    // Check for convergence
    stats.finalResidual = A.vectorNorm(config_->norm, residual);
    stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                           config_->relResidual);

    if (config_->printPerformanceEachIteration) stats.print();
  });

  if (config_->printPerformanceAfterSolve) stats.print();

  return stats;
}

template <DataType Type>
SolverStats Solver<Type>::solve(Tensor<Type>& x, Tensor<Type>& b) {
  GRAPHENE_TRACEPOINT();

  const Matrix<Type>& A = this->matrix();
  if (!A.isVectorCompatible(x, true, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false, true))
    throw std::runtime_error("b must be vector without halo cells.");

  if (config_->mixedPrecision) {
    if (config_->useDoubleWordArithmetic) {
      if (!std::is_same_v<Type, float>) {
        throw std::runtime_error(
            "Double word arithmetic is only supported for single precision "
            "solvers.");
      }
      spdlog::trace("Solving with double word arithmetic");
      return solveMixedPrecision<doubleword>(x, b);
    } else {
      spdlog::trace("Solving with double precision");
      return solveMixedPrecision<double_of_v<Type>>(x, b);
    }
  } else
    return solveSinglePrecision(x, b);
}
template <DataType Type>
Solver<Type>::Solver(const Matrix<Type>& matrix,
                     std::shared_ptr<Configuration> config)
    : solver::Solver<Type>(matrix),
      config_(std::move(config)),
      innerSolver_(solver::Solver<Type>::createSolver(this->matrix(),
                                                      config_->innerSolver)) {}

template class Solver<float>;
}  // namespace graphene::matrix::solver::iterativerefinement