#include "libgraphene/matrix/solver/pbicgstab/Solver.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/dsl/ControlFlow.hpp"
#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::pbicgstab {

#define PBICGSTAB_VERBOSE_PRINT(x) \
  if (config_->verbose) {          \
    x.print(#x);                   \
  }

template <DataType Type>
Solver<Type>::Solver(const Matrix<Type>& matrix,
                     std::shared_ptr<Configuration> config)
    : solver::Solver<Type>(matrix), config_(std::move(config)) {
  spdlog::debug("Creating PBiCGStab solver");
  if (config_->preconditioner)
    preconditioner_ = solver::Solver<Type>::createSolver(
        this->matrix(), config_->preconditioner);
}

template <DataType Type>
SolverStats Solver<Type>::solve(Tensor<Type>& x, Tensor<Type>& b) {
  GRAPHENE_TRACEPOINT();

  const Matrix<Type>& A = this->matrix();
  if (!A.isVectorCompatible(x, true, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false, true))
    throw std::runtime_error("b must be vector without halo cells.");

  DebugInfo di("PBiCGStabSolver");

  auto& program = Context::program();

  SolverStats stats(this->name(), config_->norm, A.numTiles());

  // Calculate the initial residual field rA
  Tensor<Type> rA = A.residual(x, b);

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, x);

  // Calculate the initial residual
  stats.initialResidual = A.vectorNorm(config_->norm, rA);
  stats.finalResidual = stats.initialResidual;

  Tensor<Type> rA0rAold(0);
  Tensor<Type> alpha(0);
  Tensor<Type> omega(0);

  // Store the initial residual
  Tensor<Type> rA0 = rA;
  Tensor<Type> pA = rA;
  Tensor<Type> AyA = A.template createUninitializedVector<Type>(false);

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations) || stats.singular;

  cf::While(!terminate, [&]() {
    Tensor<Type> rA0rA = Tensor<Type>(rA0 * rA).reduce();
    PBICGSTAB_VERBOSE_PRINT(rA0rA);

    // Check for rA0rA for singularity
    stats.checkSingularity(ops::Abs(rA0rA));
    cf::If(!stats.singular, [&]() {
      // Calculate pA if needed
      cf::If(stats.iterations != 0, [&]() {
        auto beta = (rA0rA / rA0rAold) * (alpha / omega);
        pA = rA + beta * (pA - omega * AyA);
      });

      // Precondition pA
      // Calculate yA = M^-1 * pA by solving M * yA = pA
      Tensor<Type> yA = A.template createUninitializedVector<Type>(true);
      if (preconditioner_) {
        if (preconditioner_->usesInitialGuess()) {
          yA = 0;
        }
        preconditioner_->solve(yA, pA);
      } else {
        A.stripHaloCellsFromVector(yA) = pA;
      }

      PBICGSTAB_VERBOSE_PRINT(pA);
      PBICGSTAB_VERBOSE_PRINT(yA);

      // Update AyA
      AyA = A * yA;

      // Update alpha
      Tensor<Type> rA0AyA = Tensor<Type>(rA0 * AyA).reduce();
      alpha = rA0rA / rA0AyA;

      PBICGSTAB_VERBOSE_PRINT(rA0AyA);
      PBICGSTAB_VERBOSE_PRINT(alpha);

      // Calculate sA
      Tensor<Type> sA = rA - alpha * AyA;

      // The original BiCGStab algorithm would check for convergence here, but
      // we skip it to prevent bloating up the graph

      // Optionally precondition sA into zA
      Tensor<Type> zA = A.template createUninitializedVector<Type>(true);
      if (preconditioner_) {
        if (preconditioner_->usesInitialGuess()) {
          zA = 0;
        }
        preconditioner_->solve(zA, sA);
      } else {
        A.stripHaloCellsFromVector(zA) = sA;
      }

      PBICGSTAB_VERBOSE_PRINT(sA);
      PBICGSTAB_VERBOSE_PRINT(zA);

      // Calculate tA
      Tensor<Type> tA = A * zA;
      Tensor<Type> tAtA = Tensor<Type>(tA * tA).reduce();

      // Update omega from tA and sA
      omega = Tensor<Type>(tA * sA).reduce() / tAtA;

      PBICGSTAB_VERBOSE_PRINT(tA);
      PBICGSTAB_VERBOSE_PRINT(tAtA);
      PBICGSTAB_VERBOSE_PRINT(omega);

      stats.checkSingularity(ops::Abs(omega));

      // Update psi and rA if omega and tAtA are not singular
      cf::If(!stats.singular, [&]() {
        // Update solution
        x = x + alpha * yA + omega * zA;
        PBICGSTAB_VERBOSE_PRINT(x);

        // Update residual
        rA = sA - omega * tA;
        PBICGSTAB_VERBOSE_PRINT(rA);

        stats.finalResidual = A.vectorNorm(config_->norm, rA);
        stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                               config_->relResidual);

        // Store previous rA0rA
        rA0rAold = rA0rA;

        // Increment the iteration counter
        stats.iterations = stats.iterations + 1;

        // Print performance if requested
        if (config_->printPerformanceEachIteration) {
          stats.print();
        }
      });
    });
  });

  if (config_->printPerformanceAfterSolve) {
    stats.print();
  }

  return stats;
}

template class Solver<float>;
}  // namespace graphene::matrix::solver::pbicgstab