/*
 * Graphene Linear Algebra Framework for Intelligence Processing Units.
 * Copyright (C) 2025 Embedded Systems and Applications, TU Darmstadt.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "libgraphene/matrix/solver/pbicgstab/Solver.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
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

Solver::Solver(const Matrix& matrix, std::shared_ptr<Configuration> config)
    : solver::Solver(matrix), config_(std::move(config)) {
  spdlog::debug("Creating PBiCGStab solver");
  if (config_->preconditioner)
    preconditioner_ =
        solver::Solver::createSolver(this->matrix(), config_->preconditioner);
}

SolverStats Solver::solve(Tensor& x, Tensor& b) {
  GRAPHENE_TRACEPOINT();

  const Matrix& A = this->matrix();
  if (!A.isVectorCompatible(x, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false))
    throw std::runtime_error("b must be vector without halo cells.");

  DebugInfo di("PBiCGStabSolver");

  auto& program = Context::program();

  SolverStats stats(this->name(), config_->norm, A.numTiles());
  TypeRef workingType = config_->workingType ? config_->workingType : x.type();

  // Calculate the initial residual field rA
  Tensor rA = A.residual(x, b, workingType);

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, x);

  // Calculate the initial residual
  stats.initialResidual = A.vectorNorm(config_->norm, rA);
  stats.finalResidual = stats.initialResidual;

  Tensor rA0rAold =
      Tensor::uninitialized(workingType, DistributedShape::scalar());
  Tensor alpha = Tensor::uninitialized(workingType, DistributedShape::scalar());
  Tensor omega = Tensor::uninitialized(workingType, DistributedShape::scalar());

  // Store the initial residual
  Tensor rA0 = rA;
  Tensor pA = rA;
  Tensor AyA = A.createUninitializedVector(workingType, false);

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations) || stats.singular;

  cf::While(!terminate, [&]() {
    Tensor rA0rA = (rA0 * rA).reduce();
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
      Tensor yA = A.createUninitializedVector(workingType, true);
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
      Tensor rA0AyA = (rA0 * AyA).reduce();
      alpha = rA0rA / rA0AyA;

      PBICGSTAB_VERBOSE_PRINT(rA0AyA);
      PBICGSTAB_VERBOSE_PRINT(alpha);

      // Calculate sA
      Tensor sA = rA - alpha * AyA;

      // The original BiCGStab algorithm would check for convergence here, but
      // we skip it to prevent bloating up the graph

      // Optionally precondition sA into zA
      Tensor zA = A.createUninitializedVector(workingType, true);
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
      Tensor tA = A * zA;
      Tensor tAtA = (tA * tA).reduce();

      // Update omega from tA and sA
      omega = (tA * sA).reduce() / tAtA;

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

}  // namespace graphene::matrix::solver::pbicgstab