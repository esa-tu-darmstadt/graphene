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

#include "libgraphene/matrix/solver/gauss-seidel/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/code/ControlFlow.hpp"
#include "libgraphene/dsl/code/Operators.hpp"
#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Execute.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::gaussseidel {

Solver::Solver(const Matrix& matrix,
               std::shared_ptr<const gaussseidel::Configuration> config)
    : solver::Solver(matrix),
      config_(std::move(config)),
      solveMulticolor_(this->shouldUseMulticolor(config_->solveMulticolor)) {}

void Solver::solveIterationCSR(Tensor& x, Tensor& b) const {
  const auto& A = this->matrix().getImpl<crs::CRSMatrix>();
  DebugInfo di("GaussSeidelSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  if (solveMulticolor_) {
    if (!A.addressing->coloring.has_value()) {
      throw std::runtime_error(
          "Matrix has no coloring. Cannot use multicolor gauss-seidel");
    }
    spdlog::trace("Using multicolor gauss-seidel");
  } else {
    spdlog::trace("Using sequential gauss-seidel");
  }

  A.exchangeHaloCells(x);

  TypeRef workingType = config_->workingType ? config_->workingType : x.type();

  // FIXME: Solve multicolor
  // FIXME: Use multiple worker threads

  using namespace codedsl;
  Execute(
      [workingType](Value x, Value b, Value rowPtr, Value colInd, Value diag,
                    Value offDiag) {
        // For each row in the matrix
        For(0, rowPtr.size() - 1, 1, [&](Value i) {
          Variable sum = diag[i].cast(workingType);
          // For each non-diagonal element in the row
          For(rowPtr[i], rowPtr[i + 1], 1, [&](Value j) {
            sum -=
                offDiag[j].cast(workingType) * x[colInd[j]].cast(workingType);
          });
          Value newX = (sum / diag[i].cast(workingType));
          x[i] = newX.cast(x[0].type());
        });
      },
      InOut(x), In(b), In(A.addressing->rowPtr), In(A.addressing->colInd),
      In(A.diagonalCoefficients), In(A.offDiagonalCoefficients));
}

void gaussseidel::Solver::solveIteration(Tensor& x, Tensor& b) const {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      solveIterationCSR(x, b);
      break;
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

SolverStats Solver::solve(Tensor& x, Tensor& b) {
  GRAPHENE_TRACEPOINT();
  spdlog::trace("Solving with Gauss-Seidel");

  DebugInfo di("GaussSeidelSolver");
  const Matrix& A = this->matrix();

  if (!A.isVectorCompatible(x, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false))
    throw std::runtime_error("b must be vector without halo cells.");

  auto& program = Context::program();
  TypeRef workingType = config_->workingType ? config_->workingType : x.type();

  SolverStats stats(name(), config_->norm, A.numTiles());

  if (config_->numFixedIterations > 0) {
    // A fixed number of iterations is requested
    // Do not check for convergence
    spdlog::trace("Using a fixed number of iterations for Gauss-Seidel: {}",
                  config_->numFixedIterations);

    cf::Repeat(config_->numFixedIterations, [&]() { solveIteration(x, b); });

    stats.iterations = config_->numFixedIterations;
    return stats;
  }

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, b);

  // Calculate the initial residual
  Tensor initialResidual = A.residual(x, b, workingType);
  stats.initialResidual = A.vectorNorm(config_->norm, initialResidual);
  stats.finalResidual = stats.initialResidual;

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations);

  cf::While(!terminate, [&]() {
    // Sweep the solver
    cf::Repeat(config_->numSweeps, [&]() {
      solveIteration(x, b);
      stats.iterations = stats.iterations + 1;
    });

    // Calculate the residual and check for convergence
    auto currentResidual = A.residual(x, b, workingType);
    stats.finalResidual = A.vectorNorm(config_->norm, currentResidual);
    stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                           config_->relResidual);

    if (config_->printPerformanceEachIteration) stats.print();
  });

  if (config_->printPerformanceAfterSolve) stats.print();

  return stats;
}
}  // namespace graphene::matrix::solver::gaussseidel