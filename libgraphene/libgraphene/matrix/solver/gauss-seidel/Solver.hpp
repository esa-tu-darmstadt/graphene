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

#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>
#include <utility>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"
namespace graphene::matrix::solver::gaussseidel {
class Solver : public solver::Solver {
  std::shared_ptr<const gaussseidel::Configuration> config_;
  bool solveMulticolor_;

  void solveIteration(Tensor& x, Tensor& b) const;
  void solveIterationCSR(Tensor& x, Tensor& b) const;

 public:
  Solver(const Matrix& matrix,
         std::shared_ptr<const gaussseidel::Configuration> config);

  SolverStats solve(Tensor& x, Tensor& b) override;

  std::string name() const override { return "GaussSeidel"; }
  bool usesInitialGuess() const override { return true; }
};
}  // namespace graphene::matrix::solver::gaussseidel