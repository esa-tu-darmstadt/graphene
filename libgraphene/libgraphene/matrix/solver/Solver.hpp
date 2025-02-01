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

#include <poplar/DebugContext.hpp>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix {
class Matrix;

namespace solver {
class SolverStats;
class Solver {
  const Matrix& matrix_;

 public:
  explicit Solver(const Matrix& matrix) : matrix_(matrix) {}
  virtual ~Solver() = default;

  const Matrix& matrix() const { return matrix_; }

  virtual SolverStats solve(Tensor& x, Tensor& b) = 0;

  virtual std::string name() const = 0;
  virtual bool usesInitialGuess() const = 0;

  static std::unique_ptr<Solver> createSolver(
      const Matrix& matrix, std::shared_ptr<Configuration> config);

 protected:
  bool shouldUseMulticolor(MultiColorMode mode) const;
};
};  // namespace solver
}  // namespace graphene::matrix