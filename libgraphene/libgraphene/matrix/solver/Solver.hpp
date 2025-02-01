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