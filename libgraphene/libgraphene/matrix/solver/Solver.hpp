#pragma once

#include <poplar/DebugContext.hpp>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/Tensor.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
namespace graphene::matrix {
template <DataType Type>
class Matrix;

namespace solver {
template <DataType Type>
class Solver {
  const Matrix<Type>& matrix_;

 public:
  explicit Solver(const Matrix<Type>& matrix) : matrix_(matrix) {}
  virtual ~Solver() = default;

  const Matrix<Type>& matrix() const { return matrix_; }

  virtual SolverStats solve(Tensor<Type>& x, Tensor<Type>& b) = 0;

  virtual std::string name() const = 0;
  virtual bool usesInitialGuess() const = 0;

  static std::unique_ptr<Solver<Type>> createSolver(
      const Matrix<Type>& matrix, std::shared_ptr<Configuration> config);

 protected:
  bool shouldUseMulticolor(MultiColorMode mode) const;
};
};  // namespace solver
}  // namespace graphene::matrix