#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>
#include <utility>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"
namespace graphene::matrix::solver::gaussseidel {
template <DataType Type>
class Solver : public solver::Solver<Type> {
  std::shared_ptr<const gaussseidel::Configuration> config_;
  bool solveMulticolor_;

  void solveIteration(Value<Type>& x, Value<Type>& b) const;
  void solveIterationCSR(Value<Type>& x, Value<Type>& b) const;

 public:
  Solver(const Matrix<Type>& matrix,
         std::shared_ptr<const gaussseidel::Configuration> config);

  SolverStats solve(Value<Type>& x, Value<Type>& b) override;

  std::string name() const override { return "GaussSeidel"; }
  bool usesInitialGuess() const override { return true; }
};
}  // namespace graphene::matrix::solver::gaussseidel