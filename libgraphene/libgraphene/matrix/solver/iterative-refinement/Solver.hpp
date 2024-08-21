#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Configuration.hpp"
namespace graphene::matrix::solver::iterativerefinement {
template <DataType Type>
class Solver : public solver::Solver<Type> {
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<solver::Solver<Type>> innerSolver_;

  template <DataType ExtendedPrecisionType>
  SolverStats solveMixedPrecision(Value<Type>& x, Value<Type>& b) const;
  SolverStats solveSinglePrecision(Value<Type>& x, Value<Type>& b) const;

 public:
  Solver(const Matrix<Type>& matrix, std::shared_ptr<Configuration> config);

  SolverStats solve(Value<Type>& x, Value<Type>& b) override;

  std::string name() const override { return "IterativeRefinement"; }
  bool usesInitialGuess() const override { return true; }
};
}  // namespace graphene::matrix::solver::iterativerefinement