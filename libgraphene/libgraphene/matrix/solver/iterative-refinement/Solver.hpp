#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Configuration.hpp"
namespace graphene::matrix::solver::iterativerefinement {
class Solver : public solver::Solver {
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<solver::Solver> innerSolver_;

  template <DataType ExtendedPrecisionType>
  SolverStats solveMixedPrecision(Tensor& x, Tensor& b) const;
  SolverStats solveSinglePrecision(Tensor& x, Tensor& b) const;

 public:
  Solver(const Matrix& matrix, std::shared_ptr<Configuration> config);

  SolverStats solve(Tensor& x, Tensor& b) override;

  std::string name() const override { return "IterativeRefinement"; }
  bool usesInitialGuess() const override { return true; }
};
}  // namespace graphene::matrix::solver::iterativerefinement