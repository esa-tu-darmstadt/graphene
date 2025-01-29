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