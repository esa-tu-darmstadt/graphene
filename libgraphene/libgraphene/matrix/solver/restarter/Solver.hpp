#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/restarter/Configuration.hpp"
namespace graphene::matrix::solver::restarter {
template <DataType Type>
class Solver : public solver::Solver<Type> {
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<solver::Solver<Type>> innerSolver_;

 public:
  Solver(const Matrix<Type>& matrix, std::shared_ptr<Configuration> config);

  SolverStats solve(Value<Type>& x, Value<Type>& b) override;

  std::string name() const override {
    return fmt::format("Restarter of {}", innerSolver_->name());
  }
  bool usesInitialGuess() const override {
    return innerSolver_->usesInitialGuess();
  }
};
}  // namespace graphene::matrix::solver::restarter