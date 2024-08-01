#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>
#include <utility>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Configuration.hpp"

namespace graphene::matrix::solver::pbicgstab {
template <DataType Type>
class Solver : public solver::Solver<Type> {
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<solver::Solver<Type>> preconditioner_;

 public:
  Solver(const Matrix<Type>& matrix, std::shared_ptr<Configuration> config)
      : solver::Solver<Type>(matrix), config_(std::move(config)) {}

  SolverStats solve(Value<Type>& x, Value<Type>& b) override;

  std::string name() const override {
    if (preconditioner_) {
      return "PBiCGStab (" + preconditioner_->name() + ")";
    } else {
      return "BiCGStab";
    }
  }
  bool usesInitialGuess() const override { return true; }
};
}  // namespace graphene::matrix::solver::pbicgstab