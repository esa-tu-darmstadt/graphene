#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Configuration.hpp"

namespace graphene::matrix::solver::pbicgstab {
class Solver : public solver::Solver {
  std::shared_ptr<Configuration> config_;
  std::shared_ptr<solver::Solver> preconditioner_;

 public:
  Solver(const Matrix& matrix, std::shared_ptr<Configuration> config);

  SolverStats solve(Tensor& x, Tensor& b) override;

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