#pragma once

#include <memory>
#include <optional>
#include <poplar/DebugContext.hpp>
#include <variant>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/ilu/Configuration.hpp"
namespace graphene::matrix::solver::ilu {
class Solver : public solver::Solver {
  std::shared_ptr<Configuration> config_;

  struct CRSFactorization {
    // The inverse of the diagonal of the factorized matrix
    std::unique_ptr<Tensor> factorizedInverseDiag;

    // The off-diagonal of the factorized matrix
    std::unique_ptr<Tensor> factorizedOffDiag;
  };

  CRSFactorization factorization_;

  bool solveMulticolor_;
  bool factorizeMulticolor_;

  void factorize();

  void solveCRS(Tensor& x, Tensor& b);

  void factorizeCRS_ILU();
  void factorizeCRS_DILU();

 public:
  Solver(const Matrix& matrix, std::shared_ptr<Configuration> config);

  SolverStats solve(Tensor& x, Tensor& b) override;

  std::string name() const override {
    return config_->diagonalBased ? "DILU" : "ILU(0)";
  }
  bool usesInitialGuess() const override { return false; }
};
}  // namespace graphene::matrix::solver::ilu