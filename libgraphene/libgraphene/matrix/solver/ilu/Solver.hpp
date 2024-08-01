#pragma once

#include <memory>
#include <poplar/DebugContext.hpp>
#include <utility>

#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/matrix/solver/ilu/Configuration.hpp"
namespace graphene::matrix::solver::ilu {
template <DataType Type>
class Solver : public solver::Solver<Type> {
  std::shared_ptr<Configuration> config_;

  struct CRSFactorization {
    // The inverse of the diagonal of the factorized matrix
    std::unique_ptr<Value<Type>> factorizedInverseDiag;

    // The off-diagonal of the factorized matrix
    std::optional<Value<Type>> factorizedOffDiag;
  };

  std::variant<CRSFactorization> factorization_;

  bool solveMulticolor_;
  bool factorizeMulticolor_;

  void factorize();

  void solveCRS(Value<Type>& x, Value<Type>& b);
  void factorizeCRS();

 public:
  Solver(const Matrix<Type>& matrix, std::shared_ptr<Configuration> config);

  SolverStats solve(Value<Type>& x, Value<Type>& b) override;

  std::string name() const override {
    return config_->diagonalBased ? "DILU" : "ILU(0)";
  }
  bool usesInitialGuess() const override { return false; }
};
}  // namespace graphene::matrix::solver::ilu