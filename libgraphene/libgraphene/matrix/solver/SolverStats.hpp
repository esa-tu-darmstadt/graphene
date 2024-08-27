#pragma once

#include <optional>
#include <poplar/DebugContext.hpp>
#include <string>

#include "libgraphene/dsl/Tensor.hpp"
#include "libgraphene/matrix/Norm.hpp"
namespace graphene::matrix::solver {
struct SolverStats {
  std::string solverName;
  VectorNorm norm;

  size_t numTiles;

  Tensor<float> initialResidual =
      Tensor<float>(std::numeric_limits<float>::infinity());
  Tensor<float> finalResidual =
      Tensor<float>(std::numeric_limits<float>::infinity());

  Tensor<int> iterations = Tensor<int>(0);

  Tensor<bool> converged = Tensor<bool>(false);
  Tensor<bool> singular = Tensor<bool>(false);

  Tensor<float> normFactor = Tensor<float>(0);
  std::optional<Tensor<float>> bNorm;

  SolverStats(std::string solverName, VectorNorm norm, size_t numTiles)
      : solverName(std::move(solverName)), norm(norm), numTiles(numTiles) {}

  /// Check if the solver has converged. The solver has converged if one of the
  /// following conditions is met:
  /// 1. ||r|| < absTolerance
  /// 2. ||r|| < relTolerance * ||b||
  /// 3. ||r|| < relResidual * ||r0||
  /// So relative tolerance describes the residual in relation to b. Relative
  /// residual describes the residual in relation to the initial residual.
  /// To be compatible with OpenFOAM, use the L1 scaled norm, set absTolerance
  /// to "tolerance", set relTolerance to 0 and relResidual to  "relTol".
  void checkConvergence(float absTolerance, float relTolerance,
                        float relResidual);

  void checkSingularity(Tensor<float> wApA,
                        float tolerance = Traits<float>::vsmall());

  bool requiresBNorm(float relTolerance) const { return relTolerance > 0; }

  void print() const;
};
}  // namespace graphene::matrix::solver