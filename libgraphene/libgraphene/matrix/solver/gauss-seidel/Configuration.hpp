#pragma once

#include <limits>
#include <nlohmann/json_fwd.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix::solver::gaussseidel {

struct Configuration : solver::Configuration,
                       public std::enable_shared_from_this<Configuration> {
  float absTolerance = 0;
  float relTolerance = 0;
  float relResidual = 0;
  int maxIterations = std::numeric_limits<int>::max();
  int minIterations = 0;
  int numSweeps = 1;
  int numFixedIterations = 0;
  MultiColorMode solveMulticolor = MultiColorMode::Auto;
  bool printPerformanceAfterSolve = false;
  bool printPerformanceEachIteration = false;
  VectorNorm norm = VectorNorm::L2;
  TypeRef workingType = nullptr;

  Configuration() = default;
  Configuration(nlohmann::json const& config);

  std::string solverName() const override { return "GaussSeidel"; }
};
}  // namespace graphene::matrix::solver::gaussseidel