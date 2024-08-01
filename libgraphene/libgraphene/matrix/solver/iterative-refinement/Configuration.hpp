#pragma once

#include <limits>
#include <nlohmann/json_fwd.hpp>

#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix::solver::iterativerefinement {

struct Configuration : solver::Configuration,
                       public std::enable_shared_from_this<Configuration> {
  float absTolerance = 0;
  float relTolerance = 0;
  float relResidual = 0;
  int maxIterations = std::numeric_limits<int>::max();
  int minIterations = 0;
  bool printPerformanceAfterSolve = false;
  bool printPerformanceEachIteration = false;
  VectorNorm norm = VectorNorm::L2;

  std::shared_ptr<solver::Configuration> innerSolver;
  bool mixedPrecision = false;

  Configuration() = default;
  Configuration(nlohmann::json const& config);

  std::string solverName() const override { return "IterativeRefinement"; }
};
}  // namespace graphene::matrix::solver::iterativerefinement