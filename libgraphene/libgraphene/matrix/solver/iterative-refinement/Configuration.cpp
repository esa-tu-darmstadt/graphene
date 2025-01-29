#include "libgraphene/matrix/solver/iterative-refinement/Configuration.hpp"

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::iterativerefinement {
Configuration::Configuration(nlohmann::json const& config) {
  setFieldFromJSON<float>(config, "absTolerance", absTolerance);
  setFieldFromJSON<float>(config, "relTolerance", relTolerance);
  setFieldFromJSON<float>(config, "relResidual", relResidual);
  setFieldFromJSON<int>(config, "maxIterations", maxIterations);
  setFieldFromJSON<int>(config, "minIterations", minIterations);
  setFieldFromJSON<bool>(config, "printPerformanceAfterSolve",
                         printPerformanceAfterSolve);
  setFieldFromJSON<bool>(config, "printPerformanceEachIteration",
                         printPerformanceEachIteration);
  setFieldFromJSON<VectorNorm>(config, "norm", norm);

  setFieldFromJSON<TypeRef>(config, "extendedPrecisionType",
                            extendedPrecisionType);

  innerSolver = solver::Configuration::fromJSON(config["innerSolver"]);
}

}  // namespace graphene::matrix::solver::iterativerefinement