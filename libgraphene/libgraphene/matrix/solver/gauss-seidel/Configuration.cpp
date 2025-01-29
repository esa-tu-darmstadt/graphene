#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::gaussseidel {
Configuration::Configuration(nlohmann::json const& config) {
  setFieldFromJSON<float>(config, "absTolerance", absTolerance);
  setFieldFromJSON<float>(config, "relTolerance", relTolerance);
  setFieldFromJSON<float>(config, "relResidual", relResidual);
  setFieldFromJSON<int>(config, "maxIterations", maxIterations);
  setFieldFromJSON<int>(config, "minIterations", minIterations);
  setFieldFromJSON<int>(config, "numSweeps", numSweeps);
  setFieldFromJSON<int>(config, "numFixedIterations", numFixedIterations);
  setFieldFromJSON<MultiColorMode>(config, "solveMulticolor", solveMulticolor);
  setFieldFromJSON<bool>(config, "printPerformanceAfterSolve",
                         printPerformanceAfterSolve);
  setFieldFromJSON<bool>(config, "printPerformanceEachIteration",
                         printPerformanceEachIteration);
  setFieldFromJSON<VectorNorm>(config, "norm", norm);
  setFieldFromJSON<TypeRef>(config, "workingType", workingType);
}

}  // namespace graphene::matrix::solver::gaussseidel