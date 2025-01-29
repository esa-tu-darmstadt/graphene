#include "libgraphene/matrix/solver/pbicgstab/Configuration.hpp"

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::pbicgstab {
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
  setFieldFromJSON<bool>(config, "verbose", verbose);
  setFieldFromJSON<TypeRef>(config, "workingType", workingType);

  if (config.contains("preconditioner"))
    preconditioner = solver::Configuration::fromJSON(config["preconditioner"]);
}

}  // namespace graphene::matrix::solver::pbicgstab