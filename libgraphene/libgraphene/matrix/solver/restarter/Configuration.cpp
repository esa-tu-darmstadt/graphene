#include "libgraphene/matrix/solver/restarter/Configuration.hpp"

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::restarter {
Configuration::Configuration(nlohmann::json const& config) {
  setFieldFromJSON<int>(config, "maxRestarts", maxRestarts);

  innerSolver = solver::Configuration::fromJSON(config["innerSolver"]);
}

}  // namespace graphene::matrix::solver::restarter