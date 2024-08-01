#include "libgraphene/matrix/solver/ilu/Configuration.hpp"

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::ilu {
Configuration::Configuration(nlohmann::json const& config) {
  setFieldFromJSON<MultiColorMode>(config, "solveMulticolor", solveMulticolor);
  setFieldFromJSON<MultiColorMode>(config, "factorizeMulticolor",
                                   factorizeMulticolor);
  setFieldFromJSON<bool>(config, "diagonalBased", diagonalBased);
}

}  // namespace graphene::matrix::solver::ilu