#pragma once

#include <limits>
#include <nlohmann/json_fwd.hpp>

#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix::solver::ilu {

struct Configuration : solver::Configuration,
                       public std::enable_shared_from_this<Configuration> {
  MultiColorMode solveMulticolor = MultiColorMode::Auto;
  MultiColorMode factorizeMulticolor = MultiColorMode::Auto;
  bool diagonalBased = true;

  Configuration() = default;
  Configuration(nlohmann::json const& config);

  std::string solverName() const override {
    return diagonalBased ? "DILU" : "ILU(0)";
  }
};
}  // namespace graphene::matrix::solver::ilu