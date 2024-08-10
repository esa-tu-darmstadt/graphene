#pragma once

#include <spdlog/fmt/bundled/core.h>

#include <limits>
#include <nlohmann/json_fwd.hpp>

#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix::solver::restarter {

struct Configuration : solver::Configuration,
                       public std::enable_shared_from_this<Configuration> {
  int maxRestarts = 1;

  std::shared_ptr<solver::Configuration> innerSolver;

  Configuration() = default;
  Configuration(nlohmann::json const& config);

  std::string solverName() const override {
    return fmt::format("Restarter of {}", innerSolver->solverName());
  }
};
}  // namespace graphene::matrix::solver::restarter