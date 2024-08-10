#include "libgraphene/matrix/solver/Configuration.hpp"

#include <spdlog/fmt/bundled/core.h>

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"
#include "libgraphene/matrix/solver/ilu/Configuration.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Configuration.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Configuration.hpp"
#include "libgraphene/matrix/solver/restarter/Configuration.hpp"

namespace graphene::matrix::solver {
template <>
void Configuration::setFieldFromJSON<VectorNorm>(nlohmann::json const& config,
                                                 std::string const& field,
                                                 VectorNorm& value) {
  if (config.find(field) != config.end()) {
    value = parseVectorNorm(config[field]);
  }
}

template <>
void Configuration::setFieldFromJSON<MultiColorMode>(
    nlohmann::json const& config, std::string const& field,
    MultiColorMode& value) {
  if (config.find(field) != config.end()) {
    if (config[field].is_boolean()) {
      value =
          config[field].get<bool>() ? MultiColorMode::On : MultiColorMode::Off;
    } else if (config[field].is_string() && config[field] == "auto") {
      value = MultiColorMode::Auto;
    } else {
      throw std::runtime_error(fmt::format(
          "Invalid value for MultiColorMode field {}: {}. Allowed values are "
          "true, false or \"auto\"",
          field, config[field]));
    }
  }
}

template <typename T>
void Configuration::setFieldFromJSON(nlohmann::json const& config,
                                     std::string const& field, T& value) {
  if (config.find(field) != config.end()) {
    value = config[field];
  }
}

std::shared_ptr<Configuration> Configuration::fromJSON(
    nlohmann::json const& config) {
  if (config.find("type") == config.end()) {
    throw std::runtime_error("No solver specified in configuration");
  }

  std::string solver = config["type"];
  if (solver == "IterativeRefinement") {
    return std::make_shared<iterativerefinement::Configuration>(config);
  } else if (solver == "GaussSeidel") {
    return std::make_shared<gaussseidel::Configuration>(config);
  } else if (solver == "ILU") {
    return std::make_shared<ilu::Configuration>(config);
  } else if (solver == "PBiCGStab") {
    return std::make_shared<pbicgstab::Configuration>(config);
  } else if (solver == "restarter") {
    return std::make_shared<restarter::Configuration>(config);
  } else {
    throw std::runtime_error("Unknown solver: " + solver);
  }
}

// Template instantiation
template void Configuration::setFieldFromJSON<float>(nlohmann::json const&,
                                                     std::string const&,
                                                     float&);
template void Configuration::setFieldFromJSON<int>(nlohmann::json const&,
                                                   std::string const&, int&);
template void Configuration::setFieldFromJSON<bool>(nlohmann::json const&,
                                                    std::string const&, bool&);
}  // namespace graphene::matrix::solver