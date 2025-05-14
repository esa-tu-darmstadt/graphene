/*
 * Graphene Linear Algebra Framework for Intelligence Processing Units.
 * Copyright (C) 2025 Embedded Systems and Applications, TU Darmstadt.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "libgraphene/matrix/solver/Configuration.hpp"

#include <fmt/format.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>
#include <sstream>

#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"
#include "libgraphene/matrix/solver/ilu/Configuration.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Configuration.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Configuration.hpp"
#include "libgraphene/matrix/solver/restarter/Configuration.hpp"

namespace nlohmann {

auto format_as(const json& j) { return j.dump(); }

}  // namespace nlohmann

namespace graphene::matrix::solver {
template <>
void Configuration::setFieldFromPTree<VectorNorm>(
    boost::property_tree::ptree const& config, std::string const& field,
    VectorNorm& value) {
  auto maybeValue = config.get_optional<std::string>(field);
  if (maybeValue) {
    value = parseVectorNorm(maybeValue.value());
  }
}

template <>
void Configuration::setFieldFromPTree<MultiColorMode>(
    boost::property_tree::ptree const& config, std::string const& field,
    MultiColorMode& value) {
  auto maybeValue = config.get_optional<std::string>(field);
  if (maybeValue) {
    std::string valueStr = maybeValue.value();
    if (valueStr == "true" || valueStr == "on") {
      value = MultiColorMode::On;
    } else if (valueStr == "false" || valueStr == "off") {
      value = MultiColorMode::Off;
    } else if (valueStr == "auto") {
      value = MultiColorMode::Auto;
    } else {
      throw std::runtime_error(fmt::format(
          "Invalid value for MultiColorMode field {}: {}. Allowed values are "
          "true, false, on, off or \"auto\"",
          field, valueStr));
    }
  }

  // Also try to read as boolean for backward compatibility
  auto maybeBool = config.get_optional<bool>(field);
  if (maybeBool) {
    value = maybeBool.value() ? MultiColorMode::On : MultiColorMode::Off;
  }
}

template <>
void Configuration::setFieldFromPTree<TypeRef>(
    boost::property_tree::ptree const& config, std::string const& field,
    TypeRef& value) {
  auto maybeValue = config.get_optional<std::string>(field);
  if (maybeValue) {
    value = parseType(maybeValue.value());
    if (!value) {
      throw std::runtime_error(
          fmt::format("Unknown type in config: {}", maybeValue.value()));
    }
  }
}

template <typename T>
void Configuration::setFieldFromPTree(boost::property_tree::ptree const& config,
                                      std::string const& field, T& value) {
  auto maybeValue = config.get_optional<T>(field);
  if (maybeValue) {
    value = maybeValue.value();
  }
}

std::shared_ptr<Configuration> Configuration::fromPTree(
    boost::property_tree::ptree const& config) {
  auto maybeType = config.get_optional<std::string>("type");
  if (!maybeType) {
    throw std::runtime_error("No solver type specified in configuration");
  }

  std::string solver = maybeType.value();
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

std::shared_ptr<Configuration> Configuration::fromJSON(
    nlohmann::json const& config) {
  // Convert JSON to property tree
  std::stringstream ss;
  ss << config;

  boost::property_tree::ptree ptree;
  boost::property_tree::read_json(ss, ptree);

  return fromPTree(ptree);
}

// Template instantiation
template void Configuration::setFieldFromPTree<float>(
    boost::property_tree::ptree const&, std::string const&, float&);
template void Configuration::setFieldFromPTree<int>(
    boost::property_tree::ptree const&, std::string const&, int&);
template void Configuration::setFieldFromPTree<bool>(
    boost::property_tree::ptree const&, std::string const&, bool&);
template void Configuration::setFieldFromPTree<std::string>(
    boost::property_tree::ptree const&, std::string const&, std::string&);
}  // namespace graphene::matrix::solver