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

#include "libgraphene/matrix/solver/restarter/Configuration.hpp"

#include <boost/property_tree/ptree.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::restarter {
Configuration::Configuration(boost::property_tree::ptree const& config) {
  setFieldFromPTree<int>(config, "maxRestarts", maxRestarts);

  auto innerSolverChild = config.get_child_optional("innerSolver");
  if (innerSolverChild) {
    innerSolver = solver::Configuration::fromPTree(innerSolverChild.value());
  } else {
    throw std::runtime_error("Restarter requires an innerSolver configuration");
  }
}

}  // namespace graphene::matrix::solver::restarter