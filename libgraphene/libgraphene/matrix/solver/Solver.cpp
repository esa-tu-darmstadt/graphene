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

#include "libgraphene/matrix/solver/Solver.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Solver.hpp"
#include "libgraphene/matrix/solver/ilu/Solver.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Solver.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Solver.hpp"
#include "libgraphene/matrix/solver/restarter/Solver.hpp"

namespace graphene::matrix::solver {
std::unique_ptr<Solver> Solver::createSolver(
    const Matrix &matrix, std::shared_ptr<Configuration> config) {
  std::string solverName = config->solverName();
  spdlog::debug("Creating solver {}", solverName);

  if (auto gaussSeidelConfig =
          std::dynamic_pointer_cast<gaussseidel::Configuration>(config)) {
    return std::make_unique<gaussseidel::Solver>(matrix, gaussSeidelConfig);
  } else if (auto irConfig =
                 std::dynamic_pointer_cast<iterativerefinement::Configuration>(
                     config)) {
    return std::make_unique<iterativerefinement::Solver>(matrix, irConfig);
  } else if (auto iluConfig =
                 std::dynamic_pointer_cast<ilu::Configuration>(config)) {
    return std::make_unique<ilu::Solver>(matrix, iluConfig);
  } else if (auto pbicgstabConfig =
                 std::dynamic_pointer_cast<pbicgstab::Configuration>(config)) {
    return std::make_unique<pbicgstab::Solver>(matrix, pbicgstabConfig);
  } else if (auto restarterConfig =
                 std::dynamic_pointer_cast<restarter::Configuration>(config)) {
    return std::make_unique<restarter::Solver>(matrix, restarterConfig);
  } else {
    throw std::runtime_error("Unknown solver: " + solverName);
  }
}

bool Solver::shouldUseMulticolor(MultiColorMode mode) const {
  switch (mode) {
    case MultiColorMode::On:
      return true;
    case MultiColorMode::Off:
      return false;
    case MultiColorMode::Auto:
      return matrix().hostMatrix().multicolorRecommended();
  }
  return false;
}

}  // namespace graphene::matrix::solver