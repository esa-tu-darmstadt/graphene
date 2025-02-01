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

#pragma once

#include <limits>
#include <nlohmann/json_fwd.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix::solver::pbicgstab {

struct Configuration : solver::Configuration,
                       public std::enable_shared_from_this<Configuration> {
  float absTolerance = 0;
  float relTolerance = 0;
  float relResidual = 0;
  int maxIterations = std::numeric_limits<int>::max();
  int minIterations = 0;
  bool printPerformanceAfterSolve = false;
  bool printPerformanceEachIteration = false;
  bool verbose = false;
  VectorNorm norm = VectorNorm::L2;
  TypeRef workingType = nullptr;

  std::shared_ptr<solver::Configuration> preconditioner;

  Configuration() = default;
  Configuration(nlohmann::json const& config);

  std::string solverName() const override {
    if (preconditioner) {
      return "PBiCGStab(" + preconditioner->solverName() + ")";
    } else {
      return "BiCGStab";
    }
  }
};
}  // namespace graphene::matrix::solver::pbicgstab