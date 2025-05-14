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

#include <boost/property_tree/ptree.hpp>
#include <limits>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"

namespace graphene::matrix::solver::iterativerefinement {

struct Configuration : solver::Configuration,
                       public std::enable_shared_from_this<Configuration> {
  float absTolerance = 0;
  float relTolerance = 0;
  float relResidual = 0;
  int maxIterations = std::numeric_limits<int>::max();
  int minIterations = 0;
  bool printPerformanceAfterSolve = false;
  bool printPerformanceEachIteration = false;
  VectorNorm norm = VectorNorm::L2;

  std::shared_ptr<solver::Configuration> innerSolver;

  TypeRef extendedPrecisionType = nullptr;

  Configuration() = default;
  Configuration(boost::property_tree::ptree const& config);

  std::string solverName() const override { return "IterativeRefinement"; }
};
}  // namespace graphene::matrix::solver::iterativerefinement