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

#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"

#include <nlohmann/json.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::gaussseidel {
Configuration::Configuration(nlohmann::json const& config) {
  setFieldFromJSON<float>(config, "absTolerance", absTolerance);
  setFieldFromJSON<float>(config, "relTolerance", relTolerance);
  setFieldFromJSON<float>(config, "relResidual", relResidual);
  setFieldFromJSON<int>(config, "maxIterations", maxIterations);
  setFieldFromJSON<int>(config, "minIterations", minIterations);
  setFieldFromJSON<int>(config, "numSweeps", numSweeps);
  setFieldFromJSON<int>(config, "numFixedIterations", numFixedIterations);
  setFieldFromJSON<MultiColorMode>(config, "solveMulticolor", solveMulticolor);
  setFieldFromJSON<bool>(config, "printPerformanceAfterSolve",
                         printPerformanceAfterSolve);
  setFieldFromJSON<bool>(config, "printPerformanceEachIteration",
                         printPerformanceEachIteration);
  setFieldFromJSON<VectorNorm>(config, "norm", norm);
  setFieldFromJSON<TypeRef>(config, "workingType", workingType);
}

}  // namespace graphene::matrix::solver::gaussseidel