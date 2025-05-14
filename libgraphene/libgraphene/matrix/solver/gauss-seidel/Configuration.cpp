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

#include <boost/property_tree/ptree.hpp>

#include "libgraphene/matrix/Norm.hpp"

namespace graphene::matrix::solver::gaussseidel {
Configuration::Configuration(boost::property_tree::ptree const& config) {
  setFieldFromPTree<float>(config, "absTolerance", absTolerance);
  setFieldFromPTree<float>(config, "relTolerance", relTolerance);
  setFieldFromPTree<float>(config, "relResidual", relResidual);
  setFieldFromPTree<int>(config, "maxIterations", maxIterations);
  setFieldFromPTree<int>(config, "minIterations", minIterations);
  setFieldFromPTree<int>(config, "numSweeps", numSweeps);
  setFieldFromPTree<int>(config, "numFixedIterations", numFixedIterations);
  setFieldFromPTree<MultiColorMode>(config, "solveMulticolor", solveMulticolor);
  setFieldFromPTree<bool>(config, "printPerformanceAfterSolve",
                        printPerformanceAfterSolve);
  setFieldFromPTree<bool>(config, "printPerformanceEachIteration",
                        printPerformanceEachIteration);
  setFieldFromPTree<VectorNorm>(config, "norm", norm);
  setFieldFromPTree<TypeRef>(config, "workingType", workingType);
}

}  // namespace graphene::matrix::solver::gaussseidel