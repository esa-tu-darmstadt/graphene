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

#include "libgraphene/matrix/Matrix.hpp"

#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/Solver.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"

namespace graphene::matrix {
void Matrix::solve(Tensor &x, Tensor &b,
                   std::shared_ptr<solver::Configuration> &config) {
  auto solver = solver::Solver::createSolver(*this, config);
  solver->solve(x, b);
}

}  // namespace graphene::matrix