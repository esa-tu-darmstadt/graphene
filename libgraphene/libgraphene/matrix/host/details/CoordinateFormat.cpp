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

#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>

#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::host {

template <DataType Type>
void sortTripletMatrx(TripletMatrix<Type> &tripletMatrix) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Sorting COO matrix in parallel");

  auto compareMatrixEntry = [](const auto &a, const auto &b) {
    return std::tie(a.row, a.col) < std::tie(b.row, b.col);
  };

  std::sort(tripletMatrix.entries.begin(), tripletMatrix.entries.end(),
            compareMatrixEntry);
}

template void sortTripletMatrx(TripletMatrix<float> &tripletMatrix);
template void sortTripletMatrx(TripletMatrix<double> &tripletMatrix);
template void sortTripletMatrx(TripletMatrix<doubleword> &tripletMatrix);
}  // namespace graphene::matrix::host