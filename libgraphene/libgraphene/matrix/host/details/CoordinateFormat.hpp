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

#include <filesystem>
#include <vector>

#include "libgraphene/common/Concepts.hpp"

namespace graphene::matrix::host {

/** A partitioning of the rows of a matrix. */
struct Partitioning {
  std::vector<size_t> rowToTile;
  size_t numTiles;
};

/** A matrix in triplet (coordinate) format. */
template <DataType Type>
struct TripletMatrix {
  struct Entry {
    size_t row, col;
    Type val;
  };
  std::vector<Entry> entries;
  size_t nrows = 0, ncols = 0;
};

/** A vector in doublet (coordinate) format. */
template <DataType Type>
struct DoubletVector {
  size_t nrows = 0;
  std::vector<size_t> indices;
  std::vector<Type> values;
};

/** Sorts the matrix in coordinate / triplet format. */
template <DataType Type>
void sortTripletMatrx(TripletMatrix<Type> &tripletMatrix);

}  // namespace graphene::matrix::host