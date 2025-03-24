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

#include <optional>

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/dsl/tensor/Traits.hpp"
#include "libgraphene/matrix/Addressing.hpp"
#include "libgraphene/matrix/Coloring.hpp"

namespace graphene::matrix::crs {

struct CRSAddressing : matrix::Addressing {
  /** For each row, the start index of the row in the \ref colInd array */
  Tensor rowPtr;

  /** The column indices of each non-zero element */
  Tensor colInd;

  /** Optional coloring of the matrix */
  std::optional<Coloring> coloring;

  CRSAddressing(Tensor rowPtr, Tensor colInd)
      : rowPtr(std::move(rowPtr)), colInd(std::move(colInd)) {}

  CRSAddressing(Tensor rowPtr, Tensor colInd, Coloring coloring)
      : rowPtr(std::move(rowPtr)),
        colInd(std::move(colInd)),
        coloring(std::move(coloring)) {}
};

}  // namespace graphene::matrix::crs