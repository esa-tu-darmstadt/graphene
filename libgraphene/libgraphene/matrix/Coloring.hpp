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

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/dsl/tensor/Traits.hpp"

namespace graphene::matrix {
struct Coloring {
  /** For each color, the start index of the color in the \ref colorPtr array */
  Tensor colorSortStartPtr;

  /** The row indices of each color */
  Tensor colorSortAddr;

  Coloring(Tensor colorSortAddr, Tensor colorSortStartPtr)
      : colorSortAddr(std::move(colorSortAddr)),
        colorSortStartPtr(std::move(colorSortStartPtr)) {}
};
}  // namespace graphene::matrix