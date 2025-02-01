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