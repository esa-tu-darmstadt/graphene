#pragma once

#include "libgraphene/dsl/tensor/TensorVariant.hpp"
#include "libgraphene/dsl/tensor/Traits.hpp"

namespace graphene::matrix {
struct Coloring {
  /** For each color, the start index of the color in the \ref colorPtr array */
  AnyUIntValue colorSortStartPtr;

  /** The row indices of each color */
  AnyUIntValue colorSortAddr;

  Coloring(AnyUIntValue colorSortAddr, AnyUIntValue colorSortStartPtr)
      : colorSortAddr(std::move(colorSortAddr)),
        colorSortStartPtr(std::move(colorSortStartPtr)) {}
};
}  // namespace graphene::matrix