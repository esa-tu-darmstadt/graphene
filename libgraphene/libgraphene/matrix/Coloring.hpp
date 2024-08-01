#pragma once

#include "libgraphene/dsl/Traits.hpp"
#include "libgraphene/dsl/ValueVariant.hpp"

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