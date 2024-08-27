#pragma once

#include <optional>

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/dsl/tensor/TensorVariant.hpp"
#include "libgraphene/dsl/tensor/Traits.hpp"
#include "libgraphene/matrix/Addressing.hpp"
#include "libgraphene/matrix/Coloring.hpp"

namespace graphene::matrix::crs {

struct CRSAddressing : matrix::Addressing {
  /** For each row, the start index of the row in the \ref colInd array */
  AnyUIntValue rowPtr;

  /** The column indices of each non-zero element */
  AnyUIntValue colInd;

  /** Optional coloring of the matrix */
  std::optional<Coloring> coloring;

  CRSAddressing(AnyUIntValue rowPtr, AnyUIntValue colInd)
      : rowPtr(std::move(rowPtr)), colInd(std::move(colInd)) {}

  CRSAddressing(AnyUIntValue rowPtr, AnyUIntValue colInd, Coloring coloring)
      : rowPtr(std::move(rowPtr)),
        colInd(std::move(colInd)),
        coloring(std::move(coloring)) {}
};

}  // namespace graphene::matrix::crs