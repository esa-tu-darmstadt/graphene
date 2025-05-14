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

#include "libgraphene/util/PoplarHelpers.hpp"

#include <optional>
#include <poplar/Interval.hpp>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/util/Context.hpp"

namespace graphene {

std::vector<size_t> getCoordinateFromElementIndex(const DistributedShape &shape,
                                                  size_t elementIndex) {
  std::vector<size_t> coordinates(shape.rank());
  size_t remainingElements = elementIndex;
  for (size_t dim = 0; dim < shape.rank(); ++dim) {
    size_t dimSize = shape[dim];
    coordinates[shape.rank() - dim - 1] = remainingElements % dimSize;
    remainingElements /= dimSize;

    if (remainingElements == 0) {
      break;
    }
  }
  return coordinates;
}

poplar::Tensor sliceTensorToIPU(poplar::Tensor tensor, size_t ipu,
                                const TileMapping &tileMapping) {
  DistributedShape shape =
      DistributedShape::fromPoplar(tensor.shape(), tileMapping.toPoplar());

  // The first dimension is broadcasted to all IPUs
  if (shape[0] == 1) return tensor;

  const size_t numTilesPerIPU = Context::graph().getTarget().getTilesPerIPU();
  TileMapping ipuMapping = tileMapping.translateToIPUMapping(numTilesPerIPU);
  return sliceTensorToTile(tensor, ipu, ipuMapping);
}

poplar::Tensor sliceTensorToTile(poplar::Tensor tensor, size_t tile,
                                 const TileMapping &tileMapping) {
  TensorShape shape(tensor.shape());

  // The first dimension is broadcasted to all tiles
  if (shape[0] == 1) return tensor;

  auto intervalsOnTile = tileMapping[tile];
  if (intervalsOnTile.size() > 1)
    throw std::runtime_error(
        "This function currently only supports tensors with at most a single "
        "interval per tile");
  if (intervalsOnTile.size() == 0) {
    // No elements on this tile, return an empty tensor
    // Is this a legal hack to return an empty tensor? We want the tensor to be
    // valid, but empty.
    return tensor.slice(0, 0);
  }

  Interval interval = *intervalsOnTile.begin();
  size_t stride = shape.stride(0);

  assert(interval.start % stride == 0 &&
         "Tensor can only be partitioned to multiple tiles along the first "
         "dimension");
  assert(interval.end % stride == 0 &&
         "Tensor can only be partitioned to multiple tiles along the first "
         "dimension");

  size_t startIndex = interval.start / stride;
  size_t endIndex = interval.end / stride;

  return tensor.slice(startIndex, endIndex);
}
}  // namespace graphene