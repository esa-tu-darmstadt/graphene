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
  if (intervalsOnTile.size() == 0) return {};

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