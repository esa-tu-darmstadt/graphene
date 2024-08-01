#include "libgraphene/util/PoplarHelpers.hpp"

#include <optional>
#include <poplar/Interval.hpp>

#include "libgraphene/util/Context.hpp"

namespace graphene {

std::vector<poplar::Interval> calculateIPUIntervals(
    const poplar::Graph::TileToTensorMapping &tileMapping) {
  size_t tilesPerIPU = Context::graph().getTarget().getTilesPerIPU();

  std::vector<poplar::Interval> intervals;
  size_t currentElement = 0;
  for (size_t tile = 0; tile < tileMapping.size(); ++tile) {
    size_t ipu = tile / tilesPerIPU;
    if (ipu >= intervals.size()) {
      intervals.emplace_back(currentElement, currentElement);
    }
    for (size_t i = 0; i < tileMapping[tile].size(); i++) {
      currentElement += tileMapping[tile][i].size();
    }
    intervals[ipu] = poplar::Interval(intervals[ipu].begin(), currentElement);
  }
  return intervals;
}

std::vector<size_t> getCoordinateFromElementIndex(
    const std::vector<size_t> &shape, size_t elementIndex) {
  std::vector<size_t> coordinates(shape.size());
  size_t remainingElements = elementIndex;
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    size_t dimSize = shape[dim];
    coordinates[shape.size() - dim - 1] = remainingElements % dimSize;
    remainingElements /= dimSize;

    if (remainingElements == 0) {
      break;
    }
  }
  return coordinates;
}

poplar::Tensor sliceTensorToIPU(poplar::Tensor tensor, size_t ipu) {
  auto ipuIntervals =
      calculateIPUIntervals(Context::graph().getTileMapping(tensor));

  // The stride is the product of all dimensions except the first one
  size_t stride = 1;
  std::vector<size_t> shape = tensor.shape();

  // Zero-dimensional tensors cannot be sliced
  if (shape.empty()) return ipu == 0 ? tensor : poplar::Tensor();

  if (shape.size() > 1)
    stride = std::accumulate(++shape.begin(), shape.end(), 1,
                             std::multiplies<size_t>());

  size_t startElements = ipuIntervals[ipu].begin();
  size_t endElements = ipuIntervals[ipu].end();

  assert(startElements % stride == 0 &&
         "Tensor can only be partitioned to multiple IPUs along the first "
         "dimension");
  assert(endElements % stride == 0 &&
         "Tensor can only be partitioned to multiple IPUs along the first "
         "dimension");

  size_t startIndex = startElements / stride;
  size_t endIndex = endElements / stride;

  return tensor.slice(startIndex, endIndex);
}

poplar::Tensor sliceTensorToTile(
    poplar::Tensor tensor, size_t tile,
    const poplar::Graph::TileToTensorMapping *mapping) {
  std::optional<poplar::Graph::TileToTensorMapping> queriedMapping;
  if (mapping == nullptr) {
    queriedMapping = Context::graph().getTileMapping(tensor);
    mapping = &queriedMapping.value();
  }

  auto intervals = (*mapping)[tile];
  if (intervals.size() > 1)
    throw std::runtime_error(
        "This function currently only supports tensors with at most a single "
        "interval per tile");
  poplar::Interval interval = intervals[0];

  // The stride is the product of all dimensions except the first one
  size_t stride = 1;
  std::vector<size_t> shape = tensor.shape();

  // Zero-dimensional tensors cannot be sliced
  if (shape.empty()) return tile == 0 ? tensor : poplar::Tensor();

  if (shape.size() > 1)
    stride = std::accumulate(++shape.begin(), shape.end(), 1,
                             std::multiplies<size_t>());

  assert(interval.begin() % stride == 0 &&
         "Tensor can only be partitioned to multiple IPUs along the first "
         "dimension");
  assert(interval.end() % stride == 0 &&
         "Tensor can only be partitioned to multiple IPUs along the first "
         "dimension");

  size_t startIndex = interval.begin() / stride;
  size_t endIndex = interval.end() / stride;

  return tensor.slice(startIndex, endIndex);
}
}  // namespace graphene