#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <vector>

namespace graphene {

/**
 * Calculates the per-IPU intervals based on the tile-to-tensor mapping.
 *
 * @param tileMapping The tile-to-tensor mapping of the graph.
 * @return A vector of poplar::Interval containing one Interval per IPU.
 */
std::vector<poplar::Interval> calculateIPUIntervals(
    const poplar::Graph::TileToTensorMapping &tileMapping);

/**
 * Returns the slice of a tensor that is mapped to a specific IPU. The tensor
 * has the same rank but is sliced along the first dimension.
 *
 * @param tensor The tensor to slice.
 * @param ipu The IPU to slice the tensor for.
 */
poplar::Tensor sliceTensorToIPU(poplar::Tensor tensor, size_t ipu);

/**
 * Returns the slice of a tensor that is mapped to a specific tile. The tensor
 * has the same rank but is sliced along the first dimension.
 *
 * @param tensor The tensor to slice.
 * @param tile The tile to slice the tensor for.
 * @param cache The mapping of the tensor to tiles. If not provided, the mapping
 * will be queried from the graph, which can take a significant amount of time.
 */
poplar::Tensor sliceTensorToTile(
    poplar::Tensor tensor, size_t tile,
    const poplar::Graph::TileToTensorMapping *mapping = nullptr);

/**
 * @brief Converts an element index to its corresponding coordinates in its
 * tensor.
 *
 * Given the shape of a tensor and an element index, this function calculates
 * the coordinates of the element in the tensor
 *
 * @param shape The shape of the tensor.
 * @param elementIndex The index of the element.
 * @return The coordinates of the element in the tensor.
 */
std::vector<size_t> getCoordinateFromElementIndex(
    const std::vector<size_t> &shape, size_t elementIndex);

}  // namespace graphene