#pragma once

#include <optional>
#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>

#include "libgraphene/common/Concepts.hpp"

namespace graphene {

template <DataType Type>
class Tensor;

/**
 * @brief A class representing a tensor stored in remote buffers.
 *
 * This class manages remote buffers and their mappings to tiles on an IPU.
 * It provides functionality to copy the remote value to a tile.
 *
 * @tparam Type The data type of the remote value.
 */
template <DataType Type>
class RemoteTensor {
  // One remote buffer per IPU
  std::vector<poplar::RemoteBuffer> buffers_;
  poplar::Graph::TileToTensorMapping tileMapping_;
  std::vector<size_t> shape_;
  std::string debugStr_;

 public:
  /**
   * @brief Deleted default constructor to prevent instantiation without
   * parameters.
   */
  RemoteTensor() = delete;

  /**
   * @brief Constructs a RemoteValue with the given buffers, tile mapping,
   * shape, and debug string.
   *
   * @param buffers A vector of remote buffers.
   * @param tileMapping The mapping of tiles to tensors.
   * @param shape The shape of the remote value.
   * @param debugStr A debug string for logging and debugging purposes.
   */
  explicit RemoteTensor(std::vector<poplar::RemoteBuffer> buffers,
                        poplar::Graph::TileToTensorMapping tileMapping,
                        std::vector<size_t> shape, std::string debugStr)
      : buffers_(std::move(buffers)),
        tileMapping_(std::move(tileMapping)),
        shape_(std::move(shape)),
        debugStr_(std::move(debugStr)) {}

  /**
   * @brief Copies the remote value to tile memory.
   *
   * @return A Value object representing the copied value in tile memory.
   */
  Tensor<Type> copyToTile() const;
};

}  // namespace graphene