#pragma once

#include <optional>
#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>
#include <unordered_map>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Type.hpp"

namespace graphene {

class Tensor;

/**
 * @brief A class representing a tensor stored in remote buffers.
 *
 * This class manages remote buffers and their mappings to tiles on an IPU.
 * It provides functionality to copy the remote value to a tile.
 *
 * @tparam Type The data type of the remote value.
 */
class RemoteTensor {
  TypeRef type_;
  // Remote buffers for each IPU
  std::unordered_map<size_t, poplar::RemoteBuffer> buffers_;
  TileMapping tileMapping_;
  DistributedShape shape_;
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
   * @param buffers Remote buffer per IPU.
   * @param tileMapping Tile mapping of the remote tensor.
   * @param shape The shape of the remote tensor.
   * @param debugStr A debug string for logging and debugging purposes.
   */
  explicit RemoteTensor(
      TypeRef type, std::unordered_map<size_t, poplar::RemoteBuffer> buffers,
      TileMapping tileMapping, DistributedShape shape, std::string debugStr)
      : type_(type),
        buffers_(std::move(buffers)),
        tileMapping_(std::move(tileMapping)),
        shape_(std::move(shape)),
        debugStr_(std::move(debugStr)) {}

  /**
   * @brief Copies the remote tensor to tile memory.
   *
   * @return A Value object representing the copied tensor in tile memory.
   */
  [[nodiscard]] Tensor copyToTile() const;

  /**
   * @brief Gets the element type of the remote tensor.
   */
  TypeRef type() const { return type_; }
};

}  // namespace graphene