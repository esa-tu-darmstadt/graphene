#pragma once

#include <optional>
#include <poplar/DataStream.hpp>
#include <poplar/Graph.hpp>

#include "libgraphene/common/Concepts.hpp"

namespace graphene {
template <DataType Type>
class Value;

template <DataType Type>
class RemoteValue {
  // One remote buffer per IPU
  std::vector<poplar::RemoteBuffer> buffers_;
  poplar::Graph::TileToTensorMapping tileMapping_;
  std::vector<size_t> shape_;
  std::string debugStr_;

 public:
  RemoteValue() = delete;
  explicit RemoteValue(std::vector<poplar::RemoteBuffer> buffers,
                       poplar::Graph::TileToTensorMapping tileMapping,
                       std::vector<size_t> shape, std::string debugStr)
      : buffers_(std::move(buffers)),
        tileMapping_(std::move(tileMapping)),
        shape_(std::move(shape)),
        debugStr_(std::move(debugStr)) {}

  Value<Type> copyToTile() const;
};
}  // namespace graphene
