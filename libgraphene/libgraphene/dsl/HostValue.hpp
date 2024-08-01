#pragma once

#include <spdlog/spdlog.h>

#include <initializer_list>
#include <poplar/Graph.hpp>
#include <stdexcept>
#include <string>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/util/Runtime.hpp"

namespace graphene {
template <DataType Type>
class RemoteValue;

template <DataType Type>
class HostValue {
  struct Storage : Runtime::HostResource {
    std::vector<Type> data;
    std::vector<size_t> shape;
    poplar::Graph::TileToTensorMapping mapping;
    std::string name;

    bool wasCopiedToRemote = false;
    Storage(std::vector<Type> data, std::vector<size_t> shape,
            poplar::Graph::TileToTensorMapping mapping, std::string name)
        : data(std::move(data)),
          shape(std::move(shape)),
          mapping(std::move(mapping)),
          name(std::move(name)) {}

    ~Storage() override { spdlog::trace("HostValue storage destructed"); }
  };

  std::shared_ptr<Storage> storage_;

 public:
  HostValue() = default;

  HostValue(std::vector<Type> data, std::vector<size_t> shape,
            poplar::Graph::TileToTensorMapping mapping, std::string name)
      : storage_(Runtime::instance().createResource<Storage>(
            std::move(data), std::move(shape), std::move(mapping),
            std::move(name))) {}

  ~HostValue() {
    if (storage_ && !storage_->wasCopiedToRemote)
      Runtime::instance().freeResource(storage_);
  }

  const auto &data() const { return storage_->data; }
  const auto &shape() const { return storage_->shape; }
  const auto &mapping() const { return storage_->mapping; }
  const auto &name() const { return storage_->name; }

  Type get(std::initializer_list<size_t> indices) const {
    if (indices.size() != storage_->shape.size()) {
      throw std::runtime_error(
          "The number of indices must match the number of dimensions");
    }

    size_t index = 0;
    size_t stride = 1;
    for (auto it = indices.begin(); it != indices.end(); ++it) {
      index += *it * stride;
      stride *= storage_->shape[std::distance(indices.begin(), it)];
    }
    return storage_->data[index];
  }

  Type get(size_t index) const { return get({index}); }

  size_t numElements() const {
    return std::accumulate(storage_->shape.begin(), storage_->shape.end(), 1,
                           std::multiplies<size_t>());
  }

  RemoteValue<Type> copyToRemote() const;
};
}  // namespace graphene