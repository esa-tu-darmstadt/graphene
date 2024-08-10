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

/**
 * @brief A class representing a host-side value and its tile mapping on an IPU.
 *
 * @tparam Type The data type of the elements stored in the HostValue.
 */
template <DataType Type>
class HostValue {
  /**
   * @brief Internal storage structure for HostValue.
   * Instances of this class are owned by the \ref Runtime, so that they are
   * still alive when the memory is actually copied to the IPUs remote memory
   * when the compiled graph program is executed.
   */
  struct Storage : Runtime::HostResource {
    std::vector<Type> data;     ///< The data stored in the HostValue.
    std::vector<size_t> shape;  ///< The shape of the data.
    poplar::Graph::TileToTensorMapping
        mapping;       ///< Mapping of tiles to tensor.
    std::string name;  ///< Name of the HostValue.

    bool wasCopiedToRemote =
        false;  ///< Flag indicating if the data was copied to remote.

    /**
     * @brief Construct a new Storage object.
     *
     * @param data The data to be stored.
     * @param shape The shape of the data.
     * @param mapping The tile to tensor mapping.
     * @param name The name of the HostValue.
     */
    Storage(std::vector<Type> data, std::vector<size_t> shape,
            poplar::Graph::TileToTensorMapping mapping, std::string name)
        : data(std::move(data)),
          shape(std::move(shape)),
          mapping(std::move(mapping)),
          name(std::move(name)) {}

    /**
     * @brief Destructor for Storage.
     */
    ~Storage() override { spdlog::trace("HostValue storage destructed"); }
  };

  std::shared_ptr<Storage>
      storage_;  ///< Shared pointer to the internal storage.

 public:
  /**
   * @brief Default constructor for HostValue.
   */
  HostValue() = default;

  /**
   * @brief Construct a new HostValue object.
   *
   * @param data The data to be stored.
   * @param shape The shape of the data.
   * @param mapping The tile to tensor mapping.
   * @param name The name of the HostValue.
   */
  HostValue(std::vector<Type> data, std::vector<size_t> shape,
            poplar::Graph::TileToTensorMapping mapping, std::string name)
      : storage_(Runtime::instance().createResource<Storage>(
            std::move(data), std::move(shape), std::move(mapping),
            std::move(name))) {}

  /**
   * @brief Destructor for HostValue.
   */
  ~HostValue() {
    if (storage_ && !storage_->wasCopiedToRemote)
      Runtime::instance().freeResource(storage_);
  }

  /**
   * @brief Get the data stored in the HostValue.
   *
   * @return const auto& Reference to the data.
   */
  const auto &data() const { return storage_->data; }

  /**
   * @brief Get the shape of the data.
   *
   * @return const auto& Reference to the shape.
   */
  const auto &shape() const { return storage_->shape; }

  /**
   * @brief Get the tile to tensor mapping.
   *
   * @return const auto& Reference to the mapping.
   */
  const auto &mapping() const { return storage_->mapping; }

  /**
   * @brief Get the name of the HostValue.
   *
   * @return const auto& Reference to the name.
   */
  const auto &name() const { return storage_->name; }

  /**
   * @brief Get the value at the specified indices.
   *
   * @param indices The indices to access.
   * @return Type The value at the specified indices.
   * @throws std::runtime_error if the number of indices does not match the
   * number of dimensions.
   */
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

  /**
   * @brief Get the value at the specified index.
   *
   * @param index The index to access.
   * @return Type The value at the specified index.
   */
  Type get(size_t index) const { return get({index}); }

  /**
   * @brief Get the total number of elements in the data.
   *
   * @return size_t The number of elements.
   */
  size_t numElements() const {
    return std::accumulate(storage_->shape.begin(), storage_->shape.end(), 1,
                           std::multiplies<size_t>());
  }

  /**
   * @brief Copy the data to a remote location.
   *
   * @return RemoteValue<Type> The remote value.
   */
  RemoteValue<Type> copyToRemote() const;
};

}  // namespace graphene