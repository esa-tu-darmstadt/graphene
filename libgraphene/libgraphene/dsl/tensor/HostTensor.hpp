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

#pragma once

#include <spdlog/spdlog.h>

#include <initializer_list>
#include <poplar/Graph.hpp>
#include <stdexcept>
#include <string>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/util/Runtime.hpp"

namespace graphene {
class RemoteTensor;

/**
 * @brief A class representing a tensor on the host and its (future) tile
 * mapping on an IPU.
 *
 */
class HostTensor {
  /**
   * @brief Internal storage structure for HostValue.
   * Instances of this class are owned by the \ref Runtime, so that they are
   * still alive when the memory is actually copied to the IPUs remote memory
   * when the compiled graph program is executed.
   */
  struct Storage : Runtime::HostResource {
    std::any storage;  // The data storage itself.
    void *data;        /// A pointer to the data.

    bool wasCopiedToRemote =
        false;  ///< Flag indicating if the data was copied to remote.

    /// Constructor for a Storage that owns data. Used for persistent storage.
    template <DataType Type>
    Storage(std::vector<Type> data)
        : storage(std::move(data)),
          data(std::any_cast<std::vector<Type> &>(storage).data()) {}

    /// Constructor for a Storage that does not own data. Used for temporary
    /// storage.
    Storage(void *data) : data(data) {}

    /// Check if the runtime owns the data storage.
    bool ownsData() const { return storage.has_value(); }

    ~Storage() override = default;
  };

  HostTensor(std::shared_ptr<Storage> storage, DistributedShape shape,
             TileMapping mapping, std::string name, TypeRef type)
      : storage_(std::move(storage)),
        shape_(std::move(shape)),
        mapping_(std::move(mapping)),
        name_(std::move(name)),
        type_(type) {}

 public:
  // Allow default construction, copy and move operations
  HostTensor() = default;
  HostTensor(const HostTensor &) = default;
  HostTensor(HostTensor &&) = default;
  HostTensor &operator=(const HostTensor &) = default;
  HostTensor &operator=(HostTensor &&) = default;

  /// Constructs a persistent HostValue. The data will be stored in the runtime,
  /// so that it is still alive when the memory is actually copied to the IPUs
  /// remote memory at concrete execution.
  template <DataType Type>
  static HostTensor createPersistent(std::vector<Type> data,
                                     DistributedShape shape,
                                     TileMapping mapping, std::string name) {
    return HostTensor(
        Runtime::instance().createResource<Storage>(std::move(data)),
        std::move(shape), std::move(mapping), std::move(name), getType<Type>());
  }

  static HostTensor createTemporary(void *data, DistributedShape shape,
                                    TileMapping mapping, std::string name,
                                    TypeRef type) {
    return HostTensor(std::make_shared<Storage>(data), std::move(shape),
                      std::move(mapping), std::move(name), type);
  }

  /**
   * @brief Destructor for HostValue.
   */
  ~HostTensor() {
    if (storage_ && storage_->ownsData() && !storage_->wasCopiedToRemote)
      Runtime::instance().freeResource(storage_);
  }

  /**
   * @brief Get the data stored in the HostValue.
   */
  const auto &data() const { return storage_->data; }

  /**
   * @brief Get the shape of the data.
   */
  const DistributedShape &shape() const { return shape_; }

  /**
   * @brief Get the tile to tensor mapping.
   */
  const TileMapping &mapping() const { return mapping_; }

  /**
   * @brief Get the name of the HostValue.
   */
  const std::string &name() const { return name_; }

  /**
   * @brief Get the data type of the elements.
   */
  TypeRef type() const { return type_; }

  /**
   * @brief Get the value at the specified indices.
   *
   * @param indices The indices to access.
   * @return Type The value at the specified indices.
   * @throws std::runtime_error if the number of indices does not match the
   * number of dimensions.
   */
  template <DataType Type>
  Type get(std::initializer_list<size_t> indices) const {
    if (indices.size() != shape_.rank()) {
      throw std::runtime_error(
          "The number of indices must match the number of dimensions");
    }

    size_t index = 0;
    for (size_t i = 0; i < shape_.rank(); ++i) {
      index += shape_.stride(i) * *(indices.begin() + i);
    }
    return reinterpret_cast<Type *>(storage_->data)[index];
  }

  template <DataType Type>
  Type getAtFlatIndex(size_t flatIndex) const {
    return reinterpret_cast<Type *>(storage_->data)[flatIndex];
  }

  /**
   * @brief Get the value at the specified index.
   *
   * @param index The index to access.
   * @return Type The value at the specified index.
   */
  template <DataType Type>
  Type get(size_t index) const {
    return get<Type>({index});
  }

  /**
   * @brief Get the total number of elements in the data.
   *
   * @return size_t The number of elements.
   */
  size_t numElements() const { return shape_.numElements(); }

  /**
   * @brief Copy the data to a remote location.
   *
   * @return RemoteTensor<Type> The remote value.
   */
  [[nodiscard]] RemoteTensor copyToRemote() const;

 protected:
  std::shared_ptr<Storage>
      storage_;  ///< Shared pointer to the internal storage.

  DistributedShape shape_;  /// The shape of the tensor.
  TileMapping mapping_;     /// The tile mapping of the tensor.
  std::string name_;        /// Name of the HostValue.
  TypeRef type_;            /// The data type of the elements.
};

namespace detail {
// Determines the number of bytes needed to store an unsigned integer value,
// rounded up to the next power of two.
constexpr size_t getUnsignedIntegerWidthForValue(size_t value) {
  if (value == 0) return 0;
  size_t width = 8;  // in bits
  while ((1UL << width) < value) {
    width *= 2;
  }
  return width / 8;
}
}  // namespace detail

// Some helper functions to construct HostValues from different data types.

/// Constructs a host value by combining the given per-tile arrays, casting it
/// to the concrete data type. Data is a range of ranges (aka a vector of
/// vectors or similiar), where the outer range represents the tiles and the
/// inner range the values of the tile.
template <typename ConcreteType, std::ranges::input_range R>
  requires std::ranges::input_range<std::ranges::range_value_t<R>>
HostTensor constructHostValueForType(R data, std::string name) {
  size_t numTiles = data.size();
  FirstDimDistribution firstDimDistribution;
  firstDimDistribution.reserve(numTiles);

  size_t numValues = 0;
  for (size_t tileID = 0; tileID < numTiles; ++tileID) {
    firstDimDistribution[tileID] = data[tileID].size();
    numValues += data[tileID].size();
  }

  DistributedShape shape =
      DistributedShape::onTiles({numValues}, firstDimDistribution);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  // Combine the data to a single vector with the concrete data type
  std::vector<ConcreteType> concreteData;
  concreteData.reserve(numValues);
  for (const auto &tile : data) {
    std::copy(tile.begin(), tile.end(), std::back_inserter(concreteData));
  }
  TypeRef type = getType<ConcreteType>();
  spdlog::trace(
      "Constructing host value for {} of type {} for {} tiles with {} datums.",
      name, type->str(), numTiles, numValues);
  return HostTensor::createPersistent(std::move(concreteData), std::move(shape),
                                      std::move(mapping), name);
}

/// Constructs a persistent host value by combining the given per-tile arrays,
/// casting it to the smallest possible integer type that can store the values.
/// Data is a range of ranges (aka a vector of vectors or similiar), where the
/// outer range represents the tiles and the inner range the values of each
/// tile.
template <std::ranges::input_range R>
  requires std::ranges::input_range<std::ranges::range_value_t<R>>
HostTensor constructSmallestIntegerHostValue(R data, std::string name) {
  // Determine the required data type
  size_t maxValue = 0;
  for (const auto &tile : data) {
    auto maxTileValue = std::max_element(tile.begin(), tile.end());
    if (maxTileValue != tile.end()) {
      maxValue = std::max(maxValue, *maxTileValue);
    }
  }
  size_t width = detail::getUnsignedIntegerWidthForValue(maxValue);

  switch (width) {
    case 0:
    case 1:
      return constructHostValueForType<uint8_t>(data, name);
    case 2:
      return constructHostValueForType<uint16_t>(data, name);
    case 4:
      return constructHostValueForType<uint32_t>(data, name);
    default:
      throw std::runtime_error(fmt::format(
          "Due to the size of the matrix, the addressing requires {} bits, "
          "but the IPU only support 8, 16, and 32 bit integers.",
          width * 32));
  }
}

}  // namespace graphene