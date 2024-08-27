#pragma once

#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/Tensor.hpp"

namespace graphene {

template <DataType... Types>
class HostValueVariant;

/**
 * @class ValueVariant
 * @brief A class representing a \ref Value of any of the given types.
 *
 * This class is a variant of Tensor<Types> for all given types.
 *
 * @tparam Types The types of values that can be stored in the variant.
 */
template <DataType... Types>
class ValueVariant {
  using InnerType = std::variant<Tensor<Types>...>;
  using RemoteValueInnerType = std::variant<RemoteTensor<Types>...>;
  using RemoteValueType = HostValueVariant<Types...>;
  InnerType value_;

 public:
  /**
   * @brief Default constructor.
   */
  ValueVariant() = default;

  /**
   * @brief Constructs a ValueVariant with the given inner variant value.
   *
   * @param value The inner variant value.
   */
  ValueVariant(InnerType value) : value_(std::move(value)) {}

  /**
   * @brief Constructs a ValueVariant with the given Value.
   *
   * @tparam T The type of the Value.
   * @param value The Value to construct the variant with.
   */
  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  ValueVariant(Tensor<T> value) : value_(std::move(value)) {}

  /**
   * @brief Assignment operator for Value.
   *
   * @tparam T The type of the Value.
   * @param value The Value to assign.
   * @return A reference to this ValueVariant.
   */
  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  ValueVariant &operator=(Tensor<T> value) {
    value_ = std::move(value);
    return *this;
  }

  /**
   * @brief Copies the value to remote buffers on the IPU.
   *
   * @return A RemoteValueType object representing the copied remote value.
   */
  RemoteValueType copyToRemote() const {
    return RemoteValueType(std::visit(
        [](auto &&arg) -> RemoteValueInnerType { return arg.copyToRemote(); },
        value_));
  }

  /**
   * @brief Gets the tile to tensor mapping of the value.
   *
   * @return The tile to tensor mapping.
   */
  poplar::Graph::TileToTensorMapping tileMapping() const {
    return std::visit([](auto &&arg) { return arg.tileMapping(); }, value_);
  }

  /**
   * @brief Gets the underlying poplar tensor of the value.
   *
   * @param flattenIfScalar Whether to flatten the tensor if it is a scalar.
   * @return The tensor representation.
   */
  poplar::Tensor tensor(bool flattenIfScalar = false) const {
    return std::visit(
        [flattenIfScalar](auto &&arg) { return arg.tensor(flattenIfScalar); },
        value_);
  }

  /**
   * @brief Gets the underlying poplar tensor of the value that is mapped to the
   * specified tile.
   *
   * @param tile The tile to get the tensor from.
   * @param flattenIfScalar Whether to flatten the tensor if it is a scalar.
   * @return The tensor representation on the specified tile.
   */
  poplar::Tensor tensorOnTile(size_t tile, bool flattenIfScalar = false) const {
    return std::visit(
        [tile, flattenIfScalar](auto &&arg) {
          return arg.tensorOnTile(tile, flattenIfScalar);
        },
        value_);
  }

  /**
   * @brief Prints the value.
   *
   * @param name The name to use for printing.
   * @param fmt The format to use for printing.
   */
  void print(std::string name = "", poplar::PrintTensorFmt fmt = {}) const {
    std::visit([&name, &fmt](auto &&arg) { arg.print(name, fmt); }, value_);
  }
};

/**
 * @brief A type alias for a ValueVariant that can hold any unsigned integer
 * type.
 */
using AnyUIntValue = ValueVariant<uint8_t, uint16_t, uint32_t>;

/**
 * @brief A type alias for a ValueVariant that can hold any signed integer type.
 */
using AnyIntValue = ValueVariant<int8_t, int16_t, int32_t>;

}  // namespace graphene