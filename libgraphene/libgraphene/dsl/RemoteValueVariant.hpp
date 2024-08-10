#pragma once

#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/RemoteValue.hpp"
#include "libgraphene/dsl/Value.hpp"
namespace graphene {
template <DataType... Types>
class ValueVariant;

template <DataType... Types>
/**
 * @class RemoteValueVariant
 * @brief A class representing a remote value of any of the given types.
 *
 * This class is a variant of RemoteValue<Types> for all given types.
 *
 * @tparam Types The types of remote values that can be stored in the variant.
 */
class RemoteValueVariant {
  using InnerType = std::variant<RemoteValue<Types>...>;
  using ValueInnerType = std::variant<Value<Types>...>;
  using ValueType = ValueVariant<Types...>;
  InnerType value_;

 public:
  RemoteValueVariant() = default;
  RemoteValueVariant(InnerType value) : value_(std::move(value)) {}

  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  /**
   * @brief Assigns a \ref RemoteValue to the \ref RemoteValueVariant.
   *
   * @param value The RemoteValue to be assigned.
   * @return A reference to the updated RemoteValueVariant.
   */
  RemoteValueVariant &operator=(RemoteValue<T> value) {
    value_ = std::move(value);
    return *this;
  }

  /**
   * @brief Copies the value stored in the RemoteValueVariant to tile memory.
   *
   * @return The \ref ValueVariant containing the value copied to tile memory.
   */
  ValueType copyToTile() const {
    return ValueType(std::visit(
        [](auto &&arg) -> ValueInnerType { return arg.copyToTile(); }, value_));
  }
};

/**
 * @brief Alias for a \ref RemoteValueVariant that can hold any unsigned integer
 * type.
 */
using AnyUIntRemoteValue = RemoteValueVariant<uint8_t, uint16_t, uint32_t>;

/**
 * @brief Alias for a \ref RemoteValueVariant that can hold any signed integer
 * type.
 */
using AnyIntRemoteValue = RemoteValueVariant<int8_t, int16_t, int32_t>;
}  // namespace graphene