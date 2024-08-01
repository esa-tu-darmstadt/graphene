#pragma once

#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/RemoteValue.hpp"
#include "libgraphene/dsl/Value.hpp"
namespace graphene {
template <DataType... Types>
class ValueVariant;

template <DataType... Types>
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
  RemoteValueVariant &operator=(RemoteValue<T> value) {
    value_ = std::move(value);
    return *this;
  }

  ValueType copyToTile() const {
    return ValueType(std::visit(
        [](auto &&arg) -> ValueInnerType { return arg.copyToTile(); }, value_));
  }
};

using AnyUIntValue = ValueVariant<uint8_t, uint16_t, uint32_t>;

using AnyIntValue = ValueVariant<int8_t, int16_t, int32_t>;
}  // namespace graphene