#pragma once

#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/Value.hpp"
namespace graphene {
template <DataType... Types>
class HostValueVariant;

template <DataType... Types>
class ValueVariant {
  using InnerType = std::variant<Value<Types>...>;
  using RemoteValueInnerType = std::variant<RemoteValue<Types>...>;
  using RemoteValueType = HostValueVariant<Types...>;
  InnerType value_;

 public:
  ValueVariant() = default;
  ValueVariant(InnerType value) : value_(std::move(value)) {}

  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  ValueVariant(Value<T> value) : value_(std::move(value)) {}

  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  ValueVariant &operator=(Value<T> value) {
    value_ = std::move(value);
    return *this;
  }

  RemoteValueType copyToRemote() const {
    return RemoteValueType(std::visit(
        [](auto &&arg) -> RemoteValueInnerType { return arg.copyToRemote(); },
        value_));
  }

  poplar::Graph::TileToTensorMapping tileMapping() const {
    return std::visit([](auto &&arg) { return arg.tileMapping(); }, value_);
  }

  poplar::Tensor tensor(bool flattenIfScalar = false) const {
    return std::visit(
        [flattenIfScalar](auto &&arg) { return arg.tensor(flattenIfScalar); },
        value_);
  }

  poplar::Tensor tensorOnTile(size_t tile, bool flattenIfScalar = false) const {
    return std::visit(
        [tile, flattenIfScalar](auto &&arg) {
          return arg.tensorOnTile(tile, flattenIfScalar);
        },
        value_);
  }

  void print(std::string name = "", poplar::PrintTensorFmt fmt = {}) const {
    std::visit([&name, &fmt](auto &&arg) { arg.print(name, fmt); }, value_);
  }
};

using AnyUIntValue = ValueVariant<uint8_t, uint16_t, uint32_t>;

using AnyIntValue = ValueVariant<int8_t, int16_t, int32_t>;
}  // namespace graphene