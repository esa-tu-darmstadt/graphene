#pragma once

#include <initializer_list>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/HostValue.hpp"
namespace graphene {
template <DataType... Types>
class RemoteValueVariant;

template <DataType... Types>
class HostValueVariant {
  using InnerType = std::variant<HostValue<Types>...>;
  using RemoteValueInnerType = std::variant<RemoteValue<Types>...>;
  using RemoteValueType = RemoteValueVariant<Types...>;
  InnerType value_;

 public:
  HostValueVariant() = default;
  HostValueVariant(InnerType value) : value_(std::move(value)) {}

  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  HostValueVariant &operator=(HostValue<T> value) {
    value_ = std::move(value);
    return *this;
  }

  RemoteValueType copyToRemote() const {
    return RemoteValueType(std::visit(
        [](auto &&arg) -> RemoteValueInnerType { return arg.copyToRemote(); },
        value_));
  }

  template <typename T>
  T get(std::initializer_list<size_t> indices) const {
    return std::visit(
        [&indices](auto &&arg) -> T { return (T)arg.get(indices); }, value_);
  }

  size_t numElements() const {
    return std::visit([](auto &&arg) -> size_t { return arg.numElements(); },
                      value_);
  }
};

using AnyUIntHostValue = HostValueVariant<uint8_t, uint16_t, uint32_t>;
}  // namespace graphene