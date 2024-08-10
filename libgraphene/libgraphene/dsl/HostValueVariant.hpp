#pragma once

#include <initializer_list>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/HostValue.hpp"

namespace graphene {
template <DataType... Types>
class RemoteValueVariant;

/**
 * @class HostValueVariant
 * @brief A class representing a \ref HostValue of any of the given types.
 *
 * This class is a variant of HostValue<Types> for all given types.
 *
 * @tparam Types The types of host values that can be stored in the variant.
 */
template <DataType... Types>
class HostValueVariant {
  using InnerType = std::variant<HostValue<Types>...>;
  using RemoteValueInnerType = std::variant<RemoteValue<Types>...>;
  using RemoteValueType = RemoteValueVariant<Types...>;
  InnerType value_;

 public:
  /**
   * @brief Default constructor.
   */
  HostValueVariant() = default;

  /**
   * @brief Constructs a HostValueVariant with the given inner variant value.
   *
   * @param value The inner variant value.
   */
  HostValueVariant(InnerType value) : value_(std::move(value)) {}

  /**
   * @brief Assignment operator for HostValue.
   *
   * @tparam T The type of the HostValue.
   * @param value The HostValue to assign.
   * @return A reference to this HostValueVariant.
   */
  template <typename T>
    requires(std::is_same_v<T, Types> || ...)
  HostValueVariant &operator=(HostValue<T> value) {
    value_ = std::move(value);
    return *this;
  }

  /**
   * @brief Copies the host value to remote buffers on the IPU.
   *
   * @return A RemoteValueType object representing the copied remote value.
   */
  RemoteValueType copyToRemote() const {
    return RemoteValueType(std::visit(
        [](auto &&arg) -> RemoteValueInnerType { return arg.copyToRemote(); },
        value_));
  }

  /**
   * @brief Gets an element from the host value.
   *
   * @tparam T The type of the element to get.
   * @param indices The indices of the element.
   * @return The element at the specified indices.
   */
  template <typename T>
  T get(std::initializer_list<size_t> indices) const {
    return std::visit(
        [&indices](auto &&arg) -> T { return (T)arg.get(indices); }, value_);
  }

  /**
   * @brief Gets the number of elements in the host value.
   *
   * @return The number of elements.
   */
  size_t numElements() const {
    return std::visit([](auto &&arg) -> size_t { return arg.numElements(); },
                      value_);
  }
};

/**
 * @brief A type alias for a HostValueVariant that can hold any unsigned integer
 * type.
 */
using AnyUIntHostValue = HostValueVariant<uint8_t, uint16_t, uint32_t>;

}  // namespace graphene