#pragma once

#include "libgraphene/dsl/Value.hpp"

namespace graphene {

/** Represents a type-agnostic, assignable tensor. This is a type erasured
 * version of \ref Value. */
class AnyValue {
  struct Concept {
    virtual ~Concept() = default;
    virtual void assign(const AnyValue &other) = 0;

    template <DataType Type>
    void assign(const Expression<Type> &other);
  };

  template <DataType Type>
  struct Model : Concept {
    Value<Type> value;

    Model(const Value<Type> &value) : value(value) {}

    void assign(const AnyValue &other) override {}

    template <DataType OtherType>
    void assign(const Expression<OtherType> &other) {
      if (!std::is_same_v<Type, OtherType>) {
        throw std::runtime_error("Cannot assign different types");
      }
      value = other;
    }
  };

  std::unique_ptr<Concept> pimp;

 public:
  template <DataType Type>
  AnyValue(const Value<Type> &value)
      : pimp(std::make_unique<Model<Type>>(value)) {}

  AnyValue &operator=(const AnyValue &other) {
    pimp->assign(other);
    return *this;
  }

  template <DataType Type>
  AnyValue &operator=(const Expression<Type> &other) {
    pimp->assign(other);
    return *this;
  }
};
}  // namespace graphene