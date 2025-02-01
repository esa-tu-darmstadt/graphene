#pragma once

#include <functional>
#include <poplar/Type.hpp>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include "libtwofloat/twofloat.hpp"

namespace graphene {

struct CTypeQualifiers {
  bool Volatile = false;
  bool Const = false;
  bool Restrict = false;
  static CTypeQualifiers getVolatile() { return {true, false, false}; }
  static CTypeQualifiers getConst() { return {false, true, false}; }
  static CTypeQualifiers getRestrict() { return {false, false, true}; }
};

struct Type;
using TypeRef = const Type *;

class Type {
 protected:
  /// Size of the type in bytes
  size_t size_;

 public:
  constexpr Type(size_t size) : size_(size) {}
  virtual ~Type() = default;

  /// Returns the size of the type in bytes
  size_t size() const { return size_; }

  virtual bool isFloat() const { return false; }
  virtual bool isInteger() const { return false; }
  virtual bool isSigned() const { return true; }
  virtual bool isVoid() const { return false; }
  virtual bool isSubscriptable() const { return false; }
  virtual TypeRef elementType() const {
    throw std::runtime_error("Type is not subscriptable.");
  }
  virtual poplar::Type poplarType() const {
    throw std::runtime_error("Type is not associated with a poplar type.");
  }
  /// Returns true if the type is natively supported by Poplar
  virtual bool isNativePoplarType() const { return false; }

  /// Returns the type by which this type is represented in Poplar or nullptr if
  /// the type cannot be represented in Poplar
  virtual TypeRef poplarEquivalentType() const {
    return nullptr;  // No equivalent type
  }
  // TODO: Add better support for function types
  virtual bool hasFunction(std::string func) const { return false; }
  virtual TypeRef functionReturnType(std::string func) const {
    throw std::runtime_error("No functions in this type.");
  }

  /** The C++ string representation of the type. */
  virtual std::string str() const = 0;

  /// Prints the value at the given address to a stream in a human-readable
  /// format
  virtual void prettyPrintValue(const void *value, std::ostream &stream) const {
    throw std::runtime_error("Type does not support printing values.");
  }

  // Shortcuts for common types
  static TypeRef BOOL;
  static TypeRef FLOAT16;
  static TypeRef FLOAT32;
  static TypeRef FLOAT64;
  static TypeRef TWOFLOAT32;
  static TypeRef INT8;
  static TypeRef UINT8;
  static TypeRef INT16;
  static TypeRef UINT16;
  static TypeRef INT32;
  static TypeRef UINT32;
  static TypeRef INT64;
  static TypeRef UINT64;
  static TypeRef VOID;
};

class BoolType : public Type {
 protected:
  constexpr BoolType() : Type(1) {}

 public:
  static BoolType *get() {
    constinit static BoolType instance;
    return &instance;
  }
  poplar::Type poplarType() const override { return poplar::BOOL; }
  bool isNativePoplarType() const override { return true; }
  TypeRef poplarEquivalentType() const override { return this; }
  std::string str() const override { return "bool"; }
  void prettyPrintValue(const void *value,
                        std::ostream &stream) const override {
    stream << (*reinterpret_cast<const bool *>(value) ? "true" : "false");
  }
};
enum class FloatImpl : size_t { IEEE754 = 0, TwoFloat = 1 };
class FloatType : public Type {
 public:
 private:
  FloatImpl impl_;

 protected:
  constexpr FloatType(size_t bits, FloatImpl impl = FloatImpl::IEEE754)
      : Type(bits / 8), impl_(impl) {
    if (bits != 16 && bits != 32 && bits != 64) {
      throw std::runtime_error("Unsupported floating point size.");
    }
    if (impl_ == FloatImpl::TwoFloat && bits != 64) {
      throw std::runtime_error("TwoFloat only supports 64-bit floats.");
    }
  }

 public:
  static FloatType *get(size_t bits, FloatImpl impl = FloatImpl::IEEE754) {
    constinit static FloatType nativeFloat16(16);
    constinit static FloatType nativeFloat32(32);
    constinit static FloatType nativeFloat64(64);
    constinit static FloatType twoFloat64(64, FloatImpl::TwoFloat);

    switch (bits) {
      case 16:
        return &nativeFloat16;
      case 32:
        return &nativeFloat32;
      case 64:
        return impl == FloatImpl::IEEE754 ? &nativeFloat64 : &twoFloat64;
      default:
        throw std::runtime_error("Unsupported floating point size.");
    }
  }

  poplar::Type poplarType() const override {
    switch (size_) {
      case 2:
        return poplar::HALF;
      case 4:
        return poplar::FLOAT;
      case 8:
        throw std::runtime_error(
            "Poplar does not support 64 bit floats. Maybe you meant to use "
            "poplarEquivalentType()->poplarType() instead?");
      default:
        throw std::runtime_error("Unsupported floating point size.");
    }
  }
  bool isNativePoplarType() const override { return size_ == 2 || size_ == 4; }
  TypeRef poplarEquivalentType() const override {
    if (size_ == 8) return Type::INT64;
    return this;
  }
  bool isFloat() const override { return true; }
  std::string str() const override {
    switch (size_) {
      case 2:
        return "half";
      case 4:
        return "float";
      case 8:
        if (impl_ == FloatImpl::TwoFloat)
          return "::twofloat::two<float>";
        else if (impl_ == FloatImpl::IEEE754)
          return "double";
      default:
        throw std::runtime_error("Unsupported floating point size.");
    }
  }

  void prettyPrintValue(const void *value,
                        std::ostream &stream) const override {
    if (size_ == 2) {
      throw std::runtime_error("Half precision not yet supported.");
    } else if (size_ == 4) {
      stream << *reinterpret_cast<const float *>(value);
    } else if (size_ == 8) {
      if (impl_ == FloatImpl::TwoFloat) {
        auto twoFloat = *reinterpret_cast<const twofloat::two<float> *>(value);
        stream << "{ " << twoFloat.h << " + " << twoFloat.l << " }";
      } else {
        stream << *reinterpret_cast<const double *>(value);
      }
    } else {
      throw std::runtime_error("Unsupported floating point size.");
    }
  }
};
class IntegerType : public Type {
  bool isSigned_;

 protected:
  constexpr IntegerType(size_t bits, bool isSigned)
      : Type(bits / 8), isSigned_(isSigned) {
    if (bits != 8 && bits != 16 && bits != 32 && bits != 64) {
      throw std::runtime_error("Unsupported integer size.");
    }
  }

 public:
  static IntegerType *get(size_t bits, bool isSigned) {
    constinit static IntegerType nativeInt8(8, true);
    constinit static IntegerType nativeUInt8(8, false);
    constinit static IntegerType nativeInt16(16, true);
    constinit static IntegerType nativeUInt16(16, false);
    constinit static IntegerType nativeInt32(32, true);
    constinit static IntegerType nativeUInt32(32, false);
    constinit static IntegerType nativeInt64(64, true);
    constinit static IntegerType nativeUInt64(64, false);

    switch (bits) {
      case 8:
        return isSigned ? &nativeInt8 : &nativeUInt8;
      case 16:
        return isSigned ? &nativeInt16 : &nativeUInt16;
      case 32:
        return isSigned ? &nativeInt32 : &nativeUInt32;
      case 64:
        return isSigned ? &nativeInt64 : &nativeUInt64;
      default:
        throw std::runtime_error("Unsupported integer size.");
    }
  }
  poplar::Type poplarType() const override {
    switch (size_) {
      case 1:
        return isSigned_ ? poplar::CHAR : poplar::UNSIGNED_CHAR;
      case 2:
        return isSigned_ ? poplar::SHORT : poplar::UNSIGNED_SHORT;
      case 4:
        return isSigned_ ? poplar::INT : poplar::UNSIGNED_INT;
      case 8:
        return isSigned_ ? poplar::LONGLONG : poplar::UNSIGNED_LONGLONG;
      default:
        throw std::runtime_error("Unsupported integer size.");
    }
  }
  bool isNativePoplarType() const override { return true; }
  TypeRef poplarEquivalentType() const override { return this; }
  bool isInteger() const override { return true; }
  bool isSigned() const override { return isSigned_; }
  std::string str() const override {
    switch (size_) {
      case 1:
        return isSigned_ ? "char" : "unsigned char";
      case 2:
        return isSigned_ ? "short" : "unsigned short";
      case 4:
        return isSigned_ ? "int" : "unsigned int";
      case 8:
        return isSigned_ ? "long long" : "unsigned long long";
      default:
        throw std::runtime_error("Unsupported integer size.");
    }
  }

  void prettyPrintValue(const void *value,
                        std::ostream &stream) const override {
    switch (size_) {
      case 1:
        if (isSigned_) {
          stream << *reinterpret_cast<const int8_t *>(value);
        } else {
          stream << *reinterpret_cast<const uint8_t *>(value);
        }
        break;
      case 2:
        if (isSigned_) {
          stream << *reinterpret_cast<const int16_t *>(value);
        } else {
          stream << *reinterpret_cast<const uint16_t *>(value);
        }
        break;
      case 4:
        if (isSigned_) {
          stream << *reinterpret_cast<const int32_t *>(value);
        } else {
          stream << *reinterpret_cast<const uint32_t *>(value);
        }
        break;
      case 8:
        if (isSigned_) {
          stream << *reinterpret_cast<const int64_t *>(value);
        } else {
          stream << *reinterpret_cast<const uint64_t *>(value);
        }
        break;
      default:
        throw std::runtime_error("Unsupported integer size.");
    }
  }
};
class VoidType : public Type {
 protected:
  constexpr VoidType() : Type(0) {}

 public:
  static VoidType *get() {
    constinit static VoidType instance;
    return &instance;
  }
  bool isVoid() const override { return true; }
  std::string str() const override { return "void"; }
};

class PtrType : public Type {
 protected:
  TypeRef elementType_;

 public:
  constexpr PtrType(TypeRef elementType) : Type(4), elementType_(elementType) {}
  static PtrType *get(TypeRef elementType) {
    static std::unordered_map<TypeRef, std::unique_ptr<PtrType>> instances;
    auto it = instances.find(elementType);
    if (it != instances.end()) {
      return it->second.get();
    }
    auto instance = std::make_unique<PtrType>(elementType);
    return (instances[elementType] = std::move(instance)).get();
  }
  TypeRef elementType() const override { return elementType_; }
  bool isSubscriptable() const override { return true; }
  std::string str() const override { return elementType_->str() + "*"; }
};

template <typename T>
constexpr TypeRef getType() {
  size_t bits = sizeof(T) * 8;
  if constexpr (std::is_same_v<T, bool>) {
    return BoolType::get();
  } else if constexpr (std::is_floating_point_v<T>) {
    return FloatType::get(bits);
  } else if constexpr (std::is_same_v<T, twofloat::two<float>>) {
    return FloatType::get(64, FloatImpl::TwoFloat);
  } else if constexpr (std::is_integral_v<T>) {
    return IntegerType::get(bits, std::is_signed_v<T>);
  } else if constexpr (std::is_same_v<T, void>) {
    return VoidType::get();
  } else {
    static_assert(std::is_same_v<T, void>, "Unsupported type.");
  }
}

TypeRef getType(poplar::Type type);

namespace details {
// A helper concept to check if f.template operator()<T>(args...) is valid.
template <typename T, typename F, typename... Args>
concept TypeSwitchInvocable = requires(F &&f, Args &&...args) {
  // We only test for the expression below; if it's invalid, it fails SFINAE.
  { std::forward<F>(f).template operator()<T>(std::forward<Args>(args)...) };
};
}  // namespace details

/// Calls the given function with the compile time type corresponding to the
/// given dynamic type.
template <class F, class... Args>
decltype(auto) typeSwitch(TypeRef type, F &&f, Args &&...args) {
#define DISPATCH(TYPEREF, CPP_TYPE)                                          \
  if (type == TYPEREF) {                                                     \
    if constexpr (details::TypeSwitchInvocable<CPP_TYPE, F, Args...>)        \
      return std::forward<F>(f).template operator()<CPP_TYPE>(               \
          std::forward<Args>(args)...);                                      \
    else                                                                     \
      throw std::invalid_argument("Function does not accept type " #CPP_TYPE \
                                  ".");                                      \
  }

  DISPATCH(Type::BOOL, bool)
  DISPATCH(Type::FLOAT16, __fp16)  // Does this work?
  DISPATCH(Type::FLOAT32, float)
  DISPATCH(Type::FLOAT64, double)
  DISPATCH(Type::TWOFLOAT32, ::twofloat::two<float>)
  DISPATCH(Type::INT8, int8_t)
  DISPATCH(Type::UINT8, uint8_t)
  DISPATCH(Type::INT16, int16_t)
  DISPATCH(Type::UINT16, uint16_t)
  DISPATCH(Type::INT32, int32_t)
  DISPATCH(Type::UINT32, uint32_t)
  DISPATCH(Type::INT64, int64_t)
  DISPATCH(Type::UINT64, uint64_t)

#undef DISPATCH

  throw std::invalid_argument("Unsupported or unknown type in typeSwitch.");
}

/// Returns the type corresponding to the given string or \c nullptr if the
/// string does not correspond to a known type. Supported type strings are:
/// - "bool"
/// - "float16", "float32", "float64", "twofloat32"
/// - "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"
TypeRef parseType(std::string name);

}  // namespace graphene