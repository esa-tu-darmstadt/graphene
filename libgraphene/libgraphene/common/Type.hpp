#pragma once

#include <poplar/Type.hpp>
#include <set>
#include <unordered_map>

#include "libtwofloat/twofloat.hpp"

namespace graphene {

struct Type;
using TypeRef = const Type *;

class Type {
 protected:
  size_t size_;

 public:
  constexpr Type(size_t size) : size_(size) {}
  virtual ~Type() = default;
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
  // TODO: Add better support for function types
  virtual bool hasFunction(std::string func) const { return false; }
  virtual TypeRef functionReturnType(std::string func) const {
    throw std::runtime_error("No functions in this type.");
  }

  /** The C++ string representation of the type. */
  virtual std::string str() const = 0;

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
  std::string str() const override { return "bool"; }
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
        return poplar::LONGLONG;
      default:
        throw std::runtime_error("Unsupported floating point size.");
    }
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
          return impl_ == FloatImpl::IEEE754 ? "double"
                                             : "twofloat::two<float>";
      default:
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
    return FloatType::get(8, FloatImpl::TwoFloat);
  } else if constexpr (std::is_integral_v<T>) {
    return IntegerType::get(bits, std::is_signed_v<T>);
  } else if constexpr (std::is_same_v<T, void>) {
    return VoidType::get();
  } else {
    static_assert(std::is_same_v<T, void>, "Unsupported type.");
  }
}
}  // namespace graphene
