#include "libgraphene/common/Type.hpp"

#include <poplar/Type.hpp>

namespace graphene {
TypeRef Type::BOOL = BoolType::get();
TypeRef Type::FLOAT16 = FloatType::get(16);
TypeRef Type::FLOAT32 = FloatType::get(32);
TypeRef Type::FLOAT64 = FloatType::get(64);
TypeRef Type::TWOFLOAT32 = FloatType::get(64, FloatImpl::TwoFloat);
TypeRef Type::INT8 = IntegerType::get(8, true);
TypeRef Type::UINT8 = IntegerType::get(8, false);
TypeRef Type::INT16 = IntegerType::get(16, true);
TypeRef Type::UINT16 = IntegerType::get(16, false);
TypeRef Type::INT32 = IntegerType::get(32, true);
TypeRef Type::UINT32 = IntegerType::get(32, false);
TypeRef Type::INT64 = IntegerType::get(64, true);
TypeRef Type::UINT64 = IntegerType::get(64, false);
TypeRef Type::VOID = VoidType::get();

TypeRef getType(poplar::Type type) {
  if (type == poplar::BOOL) {
    return Type::BOOL;
  } else if (type == poplar::HALF) {
    return Type::FLOAT16;
  } else if (type == poplar::FLOAT) {
    return Type::FLOAT32;
  } else if (type == poplar::UNSIGNED_CHAR) {
    return Type::UINT8;
  } else if (type == poplar::CHAR) {
    return Type::INT8;
  } else if (type == poplar::UNSIGNED_SHORT) {
    return Type::UINT16;
  } else if (type == poplar::SHORT) {
    return Type::INT16;
  } else if (type == poplar::UNSIGNED_INT) {
    return Type::UINT32;
  } else if (type == poplar::INT) {
    return Type::INT32;
  } else if (type == poplar::UNSIGNED_LONGLONG) {
    return Type::UINT64;
  } else if (type == poplar::LONGLONG) {
    throw std::runtime_error(
        "LONGLONG is ambiguous, as it can be used as a int64_t, double or "
        "two<float>.");
  } else {
    throw std::runtime_error("Unsupported type.");
  }
}

TypeRef parseType(std::string name) {
#define CHECK_TYPE(NAME, TYPE) \
  if (name == NAME) return TYPE;

  CHECK_TYPE("bool", Type::BOOL)
  CHECK_TYPE("float16", Type::FLOAT16)
  CHECK_TYPE("float32", Type::FLOAT32)
  CHECK_TYPE("float64", Type::FLOAT64)
  CHECK_TYPE("twofloat32", Type::TWOFLOAT32)
  CHECK_TYPE("int8", Type::INT8)
  CHECK_TYPE("uint8", Type::UINT8)
  CHECK_TYPE("int16", Type::INT16)
  CHECK_TYPE("uint16", Type::UINT16)
  CHECK_TYPE("int32", Type::INT32)
  CHECK_TYPE("uint32", Type::UINT32)
  CHECK_TYPE("int64", Type::INT64)
  CHECK_TYPE("uint64", Type::UINT64)
  CHECK_TYPE("void", Type::VOID)
  return nullptr;
}

}  // namespace graphene