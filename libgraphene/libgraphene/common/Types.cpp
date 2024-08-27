#include "libgraphene/common/Type.hpp"

namespace graphene {
TypeRef Type::BOOL = BoolType::get();
TypeRef Type::FLOAT16 = FloatType::get(16);
TypeRef Type::FLOAT32 = FloatType::get(32);
TypeRef Type::FLOAT64 = FloatType::get(64);
TypeRef Type::TWOFLOAT32 = FloatType::get(32, FloatImpl::TwoFloat);
TypeRef Type::INT8 = IntegerType::get(8, true);
TypeRef Type::UINT8 = IntegerType::get(8, false);
TypeRef Type::INT16 = IntegerType::get(16, true);
TypeRef Type::UINT16 = IntegerType::get(16, false);
TypeRef Type::INT32 = IntegerType::get(32, true);
TypeRef Type::UINT32 = IntegerType::get(32, false);
TypeRef Type::INT64 = IntegerType::get(64, true);
TypeRef Type::UINT64 = IntegerType::get(64, false);
TypeRef Type::VOID = VoidType::get();

}  // namespace graphene