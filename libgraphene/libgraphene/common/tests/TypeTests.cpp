/*
 * Graphene Linear Algebra Framework for Intelligence Processing Units.
 * Copyright (C) 2025 Embedded Systems and Applications, TU Darmstadt.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <gtest/gtest.h>

#include <libgraphene/common/Type.hpp>
#include <poplar/Type.hpp>
#include <sstream>

#include "libgraphene/common/Concepts.hpp"

using namespace graphene;

TEST(TypeTests, BasicProperties) {
  // Test integer types
  EXPECT_TRUE(Type::INT32->isInteger());
  EXPECT_TRUE(Type::INT32->isSigned());
  EXPECT_FALSE(Type::INT32->isFloat());
  EXPECT_EQ(Type::INT32->size(), 4);

  EXPECT_TRUE(Type::UINT32->isInteger());
  EXPECT_FALSE(Type::UINT32->isSigned());

  // Test float types
  EXPECT_TRUE(Type::FLOAT32->isFloat());
  EXPECT_FALSE(Type::FLOAT32->isInteger());
  EXPECT_EQ(Type::FLOAT32->size(), 4);

  // Test bool type
  EXPECT_FALSE(Type::BOOL->isFloat());
  EXPECT_FALSE(Type::BOOL->isInteger());
  EXPECT_EQ(Type::BOOL->size(), 1);

  // Test void type
  EXPECT_TRUE(Type::VOID->isVoid());
  EXPECT_EQ(Type::VOID->size(), 0);
}

TEST(TypeTests, PoplarTypeConversion) {
  EXPECT_EQ(Type::INT8->poplarType(), poplar::CHAR);
  EXPECT_EQ(Type::UINT8->poplarType(), poplar::UNSIGNED_CHAR);
  EXPECT_EQ(Type::INT32->poplarType(), poplar::INT);
  EXPECT_EQ(Type::UINT32->poplarType(), poplar::UNSIGNED_INT);
  EXPECT_EQ(Type::FLOAT32->poplarType(), poplar::FLOAT);
  EXPECT_EQ(Type::BOOL->poplarType(), poplar::BOOL);

  // Test native poplar type support
  EXPECT_TRUE(Type::INT32->isNativePoplarType());
  EXPECT_TRUE(Type::FLOAT32->isNativePoplarType());
  EXPECT_FALSE(Type::FLOAT64->isNativePoplarType());
}

TEST(TypeTests, PoplarEquivalentType) {
  EXPECT_EQ(Type::INT8->poplarEquivalentType(), Type::INT8);
  EXPECT_EQ(Type::UINT8->poplarEquivalentType(), Type::UINT8);
  EXPECT_EQ(Type::INT32->poplarEquivalentType(), Type::INT32);
  EXPECT_EQ(Type::UINT32->poplarEquivalentType(), Type::UINT32);
  EXPECT_EQ(Type::BOOL->poplarEquivalentType(), Type::BOOL);
  EXPECT_EQ(Type::FLOAT32->poplarEquivalentType(), Type::FLOAT32);
  EXPECT_EQ(Type::FLOAT64->poplarEquivalentType(), Type::INT64);
  EXPECT_EQ(Type::TWOFLOAT32->poplarEquivalentType(), Type::INT64);
}

TEST(TypeTests, PointerTypes) {
  auto intPtr = PtrType::get(Type::INT32);
  EXPECT_TRUE(intPtr->isSubscriptable());
  EXPECT_EQ(intPtr->elementType(), Type::INT32);
  EXPECT_EQ(intPtr->size(), 4);  // Pointer size
  EXPECT_EQ(intPtr->str(), "int*");

  auto floatPtr = PtrType::get(Type::FLOAT32);
  EXPECT_EQ(floatPtr->str(), "float*");
}

TEST(TypeTests, TypeStrings) {
  EXPECT_EQ(Type::INT32->str(), "int");
  EXPECT_EQ(Type::UINT32->str(), "unsigned int");
  EXPECT_EQ(Type::FLOAT32->str(), "float");
  EXPECT_EQ(Type::BOOL->str(), "bool");
  EXPECT_EQ(Type::VOID->str(), "void");
  EXPECT_EQ(Type::TWOFLOAT32->str(), "::twofloat::two<float>");
}

TEST(TypeTests, ValuePrinting) {
  std::ostringstream ss;

  int32_t intVal = 42;
  Type::INT32->prettyPrintValue(&intVal, ss);
  EXPECT_EQ(ss.str(), "42");
  ss.str("");

  float floatVal = 3.14f;
  Type::FLOAT32->prettyPrintValue(&floatVal, ss);
  EXPECT_EQ(ss.str(), "3.14");
  ss.str("");

  bool boolVal = true;
  Type::BOOL->prettyPrintValue(&boolVal, ss);
  EXPECT_EQ(ss.str(), "true");
}

TEST(TypeTests, TypeSwitch) {
  auto getSizeOf = []<typename T>() { return sizeof(T); };

  EXPECT_EQ(typeSwitch(Type::INT32, getSizeOf), Type::INT32->size());
  EXPECT_EQ(typeSwitch(Type::FLOAT32, getSizeOf), Type::FLOAT32->size());
  EXPECT_EQ(typeSwitch(Type::BOOL, getSizeOf), Type::BOOL->size());
  EXPECT_EQ(typeSwitch(Type::INT16, getSizeOf), Type::INT16->size());
  // Non-compatabile lambdas should throw
  EXPECT_ANY_THROW(typeSwitch(Type::INT32, []<FloatDataType Type>() {}));
  EXPECT_ANY_THROW(typeSwitch(Type::FLOAT64, []<std::integral Type>() {}));
}

TEST(TypeTests, GetType) {
  EXPECT_EQ(getType<int32_t>(), Type::INT32);
  EXPECT_EQ(getType<uint32_t>(), Type::UINT32);
  EXPECT_EQ(getType<int16_t>(), Type::INT16);
  EXPECT_EQ(getType<uint16_t>(), Type::UINT16);
  EXPECT_EQ(getType<int8_t>(), Type::INT8);
  EXPECT_EQ(getType<uint8_t>(), Type::UINT8);
  EXPECT_EQ(getType<float>(), Type::FLOAT32);
  EXPECT_EQ(getType<bool>(), Type::BOOL);
  EXPECT_EQ(getType<twofloat::two<float>>(), Type::TWOFLOAT32);
}