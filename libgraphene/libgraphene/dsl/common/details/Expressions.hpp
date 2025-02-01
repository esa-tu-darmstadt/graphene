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

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Type.hpp"

namespace graphene::detail {
enum class TernaryOpType { CLAMP, SELECT };

enum class BinaryOpType {
  ADD,
  ATAN2,
  BITWISE_AND,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_XNOR,
  DIVIDE,
  EQUAL,
  GREATER_THAN_EQUAL,
  GREATER_THAN,
  LESS_THAN_EQUAL,
  LOGICAL_AND,
  LOGICAL_OR,
  LESS_THAN,
  MAXIMUM,
  MINIMUM,
  MULTIPLY,
  NOT_EQUAL,
  POWER,
  REMAINDER,
  SHIFT_LEFT,
  SHIFT_RIGHT,
  SUBTRACT,
};

enum class UnaryOpType {
  ABSOLUTE,
  ASIN,
  BITWISE_NOT,
  CBRT,
  CEIL,
  COS,
  COUNT_LEADING_ZEROS,
  ERF,
  EXPONENT,
  EXPONENT_MINUS_ONE,
  EXPONENT2,
  FLOOR,
  GELU_ERF,
  INVERSE,
  IS_FINITE,
  IS_INF,
  IS_NAN,
  LOGARITHM,
  LOGARITHM_ONE_PLUS,
  LOGICAL_NOT,
  NEGATE,
  NEARBY_INT,
  POPCOUNT,
  RELU,
  SIGNUM,
  SIN,
  TAN,
  TANH,
  ROUND,
  SQRT,
  SQUARE,
  SIGMOID,
  RSQRT,
  TRUNC
};

/**
 * @brief Converts a binary operation type to a string.
 *
 * @param op The binary operation type
 * @return std::string The string representation of the operation
 */
std::string_view to_string(BinaryOpType op);

/**
 * @brief Converts a unary operation type to a string.
 *
 * @param op The unary operation type
 * @return std::string The string representation of the operation
 */
std::string_view to_string(UnaryOpType op);

/**
 * @brief Converts a ternary operation type to a string.
 *
 * @param op The ternary operation type
 * @return std::string The string representation of the operation
 */
std::string_view to_string(TernaryOpType op);

/**
 * @brief Helper function to determine the "larger" of two types.
 *
 * The "larger" type is determined based on
 * the following criteria:
 * 1. The type with the larger size is considered larger.
 * 2. If the sizes are equal, floating-point types take precedence.
 * 3. If the sizes and types are equal, signed types take precedence.
 *
 * @param a The first type
 * @param b The second type
 * @return TypeRef The larger of the two types
 */
TypeRef largerType(TypeRef a, TypeRef b);

/**
 * @brief Infers the resulting type of a binary operation in C++.
 *
 * This function implements the following rules:
 * 1. Arithmetic operations (+, -, *, /, %) return the larger of the two types,
 *    with preference for floating-point types.
 * 2. Bitwise operations work only on integral types and return the larger of
 * the two types.
 * 3. Comparison and logical operations return bool.
 * 4. Special operations like power, atan2, max, and min return the larger of
 * the two types.
 * 5. Some operations (INV_STD_DEV_TO_VARIANCE, VARIANCE_TO_INV_STD_DEV) always
 * return floating-point types.
 *
 * @param op The binary operation type
 * @param lhs The type of the left-hand side operand
 * @param rhs The type of the right-hand side operand
 * @return TypeRef The resulting type of the operation
 * @throws std::runtime_error if the operation is unsupported or the type
 * combination is invalid
 */
TypeRef inferType(BinaryOpType op, TypeRef lhs, TypeRef rhs);

/**
 * @brief Infers the resulting type of a unary operation in C++.
 *
 * This function implements the following rules:
 * 1. Most mathematical functions (sin, cos, exp, log, etc.) return
 * floating-point types.
 * 2. Bitwise operations work only on integral types and preserve the input
 * type.
 * 3. Logical operations and checks (is_finite, is_nan) return bool.
 * 4. Other operations generally preserve the input type.
 *
 * @param op The unary operation type
 * @param operand The type of the operand
 * @return TypeRef The resulting type of the operation
 * @throws std::runtime_error if the operation is unsupported or the type is
 * invalid
 */
TypeRef inferType(UnaryOpType op, TypeRef operand);

/**
 * @brief Infers the resulting type of a ternary operation in C++.
 *
 * This function implements the following rules:
 * 1. CLAMP returns the largest of the three input types.
 * 2. SELECT returns the larger of the second and third arguments if the first
 * is a bool.
 *
 * @param op The ternary operation type
 * @param a The type of the first operand
 * @param b The type of the second operand
 * @param c The type of the third operand
 * @return TypeRef The resulting type of the operation
 * @throws std::runtime_error if the operation is unsupported or the type
 * combination is invalid
 */
TypeRef inferType(TernaryOpType op, TypeRef a, TypeRef b, TypeRef c);

}  // namespace
   // graphene::detail