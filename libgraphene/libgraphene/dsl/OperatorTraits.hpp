#pragma once

#include <popops/ExprOp.hpp>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/Traits.hpp"
#include "libtwofloat/twofloat.hpp"

namespace graphene::ops {
/** @brief Type trait that determines the return type of a unary operation. */
template <popops::expr::UnaryOpType Op, PoplarNativeType T>
struct unary_op_return_type {
  using type = T;
};
template <PoplarNativeType T>
struct unary_op_return_type<popops::expr::UnaryOpType::IS_FINITE, T> {
  static_assert(std::is_same_v<T, float>, "Operand must be float");
  using type = bool;
};

template <PoplarNativeType T>
struct unary_op_return_type<popops::expr::UnaryOpType::LOGICAL_NOT, T> {
  static_assert(std::is_same_v<T, bool>, "Operand must be boolean");
  using type = bool;
};

template <popops::expr::BinaryOpType Op, DataType T1, DataType T2>
struct binary_op_return_type {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = T1;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::EQUAL, T1, T2> {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::GREATER_THAN_EQUAL, T1,
                             T2> {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::GREATER_THAN, T1, T2> {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::LESS_THAN_EQUAL, T1,
                             T2> {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::LOGICAL_AND, T1, T2> {
  static_assert(std::is_same_v<T1, T2> && std::is_same_v<T1, bool>,
                "Operands must be bool");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::LOGICAL_OR, T1, T2> {
  static_assert(std::is_same_v<T1, T2> && std::is_same_v<T1, bool>,
                "Operands must be bool");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::LESS_THAN, T1, T2> {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = bool;
};

template <PoplarNativeType T1, PoplarNativeType T2>
struct binary_op_return_type<popops::expr::BinaryOpType::NOT_EQUAL, T1, T2> {
  static_assert(std::is_same_v<T1, T2>, "Operands must have the same type");
  using type = bool;
};

// Specializations for double word arithmetic
template <>
struct binary_op_return_type<popops::expr::BinaryOpType::ADD, doubleword,
                             float> {
  using type = doubleword;
};
template <>
struct binary_op_return_type<popops::expr::BinaryOpType::ADD, float,
                             doubleword> {
  using type = doubleword;
};
template <>
struct binary_op_return_type<popops::expr::BinaryOpType::ADD, doubleword,
                             doubleword> {
  using type = doubleword;
};

// Specializations for double precision arithmetic
template <>
struct binary_op_return_type<popops::expr::BinaryOpType::ADD, double, float> {
  using type = double;
};
template <>
struct binary_op_return_type<popops::expr::BinaryOpType::ADD, float, double> {
  using type = double;
};
template <>
struct binary_op_return_type<popops::expr::BinaryOpType::ADD, double, double> {
  using type = double;
};

/** A concept that makes sure that two types are legal for the given binary op
 * type. Unwraps the expression data type if necessary. */
template <popops::expr::BinaryOpType Op, typename T1, typename T2>
concept CompatibleTypesForBinaryOp = requires {
  typename binary_op_return_type<Op, typename unwrap_expression<T1>::type,
                                 typename unwrap_expression<T2>::type>::type;
};

/** A concept that makes sure that two types are legal for the given unary op
 * type. Unwraps the expression data type if necessary. */
template <popops::expr::UnaryOpType Op, typename T>
concept CompatibleTypeForUnaryOp = requires {
  typename unary_op_return_type<Op, typename unwrap_expression<T>::type>::type;
};

}  // namespace graphene::ops