#pragma once

#include <optional>
#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/common/details/Expressions.hpp"
#include "libgraphene/dsl/tensor/Expression.hpp"
#include "libgraphene/dsl/tensor/Traits.hpp"
#include "libgraphene/dsl/tensor/details/Expressions.hpp"

namespace graphene {

namespace detail {
/**
 * @brief Helper function to wrap a DataType in an Expression if it's not
 * already one.
 */
template <DataTypeOrExpression T>
Expression wrapInExpression(const T &value) {
  if constexpr (is_expression_v<T>) {
    return value;
  } else {
    return Expression(value);
  }
}

}  // namespace detail

#define GRAPHENE_DEFINE_EXPR_UNARY_OP(name, op)            \
  namespace ops {                                          \
  inline Expression name(const Expression &value) {        \
    return Expression(std::make_unique<detail::UnaryExpr>( \
        detail::UnaryOpType::op, value.base().clone()));   \
  }                                                        \
  }  // namespace ops

#define GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(name, op, symbol) \
  GRAPHENE_DEFINE_EXPR_UNARY_OP(name, op)                          \
  inline Expression operator symbol(const Expression &value) {     \
    return ops::name(value);                                       \
  }

GRAPHENE_DEFINE_EXPR_UNARY_OP(Abs, ABSOLUTE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Asin, ASIN)
GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(BitwiseNot, BITWISE_NOT, ~)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Cbrt, CBRT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Erf, ERF)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Ceil, CEIL)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Cos, COS)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Exp, EXPONENT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Expm1, EXPONENT_MINUS_ONE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Exp2, EXPONENT2)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Floor, FLOOR)
GRAPHENE_DEFINE_EXPR_UNARY_OP(GeluErf, GELU_ERF)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Inv, INVERSE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(IsFinite, IS_FINITE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(IsInf, IS_INF)
GRAPHENE_DEFINE_EXPR_UNARY_OP(IsNaN, IS_NAN)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Log, LOGARITHM)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Log1p, LOGARITHM_ONE_PLUS)
GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Not, LOGICAL_NOT, !)
GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Neg, NEGATE, -)
GRAPHENE_DEFINE_EXPR_UNARY_OP(NearbyInt, NEARBY_INT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Signum, SIGNUM)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Sin, SIN)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Tan, TAN)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Tanh, TANH)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Round, ROUND)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Trunc, TRUNC)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Sqrt, SQRT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Square, SQUARE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Sigmoid, SIGMOID)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Rsqrt, RSQRT)

#define GRAPHENE_DEFINE_EXPR_BINARY_OP(name, op)              \
  namespace ops {                                             \
                                                              \
  template <DataTypeOrExpression T1, DataTypeOrExpression T2> \
    requires AtLeastOneExpression<T1, T2>                     \
  inline Expression name(const T1 &lhs, const T2 &rhs) {      \
    auto lhsExpr = detail::wrapInExpression(lhs);             \
    auto rhsExpr = detail::wrapInExpression(rhs);             \
                                                              \
    return Expression(std::make_unique<detail::BinaryExpr>(   \
        detail::BinaryOpType::op, lhsExpr.base().clone(),     \
        rhsExpr.base().clone()));                             \
  }                                                           \
  }  // namespace ops

#define GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(name, op, symbol) \
  GRAPHENE_DEFINE_EXPR_BINARY_OP(name, op)                          \
  template <DataTypeOrExpression T1, DataTypeOrExpression T2>       \
    requires AtLeastOneExpression<T1, T2>                           \
  inline Expression operator symbol(const T1 &lhs, const T2 &rhs) { \
    return ops::name(lhs, rhs);                                     \
  }

GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Add, ADD, +)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Atan2, ATAN2)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseAnd, BITWISE_AND, &)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseOr, BITWISE_OR, |)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseXor, BITWISE_XOR, ^)
GRAPHENE_DEFINE_EXPR_BINARY_OP(BitwiseXnor, BITWISE_XNOR)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Divide, DIVIDE, /)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Equal, EQUAL, ==)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Gte, GREATER_THAN_EQUAL, >=)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Gt, GREATER_THAN, >)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Lte, LESS_THAN_EQUAL, <=)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(And, LOGICAL_AND, &&)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Or, LOGICAL_OR, ||)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Lt, LESS_THAN, <)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Max, MAXIMUM)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Min, MINIMUM)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Mul, MULTIPLY, *)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(NotEqual, NOT_EQUAL, !=)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Pow, POWER)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Rem, REMAINDER, %)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Shl, SHIFT_LEFT, <<)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Shr, SHIFT_RIGHT, >>)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Sub, SUBTRACT, -)

#undef GRAPHENE_DEFINE_EXPR_BINARY_OP
#undef GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL

}  // namespace graphene