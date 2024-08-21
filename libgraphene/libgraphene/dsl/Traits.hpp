#pragma once

#include <variant>

#include "libgraphene/dsl/Expression.hpp"

namespace graphene {

/**
 * @brief Type trait to check if a type is an Expression.
 * @tparam T The type to check.
 */
template <typename T>
struct is_expression : std::false_type {};
template <typename T>
struct is_expression<Expression<T>> : std::true_type {};
template <typename T>
struct is_expression<Value<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_expression_v = is_expression<T>::value;

/**
 * @brief Type trait to check if a type is an Value.
 * @tparam T The type to check.
 */
template <typename T>
struct is_value : std::false_type {};
template <typename T>
struct is_value<Value<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_value_v = is_value<T>::value;

/**
 * @brief Type trait to unwrap an expression type. If the type is not an
 * expression but a DataType, the type is returned as is.
 * @tparam T The type to unwrap.
 */
template <typename T>
struct unwrap_expression {
  static_assert(DataType<T>, "T must be a DataType");
  using type = T;
};
template <DataType T>
struct unwrap_expression<Expression<T>> {
  using type = T;
};
template <DataType T>
struct unwrap_expression<Value<T>> {
  using type = T;
};

/**
 * @brief Concept to check if a type is a DataType or an Expression of such.
 * @tparam T The type to check.
 */
template <typename T>
concept DataTypeOrExpression =
    DataType<T> ||
    (is_expression_v<T> && DataType<typename unwrap_expression<T>::type>);

/**
 * @brief Concept to check if a type is a native poplar data type or an \ref
 * Expression of such.
 * @tparam T The type to check.
 */
template <typename T>
concept PoplarNativeTypeOrExpression =
    PoplarNativeType<T> ||
    (is_expression_v<T> &&
     PoplarNativeType<typename unwrap_expression<T>::type>);

/**
 * @brief Concept to check if a type is a double word arithmetic type or an \ref
 * Expression of such.
 * @tparam T The type to check.
 */
template <typename T>
concept TwoFloatTypeOrValue =
    TwoFloatType<T> ||
    (is_value_v<T> && TwoFloatType<typename unwrap_expression<T>::type>);

/**
 * @brief Concept to check if a type is a double precision type or an \ref
 * Expression of such.
 * @tparam T The type to check.
 */
template <typename T>
concept DoublePrecisionTypeOrValue =
    DoublePrecisionType<T> ||
    (is_value_v<T> && DoublePrecisionType<typename unwrap_expression<T>::type>);

/**
 * @brief Concept to check if a type is a \ref DataType type or an \ref
 * Expression of such.
 * @tparam T The type to check.
 */
template <typename T>
concept DataTypeOrValue =
    DataType<T> ||
    (is_value_v<T> && DataType<typename unwrap_expression<T>::type>);

/**
 * @brief Concept to ensure at least one operand is an Expression, the other
 * can be a DataType.
 */
template <typename T1, typename T2>
concept AtLeastOneExpression =
    (is_expression_v<T1> && DataTypeOrExpression<T2>) ||
    (DataTypeOrExpression<T1> && is_expression_v<T2>);

template <typename T1, typename T2>
concept AtLeastOneTwoFloatTypeOrValue =
    (TwoFloatTypeOrValue<T1> || TwoFloatTypeOrValue<T2>);

template <typename T1, typename T2>
concept AtLeastOneDoublePrecisionTypeOrValue =
    (DoublePrecisionTypeOrValue<T1> || DoublePrecisionTypeOrValue<T2>);

/**
 * @brief Concept to ensure at least one operand is an Value, the other
 * can be a DataType.
 */
template <typename T1, typename T2>
concept AtLeastOneValue = (is_value_v<T1> && DataTypeOrExpression<T2>) ||
                          (DataTypeOrExpression<T1> && is_value_v<T2>);

/** @brief Concept to check if the unwrapped data types of two expressions are
 * the same.
 */
template <typename T1, typename T2>
concept SameUnwrapedExpressionType =
    std::is_same_v<typename unwrap_expression<T1>::type,
                   typename unwrap_expression<T2>::type>;

}  // namespace graphene