#pragma once

#include <type_traits>
#include <variant>

#include "libgraphene/dsl/tensor/Expression.hpp"

namespace graphene {

/**
 * @brief Type trait to check if a type is an Expression.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_expression_v = std::is_base_of_v<Expression, T>;

/**
 * @brief Type trait to check if a type is an Tensor.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_tensor_v = std::is_same_v<Tensor, T>;

/**
 * @brief Concept to check if a type is a DataType or an Expression of such.
 * @tparam T The type to check.
 */
template <typename T>
concept DataTypeOrExpression = DataType<T> || is_expression_v<T>;

/**
 * @brief Concept to ensure at least one operand is an Expression, the other
 * can be a DataType.
 */
template <typename T1, typename T2>
concept AtLeastOneExpression =
    (is_expression_v<T1> && DataTypeOrExpression<T2>) ||
    (DataTypeOrExpression<T1> && is_expression_v<T2>);

/**
 * @brief Concept to ensure at least one operand is an Tensor, the other
 * can be a DataType.
 */
template <typename T1, typename T2>
concept AtLeastOneTensor = (is_tensor_v<T1> && DataTypeOrExpression<T2>) ||
                           (DataTypeOrExpression<T1> && is_tensor_v<T2>);

}  // namespace graphene