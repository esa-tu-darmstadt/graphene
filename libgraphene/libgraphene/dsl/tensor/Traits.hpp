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
concept AtLeastOneExpression = (is_expression_v<T1> &&
                                DataTypeOrExpression<T2>) ||
                               (DataTypeOrExpression<T1> &&
                                is_expression_v<T2>);

/**
 * @brief Concept to ensure at least one operand is an Tensor, the other
 * can be a DataType.
 */
template <typename T1, typename T2>
concept AtLeastOneTensor = (is_tensor_v<T1> && DataTypeOrExpression<T2>) ||
                           (DataTypeOrExpression<T1> && is_tensor_v<T2>);

}  // namespace graphene