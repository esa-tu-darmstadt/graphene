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

#include <functional>
#include <tuple>

#include "libgraphene/dsl/tensor/Tensor.hpp"

namespace graphene::cf {
/**
 * @brief Executes a block of code if the given condition is true.
 *
 * @param condition The condition to evaluate. Must be a boolean expression.
 * @param trueBody The block of code to execute if the condition is true.
 * @param falseBody The block of code to execute if the condition is false.
 * (optional)
 */
void If(Expression condition, std::function<void()> trueBody,
        std::function<void()> falseBody = {});

/**
 * @brief Executes a block of code while the given condition is true.
 *
 * @param condition The condition to evaluate. Must be a boolean expression.
 * @param body The block of code to execute while the condition is true.
 */
void While(Expression condition, std::function<void()> body);

/**
 * @brief Executes a block of code a specified number of times.
 *
 * @param count The number of times to execute the code block.
 * @param body The block of code to execute.
 */
void Repeat(int count, std::function<void()> body);

/**
 * @brief Measures the time taken to execute a block of code.
 *
 * @param body The block of code to execute.
 * @param tile The tile on which the program is timed.
 * @return A uint32_t scalar tensor storing the cycle count taken to execute the
 * block of code.
 */
Tensor Time(std::function<void()> body, size_t tile);

/**
 * @brief Measures the time taken to execute a block of code and returns the
 * result of the block of code.
 *
 * @param body The block of code to execute.
 * @param tile The tile on which the program is timed.
 * @return A tuple containing the return value of the block of code and the
 * cycle count taken to execute the block of code.
 */
std::tuple<Tensor, Tensor> Time(std::function<Tensor()> body, size_t tile);

}  // namespace graphene::cf