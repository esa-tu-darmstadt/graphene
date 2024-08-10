#pragma once

#include "libgraphene/dsl/Value.hpp"

namespace graphene::cf {
/**
 * @brief Executes a block of code if the given condition is true.
 *
 * @param condition The condition to evaluate.
 * @param trueBody The block of code to execute if the condition is true.
 * @param falseBody The block of code to execute if the condition is false.
 * (optional)
 */
void If(Expression<bool> condition, std::function<void()> trueBody,
        std::function<void()> falseBody = {});

/**
 * @brief Executes a block of code while the given condition is true.
 *
 * @param condition The condition to evaluate.
 * @param body The block of code to execute while the condition is true.
 */
void While(Expression<bool> condition, std::function<void()> body);

/**
 * @brief Executes a block of code a specified number of times.
 *
 * @param count The number of times to execute the code block.
 * @param body The block of code to execute.
 */
void Repeat(int count, std::function<void()> body);

}  // namespace graphene::cf