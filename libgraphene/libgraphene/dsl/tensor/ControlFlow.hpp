#pragma once

#include <functional>
#include <tuple>

#include "libgraphene/dsl/tensor/Tensor.hpp"

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

/**
 * @brief Measures the time taken to execute a block of code.
 *
 * @param body The block of code to execute.
 * @param tile The tile to execute the code on.
 * @return The cycle count taken to execute the block of code.
 */
Tensor<unsigned> Time(std::function<void()> body, size_t tile);

/**
 * @brief Measures the time taken to execute a block of code and returns the
 * result of the block of code.
 *
 * @param body The block of code to execute.
 * @param tile The tile to execute the code on.
 * @return A tuple containing the return value of the block of code and the
 * cycle count taken to execute the block of code.
 */
template <DataType RetType>
std::tuple<Tensor<RetType>, Tensor<unsigned>> Time(
    std::function<Tensor<RetType>()> body, size_t tile);

}  // namespace graphene::cf