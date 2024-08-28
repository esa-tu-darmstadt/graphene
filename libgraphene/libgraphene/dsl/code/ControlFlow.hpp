#pragma once

#include <poplar/Tensor.hpp>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/dsl/code/CodeGen.hpp"
#include "libgraphene/dsl/code/Value.hpp"

namespace graphene::codedsl {

/**
 * @brief Returns from a function with a given value.
 * @param value The value to return.
 */
void Return(Value value);

/**
 * @brief Prints formatted output.
 * @param formatter The format string as in C's printf. Not all format
 * specifiers are supported, see Poplar's \file{print.h} for details.
 * @param args The argument values.
 */
template <typename... Args>
inline void Printf(std::string formatter, Args... args) {
  CodeGen::emitCode("printf(\"" + formatter + "\"");
  ((CodeGen::emitCode(", " + args.expr())), ...);
  CodeGen::emitCode(");\n");
}

/**
 * @brief Prints a string.
 * @param str The string to print.
 */
void Puts(std::string str);

/**
 * @brief Represents an if-else statement in the CodeDSL language.
 * @param cond The condition Value.
 * @param thenDo The function to execute if the condition is true.
 * @param elseDo The function to execute if the condition is false (optional).
 */
void If(Value cond, std::function<void()> thenDo,
        std::function<void()> elseDo = {});

/**
 * @brief Represents a while loop in the CodeDSL language.
 * @param cond The loop condition Value.
 * @param body The function representing the loop body.
 */
void While(Value cond, std::function<void()> body);

void For(Value start, Value end, Value step, std::function<void(Value)> body,
         bool reverse = false, TypeRef iteratorType = Type::INT32);

}  // namespace graphene::codedsl