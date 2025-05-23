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
  CodeGen::emitCode("printf(\"" + formatter + "\\n\"");
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
 * @brief Represents a break statement in the CodeDSL language.
 */
void Break();

/**
 * @brief Represents a continue statement in the CodeDSL language.
 */
void Continue();

/**
 * @brief Represents a while loop in the CodeDSL language.
 * @param cond The loop condition Value.
 * @param body The function representing the loop body.
 */
void While(Value cond, std::function<void()> body);

/**
 * @brief Represents a for loop in the CodeDSL language. Iterates from start
 * (inclusive) to end (exclusive) with a given step size in positive direction.
  @details for(auto i = start; i < end; i += step) { body(i); }
 */
void For(Value start, Value end, Value step, std::function<void(Value)> body,
         TypeRef iteratorType = Type::INT32);

/**
 * @brief Represents a for loop in the CodeDSL language. Iterates from start
 * (inclusive) to end (INCLUSIVE) with a given step size, in reverse direction.
 * @details for(auto i = start; i > end; i -= step) { body(i); }
 */
void ForReverse(Value start, Value end, Value step,
                std::function<void(Value)> body,
                TypeRef iteratorType = Type::INT32);

namespace detail {
Variable ForStart(Value start, Value end, Value step, bool reverse = false,
                  TypeRef iteratorType = Type::INT32);
void ForEnd();

}  // namespace detail

}  // namespace graphene::codedsl