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

#include <spdlog/spdlog.h>

#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/common/details/Expressions.hpp"

namespace graphene::codedsl {

////////////////////////////////////////////////////////////////////////////////
// Binary operators
////////////////////////////////////////////////////////////////////////////////

/// Performs the given binary operation on the two values.
Expression operation(::graphene::detail::BinaryOpType opType, Value lhs,
                     Value rhs);

#define BINARY_OP(name, opType, symbol)                                   \
  template <typename lhs_t, typename rhs_t>                               \
  inline Expression name(lhs_t lhs, rhs_t rhs)                            \
    requires(std::derived_from<lhs_t, Value> ||                           \
             std::derived_from<rhs_t, Value>)                             \
  {                                                                       \
    return operation(::graphene::detail::BinaryOpType::opType, lhs, rhs); \
  }

#define BINARY_OP_AND_SYMBOL(name, opType, symbol)        \
  BINARY_OP(name, opType, symbol)                         \
  template <typename lhs_t, typename rhs_t>               \
  inline Expression operator symbol(lhs_t lhs, rhs_t rhs) \
    requires(std::derived_from<lhs_t, Value> ||           \
             std::derived_from<rhs_t, Value>)             \
  {                                                       \
    return name(lhs, rhs);                                \
  }

#define BINARY_OP_INPLACE(name, opType, symbol)            \
  template <typename rhs_t>                                \
  inline void name##Inplace(Value &lhs, rhs_t rhs) {       \
    Value rhsValue = rhs;                                  \
    lhs = name(lhs, rhsValue);                             \
  }                                                        \
  template <typename rhs_t>                                \
  inline Value operator symbol##=(Value &lhs, rhs_t rhs) { \
    name##Inplace(lhs, rhs);                               \
    return lhs;                                            \
  }

#define BINARY_OP_AND_SYMBOL_AND_INPLACE(name, opType, symbol) \
  BINARY_OP_AND_SYMBOL(name, opType, symbol)                   \
  BINARY_OP_INPLACE(name, opType, symbol)

#include "libgraphene/dsl/code/details/Operators.hpp.inc"

////////////////////////////////////////////////////////////////////////////////
// Unary operators
////////////////////////////////////////////////////////////////////////////////

/// Performs the given unary operation on the value.
Expression operation(::graphene::detail::UnaryOpType opType, Value arg);

#define UNARY_OP(name, opType)                                      \
  inline Expression name(Value arg) {                               \
    return operation(::graphene::detail::UnaryOpType::opType, arg); \
  }

#include "libgraphene/dsl/code/details/Operators.hpp.inc"

template <typename true_t, typename false_t>
Expression Select(Value condition, true_t trueArg, false_t falseArg) {
  Value trueValue = trueArg;
  Value falseValue = falseArg;

  if (condition.type() != Type::BOOL) {
    throw std::runtime_error("Condition must be of type bool");
  }

  if (trueValue.type() != falseValue.type()) {
    throw std::runtime_error(
        "True and false values must have the same type as implicit type "
        "conversion is not yet supported");
  }

  auto resultType = trueValue.type();
  return Expression(resultType,
                    fmt::format("({} ? {} : {})", condition.expr(),
                                trueValue.expr(), falseValue.expr()));
}

/// Returns the ID of the current tile.
Value getTileID();

/// Returns the number of worker threads per tile.
Value getNumWorkerThreadsPerTile();

/// Returns the maximum number of loop repetitions that are supported by the \c
/// rpt instruction. See `TREG_REPEAT_COUNT_WIDTH` in the Tile Vertex ISA.
Value getMaxLoopRptCount();

/// Assume that the given expression is true. This can be used to provide
/// additional information to the compiler.
void Assume(Value expr);

/// Assumes that the given iterator is small enough so that hardware loop
/// instructions can be used. This is equivalent to calling `Assume` with the
/// expression `iterator <= getMaxLoopRptCount()`.
void AssumeHardwareLoop(Value iterator);

class Function;
void syncAndStartOnAllWorkers(const Function &func);

void syncAllWorkers();

}  // namespace graphene::codedsl