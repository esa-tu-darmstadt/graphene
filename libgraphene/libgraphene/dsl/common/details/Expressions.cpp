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

#include "libgraphene/dsl/common/details/Expressions.hpp"

#include <spdlog/fmt/bundled/core.h>
#include <spdlog/spdlog.h>

#include <numeric>
#include <poplar/Interval.hpp>

namespace graphene::detail {
TypeRef largerType(TypeRef a, TypeRef b) {
  if (a->size() > b->size()) return a;
  if (b->size() > a->size()) return b;
  if (a->isFloat()) return a;
  if (b->isFloat()) return b;
  if (!a->isSigned()) return a;
  return b;
}

TypeRef inferType(BinaryOpType op, TypeRef lhs, TypeRef rhs) {
  auto isIntegral = [](TypeRef t) { return t->isInteger(); };
  auto isFloating = [](TypeRef t) { return t->isFloat(); };

  if (lhs == Type::TWOFLOAT32 || rhs == Type::TWOFLOAT32) {
    if (op != BinaryOpType::ADD && op != BinaryOpType::SUBTRACT &&
        op != BinaryOpType::MULTIPLY && op != BinaryOpType::DIVIDE) {
      throw std::runtime_error("Unsupported operation for two<float>");
    }

    if (lhs == Type::FLOAT32 || rhs == Type::FLOAT32) {
      // two<float> + float = two<float>
      return Type::TWOFLOAT32;
    } else if (lhs == Type::TWOFLOAT32 && rhs == Type::TWOFLOAT32) {
      // two<float> + two<float> = two<float>
      return Type::TWOFLOAT32;
    }
    throw std::runtime_error("Invalid type combination for two<float>");
  }

  switch (op) {
    case BinaryOpType::ADD:
    case BinaryOpType::SUBTRACT:
    case BinaryOpType::MULTIPLY:
    case BinaryOpType::DIVIDE:
    case BinaryOpType::REMAINDER:
      if (isIntegral(lhs) && isIntegral(rhs)) {
        return largerType(lhs, rhs);
      } else if (isIntegral(lhs) && isFloating(rhs)) {
        // int + float = float
        if (lhs->size() > rhs->size())
          spdlog::warn("Narrowing conversion from {} to {}", lhs->str(),
                       rhs->str());
        return rhs;
      } else if (isFloating(lhs) && isIntegral(rhs)) {
        // float + int = float
        if (rhs->size() > lhs->size())
          spdlog::warn("Narrowing conversion from {} to {}", rhs->str(),
                       lhs->str());
        return lhs;
      } else if (isFloating(lhs) && isFloating(rhs)) {
        return largerType(lhs, rhs);
      }
      break;

    case BinaryOpType::BITWISE_AND:
    case BinaryOpType::BITWISE_OR:
    case BinaryOpType::BITWISE_XOR:
    case BinaryOpType::BITWISE_XNOR:
    case BinaryOpType::SHIFT_LEFT:
    case BinaryOpType::SHIFT_RIGHT:
      if (isIntegral(lhs) && isIntegral(rhs)) {
        return largerType(lhs, rhs);
      }
      break;

    case BinaryOpType::EQUAL:
    case BinaryOpType::NOT_EQUAL:
    case BinaryOpType::LESS_THAN:
    case BinaryOpType::LESS_THAN_EQUAL:
    case BinaryOpType::GREATER_THAN:
    case BinaryOpType::GREATER_THAN_EQUAL:
    case BinaryOpType::LOGICAL_AND:
    case BinaryOpType::LOGICAL_OR:
      return Type::BOOL;

    case BinaryOpType::POWER:
    case BinaryOpType::ATAN2:
    case BinaryOpType::MAXIMUM:
    case BinaryOpType::MINIMUM:
      return largerType(lhs, rhs);

    default:
      throw std::runtime_error("Unsupported binary operation type");
  }

  throw std::runtime_error(fmt::format(
      "Invalid type combination for the given operation: {}({}, {})",
      to_string(op), lhs->str(), rhs->str()));
}

TypeRef inferType(UnaryOpType op, TypeRef operand) {
  switch (op) {
    case UnaryOpType::ABSOLUTE:
    case UnaryOpType::NEGATE:
    case UnaryOpType::SQUARE:
      return operand;

    case UnaryOpType::ASIN:
    case UnaryOpType::COS:
    case UnaryOpType::SIN:
    case UnaryOpType::TAN:
    case UnaryOpType::TANH:
    case UnaryOpType::ERF:
    case UnaryOpType::EXPONENT:
    case UnaryOpType::EXPONENT_MINUS_ONE:
    case UnaryOpType::EXPONENT2:
    case UnaryOpType::LOGARITHM:
    case UnaryOpType::LOGARITHM_ONE_PLUS:
    case UnaryOpType::SQRT:
    case UnaryOpType::RSQRT:
    case UnaryOpType::CBRT:
    case UnaryOpType::SIGMOID:
    case UnaryOpType::GELU_ERF:
      return largerType(Type::FLOAT32, operand);

    case UnaryOpType::CEIL:
    case UnaryOpType::FLOOR:
    case UnaryOpType::ROUND:
    case UnaryOpType::TRUNC:
    case UnaryOpType::NEARBY_INT:
      return operand;

    case UnaryOpType::BITWISE_NOT:
    case UnaryOpType::COUNT_LEADING_ZEROS:
    case UnaryOpType::POPCOUNT:
      if (operand->isInteger()) {
        return operand;
      }
      break;

    case UnaryOpType::LOGICAL_NOT:
    case UnaryOpType::IS_FINITE:
    case UnaryOpType::IS_INF:
    case UnaryOpType::IS_NAN:
      return Type::BOOL;

    case UnaryOpType::INVERSE:
      return largerType(Type::FLOAT32, operand);

    case UnaryOpType::RELU:
    case UnaryOpType::SIGNUM:
      return operand;

    default:
      throw std::runtime_error("Unsupported unary operation type");
  }

  throw std::runtime_error(
      fmt::format("Invalid type combination for the given operation: {}({})",
                  to_string(op), operand->str()));
}

TypeRef inferType(TernaryOpType op, TypeRef a, TypeRef b, TypeRef c) {
  switch (op) {
    case TernaryOpType::CLAMP:
      return largerType(largerType(a, b), c);

    case TernaryOpType::SELECT:
      if (a == Type::BOOL) {
        return largerType(b, c);
      }
      break;

    default:
      throw std::runtime_error("Unsupported ternary operation type");
  }

  throw std::runtime_error(fmt::format(
      "Invalid type combination for the given operation: {}({}, {}, {})",
      to_string(op), a->str(), b->str(), c->str()));
}

std::string_view to_string(BinaryOpType op) {
  static const std::unordered_map<BinaryOpType, std::string_view> opStrings = {
      {BinaryOpType::ADD, "add"},
      {BinaryOpType::SUBTRACT, "subtract"},
      {BinaryOpType::MULTIPLY, "multiply"},
      {BinaryOpType::DIVIDE, "divide"},
      {BinaryOpType::REMAINDER, "remainder"},
      {BinaryOpType::BITWISE_AND, "bitwise_and"},
      {BinaryOpType::BITWISE_OR, "bitwise_or"},
      {BinaryOpType::BITWISE_XOR, "bitwise_xor"},
      {BinaryOpType::BITWISE_XNOR, "bitwise_xnor"},
      {BinaryOpType::SHIFT_LEFT, "shift_left"},
      {BinaryOpType::SHIFT_RIGHT, "shift_right"},
      {BinaryOpType::EQUAL, "equal"},
      {BinaryOpType::NOT_EQUAL, "not_equal"},
      {BinaryOpType::LESS_THAN, "less_than"},
      {BinaryOpType::LESS_THAN_EQUAL, "less_than_equal"},
      {BinaryOpType::GREATER_THAN, "greater_than"},
      {BinaryOpType::GREATER_THAN_EQUAL, "greater_than_equal"},
      {BinaryOpType::LOGICAL_AND, "logical_and"},
      {BinaryOpType::LOGICAL_OR, "logical_or"},
      {BinaryOpType::POWER, "power"},
      {BinaryOpType::ATAN2, "atan2"},
      {BinaryOpType::MAXIMUM, "maximum"},
      {BinaryOpType::MINIMUM, "minimum"}};

  auto it = opStrings.find(op);
  if (it != opStrings.end()) {
    return it->second;
  }
  throw std::runtime_error("Unsupported binary operation type");
}

std::string_view to_string(UnaryOpType op) {
  static const std::unordered_map<UnaryOpType, std::string_view> opStrings = {
      {UnaryOpType::ABSOLUTE, "absolute"},
      {UnaryOpType::NEGATE, "negate"},
      {UnaryOpType::SQUARE, "square"},
      {UnaryOpType::ASIN, "asin"},
      {UnaryOpType::COS, "cos"},
      {UnaryOpType::SIN, "sin"},
      {UnaryOpType::TAN, "tan"},
      {UnaryOpType::TANH, "tanh"},
      {UnaryOpType::ERF, "erf"},
      {UnaryOpType::EXPONENT, "exponent"},
      {UnaryOpType::EXPONENT_MINUS_ONE, "exponent_minus_one"},
      {UnaryOpType::EXPONENT2, "exponent2"},
      {UnaryOpType::LOGARITHM, "logarithm"},
      {UnaryOpType::LOGARITHM_ONE_PLUS, "logarithm_one_plus"},
      {UnaryOpType::SQRT, "sqrt"},
      {UnaryOpType::RSQRT, "rsqrt"},
      {UnaryOpType::CBRT, "cbrt"},
      {UnaryOpType::SIGMOID, "sigmoid"},
      {UnaryOpType::CEIL, "ceil"},
      {UnaryOpType::FLOOR, "floor"},
      {UnaryOpType::ROUND, "round"},
      {UnaryOpType::TRUNC, "trunc"},
      {UnaryOpType::NEARBY_INT, "nearby_int"},
      {UnaryOpType::BITWISE_NOT, "bitwise_not"},
      {UnaryOpType::COUNT_LEADING_ZEROS, "count_leading_zeros"},
      {UnaryOpType::POPCOUNT, "popcount"},
      {UnaryOpType::LOGICAL_NOT, "logical_not"},
      {UnaryOpType::IS_FINITE, "is_finite"},
      {UnaryOpType::IS_INF, "is_inf"},
      {UnaryOpType::IS_NAN, "is_nan"},
      {UnaryOpType::INVERSE, "inverse"},
      {UnaryOpType::RELU, "relu"},
      {UnaryOpType::SIGNUM, "signum"},
      {UnaryOpType::GELU_ERF, "gelu_erf"}};

  auto it = opStrings.find(op);
  if (it != opStrings.end()) {
    return it->second;
  }
  throw std::runtime_error("Unsupported unary operation type");
}

std::string_view to_string(TernaryOpType op) {
  static const std::unordered_map<TernaryOpType, std::string_view> opStrings = {
      {TernaryOpType::CLAMP, "clamp"}, {TernaryOpType::SELECT, "select"}};

  auto it = opStrings.find(op);
  if (it != opStrings.end()) {
    return it->second;
  }
  throw std::runtime_error("Unsupported ternary operation type");
}

}  // namespace graphene::detail