#include "Operators.hpp"

#include <spdlog/spdlog.h>

namespace graphene::codedsl {
Value getTileID() {
  return Expression(PtrType::get(Type::VOID), "__builtin_ipu_get_tile_id()")
      .cast(Type::INT32);
}
namespace detail {
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

  switch (op) {
    case BinaryOpType::ADD:
    case BinaryOpType::SUBTRACT:
    case BinaryOpType::MULTIPLY:
    case BinaryOpType::DIVIDE:
    case BinaryOpType::REMAINDER:
      if (isIntegral(lhs) && isIntegral(rhs)) {
        return largerType(lhs, rhs);
      } else if (isFloating(lhs) || isFloating(rhs)) {
        return largerType(lhs, rhs);
      }
      break;

    case BinaryOpType::BITWISE_AND:
    case BinaryOpType::BITWISE_OR:
    case BinaryOpType::BITWISE_XOR:
    case BinaryOpType::BITWISE_XNOR:
    case BinaryOpType::SHIFT_LEFT:
    case BinaryOpType::SHIFT_RIGHT:
    case BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND:
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

    case BinaryOpType::INV_STD_DEV_TO_VARIANCE:
    case BinaryOpType::VARIANCE_TO_INV_STD_DEV:
      return largerType(Type::FLOAT32, largerType(lhs, rhs));

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
      {BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, "shift_right_sign_extend"},
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
      {BinaryOpType::MINIMUM, "minimum"},
      {BinaryOpType::INV_STD_DEV_TO_VARIANCE, "inv_std_dev_to_variance"},
      {BinaryOpType::VARIANCE_TO_INV_STD_DEV, "variance_to_inv_std_dev"}};

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

}  // namespace detail
}  // namespace graphene::codedsl