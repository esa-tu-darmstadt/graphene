#include "Operators.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/dsl/code/CodeGen.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/dsl/common/details/Expressions.hpp"

namespace graphene::codedsl {
Value getTileID() {
  return Expression(PtrType::get(Type::VOID), "__builtin_ipu_get_tile_id()")
      .cast(Type::INT32);
}

Expression operation(detail::BinaryOpType opType, Value lhs, Value rhs) {
  TypeRef resultType = detail::inferType(opType, lhs.type(), rhs.type());
  std::string expr;
  switch (opType) {
    default:
      throw std::runtime_error("Unsupported operation. Feel free to add it.");
    case detail::BinaryOpType::ADD:
      return Expression(resultType,
                        "(" + lhs.expr() + " + " + rhs.expr() + ")");
    case detail::BinaryOpType::SUBTRACT:
      return Expression(resultType,
                        "(" + lhs.expr() + " - " + rhs.expr() + ")");
    case detail::BinaryOpType::MULTIPLY:
      return Expression(resultType,
                        "(" + lhs.expr() + " * " + rhs.expr() + ")");
    case detail::BinaryOpType::DIVIDE:
      return Expression(resultType,
                        "(" + lhs.expr() + " / " + rhs.expr() + ")");
    case detail::BinaryOpType::REMAINDER:
      return Expression(resultType,
                        "(" + lhs.expr() + " % " + rhs.expr() + ")");
    case detail::BinaryOpType::EQUAL:
      return Expression(resultType,
                        "(" + lhs.expr() + " == " + rhs.expr() + ")");
    case detail::BinaryOpType::NOT_EQUAL:
      return Expression(resultType,
                        "(" + lhs.expr() + " != " + rhs.expr() + ")");
    case detail::BinaryOpType::LESS_THAN:
      return Expression(resultType,
                        "(" + lhs.expr() + " < " + rhs.expr() + ")");
    case detail::BinaryOpType::LESS_THAN_EQUAL:
      return Expression(resultType,
                        "(" + lhs.expr() + " <= " + rhs.expr() + ")");
    case detail::BinaryOpType::GREATER_THAN:
      return Expression(resultType,
                        "(" + lhs.expr() + " > " + rhs.expr() + ")");
    case detail::BinaryOpType::GREATER_THAN_EQUAL:
      return Expression(resultType,
                        "(" + lhs.expr() + " >= " + rhs.expr() + ")");
    case detail::BinaryOpType::LOGICAL_AND:
      return Expression(resultType,
                        "(" + lhs.expr() + " && " + rhs.expr() + ")");
    case detail::BinaryOpType::LOGICAL_OR:
      return Expression(resultType,
                        "(" + lhs.expr() + " || " + rhs.expr() + ")");
    case detail::BinaryOpType::BITWISE_AND:
      return Expression(resultType,
                        "(" + lhs.expr() + " & " + rhs.expr() + ")");
    case detail::BinaryOpType::BITWISE_OR:
      return Expression(resultType,
                        "(" + lhs.expr() + " | " + rhs.expr() + ")");
    case detail::BinaryOpType::BITWISE_XOR:
      return Expression(resultType,
                        "(" + lhs.expr() + " ^ " + rhs.expr() + ")");
    case detail::BinaryOpType::BITWISE_XNOR:
      return Expression(resultType,
                        "(~(" + lhs.expr() + " ^ " + rhs.expr() + "))");
    case detail::BinaryOpType::SHIFT_LEFT:
      return Expression(resultType,
                        "(" + lhs.expr() + " << " + rhs.expr() + ")");
    case detail::BinaryOpType::SHIFT_RIGHT:
      return Expression(resultType,
                        "(" + lhs.expr() + " >> " + rhs.expr() + ")");
    case detail::BinaryOpType::MAXIMUM:
      return Expression(resultType,
                        "std::max(" + lhs.expr() + ", " + rhs.expr() + ")");
    case detail::BinaryOpType::MINIMUM:
      return Expression(resultType,
                        "std::min(" + lhs.expr() + ", " + rhs.expr() + ")");
    case detail::BinaryOpType::POWER:
      return Expression(resultType,
                        "std::pow(" + lhs.expr() + ", " + rhs.expr() + ")");
  }
}

Expression operation(detail::UnaryOpType opType, Value arg) {
  TypeRef resultType = detail::inferType(opType, arg.type());
  std::string expr;
  switch (opType) {
    default:
      throw std::runtime_error("Unsupported operation. Feel free to add it.");
    case detail::UnaryOpType::ABSOLUTE:
      return Expression(resultType, "std::abs(" + arg.expr() + ")");
    case detail::UnaryOpType::ASIN:
      return Expression(resultType, "std::asin(" + arg.expr() + ")");
    case detail::UnaryOpType::BITWISE_NOT:
      return Expression(resultType, "(~" + arg.expr() + ")");
    case detail::UnaryOpType::CBRT:
      return Expression(resultType, "std::cbrt(" + arg.expr() + ")");
    case detail::UnaryOpType::CEIL:
      return Expression(resultType, "std::ceil(" + arg.expr() + ")");
    case detail::UnaryOpType::COS:
      return Expression(resultType, "std::cos(" + arg.expr() + ")");
    case detail::UnaryOpType::COUNT_LEADING_ZEROS:
      return Expression(resultType, "__builtin_clz(" + arg.expr() + ")");
    case detail::UnaryOpType::ERF:
      return Expression(resultType, "std::erf(" + arg.expr() + ")");
    case detail::UnaryOpType::EXPONENT:
      return Expression(resultType, "std::exp(" + arg.expr() + ")");
    case detail::UnaryOpType::EXPONENT_MINUS_ONE:
      return Expression(resultType, "std::expm1(" + arg.expr() + ")");
    case detail::UnaryOpType::EXPONENT2:
      return Expression(resultType, "std::exp2(" + arg.expr() + ")");
    case detail::UnaryOpType::FLOOR:
      return Expression(resultType, "std::floor(" + arg.expr() + ")");
    case detail::UnaryOpType::LOGICAL_NOT:
      return Expression(resultType, "!" + arg.expr());
    case detail::UnaryOpType::SQRT:
      return Expression(resultType, "std::sqrt(" + arg.expr() + ")");
    case detail::UnaryOpType::SQUARE:
      return Expression(resultType,
                        "(" + arg.expr() + " * " + arg.expr() + ")");
  }
}

void Assume(Value expr) {
  CodeGen::emitStatement("__builtin_assume(" + expr.expr() + ")");
}

Value getNumWorkerThreadsPerTile() { return (uint8_t)6; }

Value getMaxLoopRptCount() { return (uint16_t)4095; }

void AssumeHardwareLoop(Value iterator) {
  Assume(iterator <= getMaxLoopRptCount());
}

void syncAndStartOnAllWorkers(const Function &func) {
  if (func.threadKind() != ThreadKind::Worker)
    throw std::runtime_error("Function must be a worker function");
  CodeGen::emitStatement(
      "::ipu::syncAndStartOnAllWorkers<ConcreteVertexType, "
      "&ConcreteVertexType::" +
      func.name() + ">(this)");
}

void syncAllWorkers() { CodeGen::emitStatement("::ipu::syncAllWorkers()"); }

}  // namespace graphene::codedsl