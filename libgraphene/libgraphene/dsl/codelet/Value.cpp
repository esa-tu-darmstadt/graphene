#include "Value.hpp"

#include <spdlog/spdlog.h>

#include <poplar/CodeletFileType.hpp>
#include <poplar/GraphElements.hpp>

#include "CodeGen.hpp"

namespace graphene::codelet::dsl {

Value Value::getVoid() { return Value(Type::VOID, ""); }

TypeRef Value::type() const { return type_; }

std::string Value::expr() const { return expr_; }

void Value::emitValue() const { CodeGen::emitCode(expr_); }

Value Value::cast(TypeRef type) {
  return Value(type, fmt::format("({}){}", type->str(), expr_), false);
}

Value Value::operator[](Value index) {
  if (!type_->isSubscriptable()) {
    throw std::runtime_error(
        fmt::format("Type {} is not subscriptable", type_->str()));
  }
  return Value(type()->elementType(), expr() + "[" + index.expr() + "]",
               isAssignable_);
}

Value& Value::operator=(const Value& other) {
  if (!isAssignable_) {
    throw std::runtime_error("Value is not assignable");
  }

  if (type() != other.type()) {
    spdlog::debug("Implicit cast from {} to {} during assignment",
                  other.type()->str(), type()->str());
  }
  CodeGen::emitStatement(this->expr() + " = " + other.expr());
  return *this;
}

Value::Value(TypeRef type, std::string expr, bool isAssignable)
    : expr_(std::move(expr)),
      type_(std::move(type)),
      isAssignable_(isAssignable) {}

Variable::Variable(TypeRef type)
    : Value(type,
            CodeGen::emitVariableDeclaration(type,
                                             CodeGen::generateVariableName()),
            true) {}

Variable::Variable(TypeRef type, Value initializer)
    : Value(type,
            CodeGen::emitVariableDeclaration(
                type, CodeGen::generateVariableName(), initializer.expr()),
            true) {}

Variable::Variable(Value initializer)
    : Variable(initializer.type(), initializer) {}

MemberVariable::MemberVariable(TypeRef type)
    : Value(type, CodeGen::generateVariableName(), true) {}

MemberVariable::MemberVariable(TypeRef type, Value initializer)
    : Value(type, CodeGen::generateVariableName(), true),
      initializer_(initializer) {}

void MemberVariable::declare() const {
  if (initializer_) {
    CodeGen::emitVariableDeclaration(type(), expr(), initializer_->expr());
  } else {
    CodeGen::emitVariableDeclaration(type(), expr());
  }
}

Parameter::Parameter(int index, TypeRef type)
    : Value(type, "arg" + std::to_string(index), true) {}

Void::Void() : Value(Type::VOID, "") {}

Expression::Expression(TypeRef type, std::string expr) : Value(type, expr) {}

}  // namespace graphene::codelet::dsl