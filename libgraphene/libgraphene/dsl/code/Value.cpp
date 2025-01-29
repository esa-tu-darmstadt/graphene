#include "Value.hpp"

#include <spdlog/spdlog.h>

#include <poplar/CodeletFileType.hpp>
#include <poplar/GraphElements.hpp>

#include "CodeGen.hpp"

namespace graphene::codedsl {

Value Value::getVoid() { return Value(Type::VOID, ""); }

TypeRef Value::type() const { return type_; }

std::string Value::expr() const { return expr_; }

void Value::emitValue() const { CodeGen::emitCode(expr_); }

Value Value::cast(TypeRef type) {
  return Value(type, fmt::format("(({}){})", type->str(), expr_), false);
}

Value Value::reinterpretCast(TypeRef type) {
  return Value(type,
               fmt::format("(*reinterpret_cast<{}>(&{}))",
                           PtrType::get(type)->str(), expr_),
               false);
}

Value Value::operator[](Value index) {
  if (!type_->isSubscriptable()) {
    throw std::runtime_error(
        fmt::format("Type {} is not subscriptable", type_->str()));
  }
  // TODO: Determine whether the result is assignable
  return Value(type()->elementType(), expr() + "[" + index.expr() + "]", true);
}

Value Value::size() const {
  if (!type_->hasFunction("size")) {
    throw std::runtime_error(
        fmt::format("Type {} has no size function", type_->str()));
  }
  return Value(type_->functionReturnType("size"), expr() + ".size()");
}

void Value::assign(const Value& other, bool endWithSemicolon) {
  if (!isAssignable_) {
    throw std::runtime_error("Value is not assignable");
  }

  if (type() != other.type()) {
    spdlog::debug("Implicit cast from {} to {} during assignment",
                  other.type()->str(), type()->str());
  }
  CodeGen::emitCode(this->expr() + " = " + other.expr());
  if (endWithSemicolon) {
    CodeGen::emitEndStatement();
  }
}

Value::Value(TypeRef type, std::string expr, bool isAssignable)
    : expr_(std::move(expr)),
      type_(std::move(type)),
      isAssignable_(isAssignable) {}

Variable::Variable(TypeRef type)
    : Value(type,
            CodeGen::emitVariableDeclaration(
                type, CodeGen::generateVariableName(), /*isConst=*/false),
            true) {}

Variable::Variable(TypeRef type, Value initializer, bool isConst)
    : Value(type,
            CodeGen::emitVariableDeclaration(type,
                                             CodeGen::generateVariableName(),
                                             isConst, initializer.expr()),
            !isConst) {}

Variable::Variable(Value initializer, bool isConst)
    : Variable(initializer.type(), initializer, isConst) {}

MemberVariable::MemberVariable(TypeRef type)
    : Value(type, CodeGen::generateVariableName(), true) {}

MemberVariable::MemberVariable(TypeRef type, Value initializer)
    : Value(type, CodeGen::generateVariableName(), true),
      initializer_(initializer) {}

void MemberVariable::declare() const {
  if (initializer_) {
    CodeGen::emitVariableDeclaration(type(), expr(),
                                     /*isConst=*/false, initializer_->expr());
  } else {
    CodeGen::emitVariableDeclaration(type(), expr(), /*isConst=*/false);
  }
}

Parameter::Parameter(int index, TypeRef type)
    : Value(type, "arg" + std::to_string(index), true) {}

Void::Void() : Value(Type::VOID, "") {}

Expression::Expression(TypeRef type, std::string expr) : Value(type, expr) {}

}  // namespace graphene::codedsl