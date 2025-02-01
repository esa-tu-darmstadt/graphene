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

#include "Value.hpp"

#include <spdlog/spdlog.h>

#include <poplar/CodeletFileType.hpp>
#include <poplar/GraphElements.hpp>

#include "CodeGen.hpp"
#include "libgraphene/common/Type.hpp"

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

Variable::Variable(TypeRef type, CTypeQualifiers qualifiers)
    : Value(type,
            CodeGen::emitVariableDeclaration(
                type, CodeGen::generateVariableName(), qualifiers),
            true) {
  if (qualifiers.Const) {
    throw std::runtime_error(
        "Variable with const qualifier must have an initializer");
  }
}

Variable::Variable(TypeRef type, Value initializer, CTypeQualifiers qualifiers)
    : Value(type,
            CodeGen::emitVariableDeclaration(type,
                                             CodeGen::generateVariableName(),
                                             qualifiers, initializer.expr()),
            !qualifiers.Const) {}

Variable::Variable(Value initializer, CTypeQualifiers qualifiers)
    : Variable(initializer.type(), initializer, qualifiers) {}

MemberVariable::MemberVariable(TypeRef type, CTypeQualifiers qualifiers)
    : Value(type, CodeGen::generateVariableName(), true),
      qualifiers_(qualifiers) {}

MemberVariable::MemberVariable(TypeRef type, Value initializer,
                               CTypeQualifiers qualifiers)
    : Value(type, CodeGen::generateVariableName(), true),
      initializer_(initializer),
      qualifiers_(qualifiers) {}

void MemberVariable::declare() const {
  if (initializer_) {
    CodeGen::emitVariableDeclaration(type(), expr(), qualifiers_,
                                     initializer_->expr());
  } else {
    CodeGen::emitVariableDeclaration(type(), expr(), qualifiers_);
  }
}

Parameter::Parameter(int index, TypeRef type)
    : Value(type, "arg" + std::to_string(index), true) {}

Void::Void() : Value(Type::VOID, "") {}

Expression::Expression(TypeRef type, std::string expr) : Value(type, expr) {}

}  // namespace graphene::codedsl