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

#include "libgraphene/dsl/code/Function.hpp"

#include "libgraphene/dsl/code/Vertex.hpp"

using namespace graphene::codedsl;

/**
 * @brief Constructs and emits a named Function with a given return type,
 * argument types, and code. The code must accept a vector of arguments as its
 * only argument.
 * @tparam F The type of the function code.
 * @param name The name of the function.
 * @param resType The return type of the function.
 * @param argTypes The argument types of the function.
 * @param code The function code.
 */
Function::Function(std::string name, TypeRef resType,
                   std::initializer_list<TypeRef> argTypes, ThreadKind kind,
                   std::function<void(std::vector<Parameter>)> code)
    : Value(VoidType::get(), name),
      resType_(resType),
      name_(name),
      kind_(kind) {
  // Begin function generation
  CodeGen::beginFunction();

  // Emit the target attribute
  CodeGen::emitCode("__attribute__((target(\"");
  CodeGen::emitCode(kind == ThreadKind::Worker ? "worker" : "supervisor");
  CodeGen::emitCode("\"))) ");

  // Emit the return type
  CodeGen::emitType(resType);

  // Emit the function name
  CodeGen::emitCode(" " + this->expr() + "(");

  // Create the parameters
  for (size_t i = 0; i < argTypes.size(); i++) {
    args_.emplace_back(i, *(argTypes.begin() + i));
  }

  // Emit the arguments
  for (size_t i = 0; i < argTypes.size(); i++) {
    CodeGen::emitType(args_[i].type());
    CodeGen::emitCode(" " + args_[i].expr());
    if (i < argTypes.size() - 1) {
      CodeGen::emitCode(", ");
    }
  }

  CodeGen::emitCode(") {\n");

  // Call the user-provided code
  code(args_);

  CodeGen::emitCode("}");

  // End function generation
  code_ = CodeGen::endFunction(this->expr());
  Vertex::addFunctionToCurrentVertex(*this);
}