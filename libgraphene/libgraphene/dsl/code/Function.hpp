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

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/dsl/code/CodeGen.hpp"
#include "libgraphene/dsl/code/Value.hpp"
namespace graphene::codedsl {

enum class ThreadKind { Worker, Supervisor };

/**
 * @brief Represents a function in the CodeDSL language.
 */
class Function : public Value {
 public:
  /**
   * @brief Constructs and emits a named Function with a given return type,
   * argument types, and code. The code must accept one argument for each
   * function argument.
   * @tparam F The type of the function code.
   * @param name The name of the function.
   * @param resType The return type of the function.
   * @param argTypes The argument types of the function.
   * @param code The function code.
   */
  template <typename F>
  requires ::graphene::detail::invocable_with_args_of<F, Parameter> Function(
      std::string name, TypeRef resType,
      std::initializer_list<TypeRef> argTypes, ThreadKind kind, F code)
      : Function(name, resType, argTypes, kind,
                 [&code](std::vector<Parameter> args) {
                   ::graphene::detail::callFunctionWithUnpackedArgs<void>(code,
                                                                          args);
                 }) {}

  /**
   * @brief Constructs and emits a named Function with a given return type,
   *  no arguments, and code. The code must accept no arguments.
   * @tparam F The type of the function code.
   * @param name The name of the function.
   * @param resType The return type of the function.
   * @param code The function code.
   */
  Function(std::string name, TypeRef resType, ThreadKind kind,
           std::function<void()> code)
      : Function(name, resType, {}, kind,
                 [&code](std::vector<Parameter> args) { code(); }) {}

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
  Function(std::string name, TypeRef resType,
           std::initializer_list<TypeRef> argTypes, ThreadKind kind,
           std::function<void(std::vector<Parameter>)> code);

  /**
   * @brief Calls the function with given arguments.
   * @tparam args_t The types of the arguments.
   * @param args The argument values.
   * @return The result of the function call as a Value.
   */
  template <typename... args_t>
  Value operator()(args_t... args) {
    std::string call = this->expr() + "(";

    ((call += args.expr() + ", "), ...);
    if (sizeof...(args) > 0) {
      call.pop_back();
      call.pop_back();
    }
    call += ")";
    if (resType_->isVoid()) {
      CodeGen::emitStatement(call);
      return Void();
    } else {
      return Variable(Expression(resType_, call));
    }
  }

  TypeRef returnType() const { return resType_; }
  const std::vector<Parameter>& args() const { return args_; }
  const std::string& name() const { return name_; }
  ThreadKind threadKind() const { return kind_; }
  const std::string& code() const { return code_; }

  template <typename... Args>
  using FunctionGenerator = std::function<Function(Args...)>;

 private:
  TypeRef resType_;
  std::vector<Parameter> args_;
  std::string name_;
  ThreadKind kind_;
  std::string code_;
};
}  // namespace graphene::codedsl