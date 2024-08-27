#pragma once

#include <spdlog/spdlog.h>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/dsl/code/CodeGen.hpp"
#include "libgraphene/dsl/code/Value.hpp"
namespace graphene::codedsl {
/**
 * @brief Represents a function in the CodeDSL language.
 */
class Function : public Value {
 public:
  /**
   * @brief Constructs and emits a Function with a given return type, argument
   * types, and code.
   * @tparam F The type of the function code.
   * @param resType The return type of the function.
   * @param argTypes The argument types of the function.
   * @param code The function code.
   */
  template <typename F>
  Function(TypeRef resType, std::initializer_list<TypeRef> argTypes, F code);

  /**
   * @brief Constructs and emits a named Function with a given return type,
   * argument types, and code.
   * @tparam F The type of the function code.
   * @param name The name of the function.
   * @param resType The return type of the function.
   * @param argTypes The argument types of the function.
   * @param code The function code.
   */
  template <typename F>
  Function(std::string name, TypeRef resType,
           std::initializer_list<TypeRef> argTypes, F code)
      : Value(VoidType::get(), name), resType_(resType) {
    constexpr size_t numArgs = detail::function_traits<F>::arity;
    if (numArgs != argTypes.size()) {
      throw std::runtime_error("Number of arguments does not match");
    }

    // Begin function generation
    CodeGen::beginFunction();

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
    detail::callFunctionWithUnpackedArgs<void>(code, args_);

    CodeGen::emitCode("}");

    // End function generation
    std::string content = CodeGen::endFunction(this->expr());
    spdlog::info("Generated function: {}", content);
  }

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

  template <typename... Args>
  using FunctionGenerator = std::function<Function(Args...)>;

 private:
  TypeRef resType_;
  std::vector<Parameter> args_;
};
}  // namespace graphene::codedsl