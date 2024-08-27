#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <sstream>
#include <stack>

#include "libgraphene/common/Type.hpp"
namespace graphene::codedsl {

struct Auto {};

class CodeGen {
  static std::stringstream vertexStream_;
  static std::optional<std::stringstream> functionStream_;

  static std::map<std::string, std::string> functions_;

  static std::ostream &stream() {
    return functionStream_ ? *functionStream_ : vertexStream_;
  }

  static size_t functionCounter_;
  static size_t varCounter_;

 public:
  static void beginFunction() {
    if (functionStream_) throw std::runtime_error("Function already started");
    functionStream_ = std::stringstream();
  }
  static std::string endFunction(std::string name) {
    auto code = functionStream_->str();
    functionStream_ = std::nullopt;
    functions_[name] = code;

    return code;
  }

  static void emitCode(std::string code) { stream() << code; }

  static void emitStatement(std::string code) {
    emitCode(code);
    emitEndStatement();
  }
  static void emitInclude(std::string path, bool system = false) {
    stream() << "#include " << (system ? "<" : "\"") << path
             << (system ? ">\n" : "\"\n");
  }
  static void emitEndStatement() { stream() << ";\n"; }

  static std::string emitVariableDeclaration(TypeRef type, std::string name,
                                             std::string value = "") {
    emitType(type);
    stream() << " " << name;
    if (!value.empty()) stream() << " = " << value;
    emitEndStatement();
    return name;
  }

  static std::stringstream reset() {
    functionCounter_ = 0;
    varCounter_ = 0;

    std::stringstream stream;
    std::swap(vertexStream_, stream);
    return stream;
  }

  static void emitType(TypeRef type) { stream() << type->str(); }

  static const auto &functions() { return functions_; }

  static std::string generateVariableName() {
    return "var" + std::to_string(varCounter_++);
  }
  static std::string generateFunctionName() {
    return "func" + std::to_string(functionCounter_++);
  }
  static std::string generateVertexName() {
    static int counter = 0;
    return "vertex" + std::to_string(counter++);
  }
};
}  // namespace graphene::codedsl