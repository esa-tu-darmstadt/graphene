#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <sstream>
#include <stack>
#include <unordered_set>

#include "libgraphene/common/Type.hpp"
namespace graphene::codedsl {

struct Auto {};

class CodeGen {
  static std::stringstream vertexStream_;
  static std::stack<std::stringstream> functionStreams_;

  static std::ostream &stream() {
    return functionStreams_.empty() ? vertexStream_ : functionStreams_.top();
  }

  static size_t functionCounter_;
  static size_t varCounter_;

 public:
  static void beginFunction() { functionStreams_.push(std::stringstream()); }
  static std::string endFunction(std::string name) {
    if (functionStreams_.empty()) {
      throw std::runtime_error("No function to end");
    }
    std::string code = functionStreams_.top().str();
    functionStreams_.pop();

    return code;
  }

  static void emitCode(std::string code) { stream() << code; }

  static void emitSingleLineComment(std::string comment) {
    stream() << "// " << comment << "\n";
  }

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
                                             CTypeQualifiers qualifiers,
                                             std::string value = "") {
    if (qualifiers.Volatile) stream() << "volatile ";
    if (qualifiers.Const) stream() << "const ";
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

  /// Vertices are generated with a common placeholder, so that the name does
  /// not influence its hash, which is used to cache the vertex. Before a vertex
  /// is compiled, the placeholder is replaced with a unique name.
  static void replaceVertexNamePlaceholder(std::string &code,
                                           std::string placeholder,
                                           std::string actualName) {
    if (placeholder.size() != actualName.size()) {
      throw std::runtime_error(
          "Placeholder and actual name must have the same size");
    }
    while (code.find(placeholder) != std::string::npos) {
      size_t pos = code.find(placeholder);
      code.replace(pos, placeholder.size(), actualName);
    }
  }
};
}  // namespace graphene::codedsl