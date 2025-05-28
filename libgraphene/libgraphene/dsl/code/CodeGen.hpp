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
  static void emitInclude(std::string path, bool system = false,
                          bool ipuOnly = false) {
    if (ipuOnly) stream() << "#ifdef __IPU__\n";
    stream() << "#include " << (system ? "<" : "\"") << path
             << (system ? ">\n" : "\"\n");
    if (ipuOnly) stream() << "#endif // __IPU__\n";
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