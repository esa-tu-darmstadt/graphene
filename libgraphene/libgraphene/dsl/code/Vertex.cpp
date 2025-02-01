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

#include "libgraphene/dsl/code/Vertex.hpp"

#include "libgraphene/dsl/code/CodeGen.hpp"

using namespace graphene::codedsl;

Vertex* Vertex::currentVertex = nullptr;

Vertex::Vertex(
    std::string name, std::vector<MemberVarInfo> memberVars, VertexKind kind,
    std::function<Function(std::vector<Value>)> computeFunctionGenerator) {
  if (currentVertex) {
    throw std::runtime_error("Cannot nest vertices");
  }
  currentVertex = this;
  // Create the fields
  std::vector<Value> convertedFields_;
  for (size_t i = 0; i < memberVars.size(); i++) {
    if (memberVars[i].isTensorMemberVar()) {
      const auto& tensorInfo = memberVars[i].tensorMemberVar();
      TypeRef elementType = tensorInfo.elementType;
      if (!elementType->poplarEquivalentType()) {
        throw std::runtime_error(
            "Element type must be representable in Poplar");
      }
      if (!elementType->poplarEquivalentType()) {
        throw std::runtime_error(
            "Element type must be representable in Poplar");
      }
      auto* vectorType =
          VertexVectorType::get(elementType->poplarEquivalentType());
      auto* fieldType = VertexInOutType::get(tensorInfo.direction, vectorType);

      fields_.emplace_back(fieldType);

      // Tensors of types unsupported by Poplar are converted to pointers of the
      // correct type
      if (elementType != elementType->poplarEquivalentType()) {
        // Cast the vector type to the actual element type
        auto* convertedVectorType = VertexVectorType::get(elementType);
        auto* convertedFieldType =
            VertexInOutType::get(tensorInfo.direction, convertedVectorType);
        convertedFields_.push_back(
            fields_[i].reinterpretCast(convertedFieldType));
      } else {
        convertedFields_.push_back(fields_[i]);
      }
    } else if (memberVars[i].isUnconnectedMemberVar()) {
      const auto& unconnectedInfo = memberVars[i].unconnectedMemberVar();
      fields_.emplace_back(unconnectedInfo.type, unconnectedInfo.qualifiers);
      convertedFields_.push_back(fields_[i]);
    } else {
      throw std::runtime_error("Unknown member variable type");
    }
  }

  // Generate the functions
  Function computeFunc = computeFunctionGenerator(convertedFields_);

  // Check if the compute function is valid
  if (computeFunc.returnType() != BoolType::get()) {
    throw std::runtime_error("Compute function must return a boolean");
  }
  std::string vertexClass;
  switch (kind) {
    case VertexKind::Vertex:
      if (computeFunc.threadKind() != ThreadKind::Worker)
        throw std::runtime_error(
            "Vertex compute function must be a worker function");

      if (computeFunc.args().size() != 0)
        throw std::runtime_error(
            "Vertex compute function must not have arguments");
      vertexClass = "Vertex";
      break;
    case VertexKind::MultiVertex:
      if (computeFunc.threadKind() != ThreadKind::Worker)
        throw std::runtime_error(
            "MultiVertex compute function must be a worker function");
      if (computeFunc.args().size() != 1) {
        throw std::runtime_error(
            "MultiVertex compute function must have exactly one argument");
      }
      vertexClass = "MultiVertex";
      break;
    case VertexKind::SupervisorVertex:
      if (computeFunc.threadKind() != ThreadKind::Supervisor)
        throw std::runtime_error(
            "SupervisorVertex compute function must be a supervisor "
            "function");
      if (computeFunc.args().size() != 0) {
        throw std::runtime_error(
            "SupervisorVertex compute function must not have arguments");
      }
      vertexClass = "SupervisorVertex";
      break;
  }

  // Begin vertex generation
  CodeGen::emitCode("class " + name + " : public ::poplar::" + vertexClass +
                    " {\n");
  CodeGen::emitCode("public:\n");
  CodeGen::emitStatement("using ConcreteVertexType = " + name + ";");

  // Emit the member variables
  for (const auto& field : fields_) {
    field.declare();
  }

  for (const Function& function : functions_) {
    CodeGen::emitCode(function.code());
    CodeGen::emitCode("\n");
  }

  CodeGen::emitCode("};");
}

void Vertex::addFunctionToCurrentVertex(Function function) {
  if (!currentVertex) {
    throw std::runtime_error("No current vertex");
  }
  currentVertex->functions_.emplace_back(std::move(function));
}