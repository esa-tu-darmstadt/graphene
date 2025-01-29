#pragma once
#include <vector>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/CodeGen.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
namespace graphene::codedsl {
/**
 * @brief Represents a vertex in the CodeDSL language.
 */
class Vertex {
 public:
  /**
   * @brief Constructs and emits a Vertex with a given name, types, directions,
   * and compute function generator.
   * @param name The name of the vertex.
   * @param types The types of the tensors. Must be native Poplar types.
   * @param directions The directions of the vertex fields (Input, Output,
   * InOut).
   * @param computeFunctionGenerator The compute function generator. Must accept
   * a vector of \ref MemberVariable and return the compute function. The
   * generated function must be named "compute", accept a single argument (the
   * worker id) or no argument, and return a boolean.
   */
  Vertex(std::string name, std::vector<TypeRef> elementTypes,
         std::vector<VertexInOutType::Direction> directions,
         std::function<Function(std::vector<Value>)> computeFunctionGenerator) {
    // Create the fields
    std::vector<Value> convertedFields_;
    for (size_t i = 0; i < elementTypes.size(); i++) {
      auto* elementType = elementTypes[i];

      if (!elementType->poplarEquivalentType()) {
        throw std::runtime_error(
            "Element type must be representable in Poplar");
      }

      auto* vectorType =
          VertexVectorType::get(elementType->poplarEquivalentType());
      auto* fieldType = VertexInOutType::get(directions[i], vectorType);

      spdlog::trace("Adding field of type {} with native element type {}",
                    fieldType->str(),
                    elementType->poplarEquivalentType()->str());
      fields_.emplace_back(fieldType);

      // Tensors of types unsupported by Poplar are converted to pointers of the
      // correct type
      if (elementType != elementType->poplarEquivalentType()) {
        // Cast the vector type to the actual element type
        auto* convertedVectorType = VertexVectorType::get(elementType);
        auto* convertedFieldType =
            VertexInOutType::get(directions[i], convertedVectorType);
        convertedFields_.push_back(
            fields_[i].reinterpretCast(convertedFieldType));
      } else {
        convertedFields_.push_back(fields_[i]);
      }
    }

    // Generate the compute function
    Function userFunc = computeFunctionGenerator(convertedFields_);
    if (userFunc.expr() != "compute") {
      throw std::runtime_error("Compute function must be named 'compute'");
    }

    // Check if the compute function is valid
    if (userFunc.returnType() != BoolType::get()) {
      throw std::runtime_error("Compute function must return a boolean");
    }
    if (userFunc.args().size() > 1 ||
        (userFunc.args().size() == 1 &&
         userFunc.args()[0].type() != Type::UINT32)) {
      throw std::runtime_error(
          "Compute function must either take no arguments or a single "
          "uint32_t argument, which is the worker id");
    }

    bool isMultiVertex = userFunc.args().size() == 1;

    // Begin vertex generation
    CodeGen::emitCode("class " + name + " : public ::poplar::" +
                      (isMultiVertex ? "MultiVertex" : "Vertex") + " {\n");
    CodeGen::emitCode("public:\n");

    // Emit the member variables
    for (const auto& field : fields_) {
      field.declare();
    }

    for (const auto& function : CodeGen::functions()) {
      CodeGen::emitCode(function.second);
      CodeGen::emitCode("\n");
    }

    CodeGen::emitCode("};");
  }

  /**
   * @brief Constructs and emits a Vertex with a given name, types, directions,
   * and compute function generator.
   * @param name The name of the vertex.
   * @param types The types of the vertex fields.
   * @param directions The directions of the vertex fields (Input, Output,
   * InOut).
   * @param computeFunctionGenerator The compute function generator. Must accept
   * a one \ref MemberVariable per field and return the compute function. The
   * generated function must be named "compute", accept a single argument (the
   * worker id) or no argument, and return a boolean.
   */
  template <typename F>
    requires ::graphene::detail::invocable_with_args_of<F, Value>
  Vertex(std::string name, std::vector<TypeRef> types,
         std::vector<VertexInOutType::Direction> directions,
         F computeFunctionGenerator)
      : Vertex(
            name, types, directions,
            [&computeFunctionGenerator](std::vector<MemberVariable> fields) {
              return graphene::detail::callFunctionWithUnpackedArgs<Function>(
                  computeFunctionGenerator, fields);
            }) {
    static_assert(
        std::is_same_v<
            typename graphene::detail::function_traits<F>::return_type,
            Function>,
        "Compute function generator must return a Function object");
  }

  const std::vector<MemberVariable>& fields() const { return fields_; }

 private:
  std::vector<MemberVariable> fields_;
};
}  // namespace graphene::codedsl