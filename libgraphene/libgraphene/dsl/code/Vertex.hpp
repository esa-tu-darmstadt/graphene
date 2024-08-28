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
   * @tparam F The type of the compute function generator.
   * @param name The name of the vertex.
   * @param types The types of the vertex fields.
   * @param directions The directions of the vertex fields (Input, Output,
   * InOut).
   * @param computeFunctionGenerator The compute function generator. Must take
   * the vertex fields as \ref Value arguments and return the compute \ref
   * Function.
   */
  template <typename F>
  Vertex(std::string name, std::vector<TypeRef> types,
         std::vector<VertexInOutType::Direction> directions,
         F computeFunctionGenerator) {
    static_assert(
        std::is_same_v<
            typename graphene::detail::function_traits<F>::return_type,
            Function>,
        "Compute function generator must return a Function object");

    // Create the fields
    for (size_t i = 0; i < types.size(); i++) {
      TypeRef type = *(types.begin() + i);
      auto direction = *(directions.begin() + i);
      fields_.emplace_back(VertexInOutType::get(direction, type));
    }

    // Generate the compute function
    Function userFunc =
        graphene::detail::callFunctionWithUnpackedArgs<Function>(
            computeFunctionGenerator, fields_);

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
    bool workerFuncHasCorrectName = userFunc.expr() == "compute";

    // Create the worker function, which is a wrapper around the user-provided
    // compute function with the correct name. If the user-provided compute
    // function already has the correct name, we can use it directly.
    Function workerFunc =
        workerFuncHasCorrectName ? userFunc
        : isMultiVertex
            ? Function("compute", Type::BOOL, {Type::UINT32},
                       [&](Parameter workerId) { return userFunc(workerId); })
            : Function("compute", Type::BOOL, {}, [&]() { return userFunc(); });

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

  const std::vector<MemberVariable>& fields() const { return fields_; }

 private:
  std::vector<MemberVariable> fields_;
};
}  // namespace graphene::codedsl