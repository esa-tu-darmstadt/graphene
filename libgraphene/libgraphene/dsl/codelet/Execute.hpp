#pragma once

#include <poplar/CodeletFileType.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/codelet/ControlFlow.hpp"
#include "libgraphene/dsl/codelet/Function.hpp"
#include "libgraphene/dsl/codelet/Vertex.hpp"
#include "libgraphene/dsl/codelet/VertexTypes.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
namespace graphene::codelet::dsl {
/**
 * @brief Executes the provided code function on each tile, operating on the
 * slices of the given tensors that are mapped to the tile.
 * @tparam F The type of the code function.
 * @param tensors The input tensors.
 * @param tensorTypes The types of the tensors.
 * @param directions The directions of the tensors.
 * @param code The code function to execute.
 */
template <typename F>
void ExecuteAsMapped(std::vector<poplar::Tensor> tensors,
                     std::vector<TypeRef> tensorTypes,
                     std::vector<VertexInOutType::Direction> directions, F code)
  requires(
      std::is_same_v<typename detail::function_traits<F>::return_type, void>)
{
  if (tensorTypes.size() != tensors.size() ||
      directions.size() != tensors.size()) {
    throw std::runtime_error(
        "Number of tensors, types, and directions must "
        "match");
  }

  constexpr size_t numTensors = detail::function_traits<F>::arity;
  if (numTensors != tensors.size())
    throw std::runtime_error(
        "The user code must take one argument for each "
        "tensor");

  using TupleArgsType = typename detail::function_traits<F>::args_type;
  using FunctionGeneratorType =
      typename graphene::detail::apply_tuple_args<Function::FunctionGenerator,
                                                  TupleArgsType>::type;

  FunctionGeneratorType transformed_code = [code](auto... args) -> Function {
    return Function("compute", Type::BOOL, {}, [&]() -> void {
      code(args...);
      // Return true if the user code does not do so
      Return(true);
    });
  };

  // Wrap each tensor type in a VertexVectorType
  std::vector<TypeRef> transformedTypes;
  for (const auto& type : tensorTypes) {
    transformedTypes.push_back(VertexVectorType::get(type));
  }

  // Emit the necessary includes
  CodeGen::emitInclude("poplar/Vertex.hpp", true);
  CodeGen::emitInclude("print.h", true);

  // Generate the vertex
  std::string vertexName = CodeGen::generateVertexName();
  Vertex vertex(vertexName, transformedTypes, directions, transformed_code);

  std::stringstream ss = CodeGen::reset();
  spdlog::trace("Generated vertex: {}", ss.str());
  auto& graph = Context::graph();

  // Compile the vertex
  graph.addCodelets(ss, "-O3", poplar::CodeletFileType::CppSource);

  // Get the tensor mappings
  std::vector<poplar::Graph::TileToTensorMapping> tensorMappings;
  for (const auto& tensor : tensors) {
    tensorMappings.push_back(graph.getTileMapping(tensor));
  }

  // Add an instance of the vertex to each tile
  DebugInfo di("CodeDSL");
  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < graph.getTarget().getNumTiles(); tile++) {
    // Check if the vertex has any data mapped to this tile
    bool isEmpty = true;
    for (size_t i = 0; i < tensors.size(); i++) {
      if (sliceTensorToTile(tensors[i], tile, &tensorMappings[i]).valid()) {
        isEmpty = false;
        break;
      }
    }
    if (isEmpty) continue;

    poplar::VertexRef v = graph.addVertex(cs, vertexName);
    graph.setTileMapping(v, tile);

    // Connect the vertex to the tensors
    for (size_t i = 0; i < tensors.size(); i++) {
      // Either broadcast or slice the tensor to the tile
      poplar::Tensor localTensor =
          tensors[i].dim(0) == 1
              ? tensors[i]
              : sliceTensorToTile(tensors[i], tile, &tensorMappings[i]);
      graph.connect(v[vertex.fields()[i].expr()], localTensor.flatten());
    }
  }

  // Add the compute set to the program
  Context::program().add(poplar::program::Execute(cs, di));
}
}  // namespace graphene::codelet::dsl