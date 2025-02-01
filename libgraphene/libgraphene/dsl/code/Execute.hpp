#pragma once

#include <fstream>
#include <poplar/CodeletFileType.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <sstream>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/code/Vertex.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
namespace graphene::codedsl {

/**
 * @brief Executes the provided code function on each tile, operating on the
 * slices of the given tensors that are mapped to the tile.
 * @details Generates the source code for a vertex with the given code function,
 * compiles it, adds it to the graph, and schedules it to run on each that has
 * data mapped to it.
 * @param vars The member variables of the vertex. Each member variable can be
 * either a tensor or an additional member variable.
 * @param code The code of the function. Must accept a vector of \ref Value, one
 * for each tensor and the additional member variables.
 * @param broadcastTensors True, tensors will be broadcast to all tiles if (1.)
 * their first dimension is 1 or (2.) their rank is less than the rank of the
 * tensor with the highest rank.
 */
void ExecuteAsMapped(std::vector<Vertex::MemberVarInfo> vars, VertexKind kind,
                     std::function<void(std::vector<Value>)> code,
                     bool broadcastTensors = true);

/**
 * @brief Executes the provided code function on each tile, operating on the
 * slices of the given tensors that are mapped to the tile.
 * @tparam F The type of the code function.
 * @param vars The member variables of the vertex. Each member variable can be
 * either a tensor or an additional member variable.
 * @param kind The kind (Vertex, MultiVertex, SupervisorVertex) of the vertex to
 * generate.
 * @param code The code function to execute. Must accept one argument of type
 * \ref Value per tensor. If \p multiVertex is true, the first argument will be
 * the worker ID.
 */
template <typename F>
  requires ::graphene::detail::invocable_with_args_of<F, Value>
void ExecuteAsMapped(std::vector<Vertex::MemberVarInfo> vars, VertexKind kind,
                     F code, bool broadcastTensors = true) {
  ExecuteAsMapped(
      vars, kind,
      [&code](std::vector<Value> args) {
        ::graphene::detail::callFunctionWithUnpackedArgs<void>(code, args);
      },
      broadcastTensors);
}
}  // namespace graphene::codedsl