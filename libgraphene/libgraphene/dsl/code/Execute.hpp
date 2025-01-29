#pragma once

#include <fstream>
#include <poplar/CodeletFileType.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <sstream>

#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/ControlFlow.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/code/Vertex.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"
namespace graphene::codedsl {
/**
 * @brief Executes the provided code function on each tile, operating on the
 * slices of the given tensors that are mapped to the tile.
 * @tparam F The type of the code function.
 * @param tensors The input tensors.
 * @param tensorTypes The types of the tensors.
 * @param directions The directions of the tensors.
 * @param multiVertex True, if the code should be executed on all worker
 * threads.
 * @param code The code of the function. Must accept a vector of \ref Value, one
 * for each tensor.
 * @param broadcastTensors True, tensors will be broadcast to all tiles if (1.)
 * their first dimension is 1 or (2.) their rank is less than the rank of the
 * tensor with the highest rank.
 */
void ExecuteAsMapped(std::vector<poplar::Tensor> tensors,
                     std::vector<TypeRef> tensorTypes,
                     std::vector<VertexInOutType::Direction> directions,
                     bool multiVertex,
                     std::function<void(std::vector<Value>)> code,
                     bool broadcastTensors = true);

/**
 * @brief Executes the provided code function on each tile, operating on the
 * slices of the given tensors that are mapped to the tile.
 * @tparam F The type of the code function.
 * @param tensors The input tensors.
 * @param tensorTypes The types of the tensors.
 * @param directions The directions of the tensors.
 * @param multiVertex True, if the code should be executed on all worker
 * threads.
 * @param code The code function to execute. Must accept one argument of type
 * \ref Value per tensor. If \p multiVertex is true, the first argument will be
 * the worker ID.
 */
template <typename F>
  requires ::graphene::detail::invocable_with_args_of<F, Value>
void ExecuteAsMapped(std::vector<poplar::Tensor> tensors,
                     std::vector<TypeRef> tensorTypes,
                     std::vector<VertexInOutType::Direction> directions,
                     bool multiVertex, F code) {
  ExecuteAsMapped(tensors, tensorTypes, directions, multiVertex,
                  [&code](std::vector<Value> args) {
                    ::graphene::detail::callFunctionWithUnpackedArgs<void>(
                        code, args);
                  });
}
}  // namespace graphene::codedsl