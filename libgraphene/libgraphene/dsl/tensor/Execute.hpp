#pragma once

#include <functional>
#include <poplar/Tensor.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/Execute.hpp"
#include "libgraphene/dsl/code/Vertex.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
namespace graphene {

/// This file contains functions to execute code written in CodeDSL that
/// connects to TensorDSL tensors.

/// Describes a connection of a tensor to a vertex as an input.
inline codedsl::Vertex::MemberVarInfo In(const Tensor& tensor) {
  return codedsl::Vertex::MemberVarInfo::create(
      tensor.type(), tensor.tensor(),
      codedsl::VertexInOutType::Direction::Input);
}

/// Describes a connection of a tensor to a vertex as an output.
inline codedsl::Vertex::MemberVarInfo Out(const Tensor& tensor) {
  return codedsl::Vertex::MemberVarInfo::create(
      tensor.type(), tensor.tensor(),
      codedsl::VertexInOutType::Direction::Output);
}

/// Describes a connection of a tensor to a vertex as an input and output.
inline codedsl::Vertex::MemberVarInfo InOut(const Tensor& tensor) {
  return codedsl::Vertex::MemberVarInfo::create(
      tensor.type(), tensor.tensor(),
      codedsl::VertexInOutType::Direction::InOut);
}

/// Describes an additional member variable of a vertex.
inline codedsl::Vertex::MemberVarInfo Member(TypeRef type,
                                             CTypeQualifiers qualifiers = {}) {
  return codedsl::Vertex::MemberVarInfo::create(type, qualifiers);
}

/// Executes the given code on a single worker thread. Generates a `Vertex`.
template <typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void Execute(F&& code, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::Vertex,
                           std::forward<F>(code));
}

/// Execute the code on all worker threads. The first
/// argument of the code function will be the worker ID. Generates a
/// `MultiVertex`
template <typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void ExecuteThreaded(F&& code, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::MultiVertex,
                           std::forward<F>(code));
}

/// Executes the given code on a supervisor thread. Generates a
/// `SupervisorVertex`.
template <typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void ExecuteOnSupervisor(F&& code, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::SupervisorVertex,
                           std::forward<F>(code));
}

}  // namespace graphene