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

/// Executes code on one worker thread per tile for every tile that contains
/// mapped tensor data. Generates a `Vertex`.
template <bool broadcastTensors = false, typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void Execute(F&& code, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::Vertex,
                           std::forward<F>(code), broadcastTensors);
}

/// Executes code on a single worker thread on the given tile. Generates a
/// `Vertex`.
template <bool broadcastTensors = false, typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void ExecuteOnSingleTile(F&& code, size_t tile, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::Vertex,
                           std::forward<F>(code), broadcastTensors);
}

/// Execute the code on all worker threads. The first
/// argument of the code function will be the worker ID. Generates a
/// `MultiVertex`
template <bool broadcastTensors = false, typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void ExecuteThreaded(F&& code, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::MultiVertex,
                           std::forward<F>(code), broadcastTensors);
}

/// Executes the given code on a supervisor thread. Generates a
/// `SupervisorVertex`.
template <bool broadcastTensors = false, typename F, typename... MemberVars>
  requires(std::is_same_v<std::decay_t<MemberVars>,
                          codedsl::Vertex::MemberVarInfo> &&
           ...)
void ExecuteOnSupervisor(F&& code, MemberVars&&... vars) {
  codedsl::ExecuteAsMapped({vars...}, codedsl::VertexKind::SupervisorVertex,
                           std::forward<F>(code), broadcastTensors);
}

}  // namespace graphene