#pragma once

#include <functional>
#include <poplar/Tensor.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/Execute.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
namespace graphene {

enum class VertexDirection : size_t { Input = 0, Output, InOut };

namespace details {
/// A small helper struct that carries both the Tensor reference and its
/// direction.
struct DirectionedTensor {
  std::reference_wrapper<const Tensor> tensor;
  VertexDirection dir;
};

}  // namespace details
/// Helper functions to build a DirectionedTensor object with the given
/// direction.
inline details::DirectionedTensor In(const Tensor& tensor) {
  return {std::ref(tensor), VertexDirection::Input};
}
inline details::DirectionedTensor Out(Tensor& tensor) {
  return {std::ref(tensor), VertexDirection::Output};
}
inline details::DirectionedTensor InOut(Tensor& tensor) {
  return {std::ref(tensor), VertexDirection::InOut};
}

template <typename F, typename... DirTensors>
void Execute(F&& code, DirTensors&&... dts) {
  // Collect poplar::Tensor and TypeRef
  std::vector<poplar::Tensor> poplarTensors = {dts.tensor.get().tensor()...};
  std::vector<TypeRef> tensorTypes = {dts.tensor.get().type()...};

  // Collect directions
  std::vector<codedsl::VertexInOutType::Direction> directionsConverted = {
      static_cast<codedsl::VertexInOutType::Direction>(dts.dir)...};
  bool multiVertex = false;
  codedsl::ExecuteAsMapped(poplarTensors, tensorTypes, directionsConverted,
                           multiVertex, std::forward<F>(code));
}

/// The first argument of the code function will be the worker ID.
template <typename F, typename... DirTensors>
void ExecuteThreaded(F&& code, DirTensors&&... dts) {
  // Collect poplar::Tensor and TypeRef
  std::vector<poplar::Tensor> poplarTensors = {dts.tensor.get().tensor()...};
  std::vector<TypeRef> tensorTypes = {dts.tensor.get().type()...};

  // Collect directions
  std::vector<codedsl::VertexInOutType::Direction> directionsConverted = {
      static_cast<codedsl::VertexInOutType::Direction>(dts.dir)...};
  bool multiVertex = true;
  codedsl::ExecuteAsMapped(poplarTensors, tensorTypes, directionsConverted,
                           multiVertex, std::forward<F>(code));
}

}  // namespace graphene