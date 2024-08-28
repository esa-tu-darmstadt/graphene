#pragma once

#include <poplar/Tensor.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/Execute.hpp"
#include "libgraphene/dsl/code/VertexTypes.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
namespace graphene {

enum class VertexDirection : size_t { Input = 0, Output, InOut };

template <typename F, typename... Args>
void Execute(std::tuple<Tensor<Args>&...> tensors,
             std::vector<VertexDirection> directions, F code) {
  std::vector<poplar::Tensor> poplarTensors;
  poplarTensors.reserve(sizeof...(Args));
  std::apply(
      [&](auto&... args) { (poplarTensors.push_back(args.tensor()), ...); },
      tensors);

  std::vector<TypeRef> tensorTypes;
  tensorTypes.reserve(sizeof...(Args));
  std::apply([&](auto&... args) { (tensorTypes.push_back(args.type()), ...); },
             tensors);

  std::vector<codedsl::VertexInOutType::Direction> directionsConverted;
  directionsConverted.reserve(directions.size());
  for (const auto& direction : directions) {
    directionsConverted.push_back(
        static_cast<codedsl::VertexInOutType::Direction>(direction));
  }

  codedsl::ExecuteAsMapped(poplarTensors, tensorTypes, directionsConverted,
                           code);
}
}  // namespace graphene