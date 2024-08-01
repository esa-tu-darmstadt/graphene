#pragma once

#include <poplar/Tensor.hpp>

template <>
struct std::hash<poplar::Tensor> {
  std::size_t operator()(const poplar::Tensor &k) const {
    using std::hash;
    using std::size_t;
    using std::string;

    return hash<string>()(k.getVarStr());
  }
};
