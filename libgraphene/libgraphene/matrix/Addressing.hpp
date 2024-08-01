#pragma once

#include <poplar/Graph.hpp>

namespace graphene::matrix {
class Addressing {
 public:
  poplar::Graph::TileToTensorMapping getVectorMapping(
      bool withHalo = false) const;
  std::vector<size_t> getVectorShape(bool withHalo = false) const;
};
}  // namespace graphene::matrix