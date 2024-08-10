#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"
namespace graphene::matrix {
template <DataType Type>
class Matrix;

namespace host {

template <DataType Type>
class HostMatrixBase
    : public Runtime::HostResource,
      public std::enable_shared_from_this<HostMatrixBase<Type>> {
 protected:
  size_t numTiles_;
  std::string name_;

  std::vector<TileLayout> tileLayout_;
  Partitioning partitioning_;

  bool multicolorRecommended_ = false;

 public:
  HostMatrixBase(size_t numTiles, std::string name)
      : numTiles_(numTiles), name_(name) {}
  HostMatrixBase(const HostMatrixBase &other) = default;
  HostMatrixBase(HostMatrixBase &&other) = default;

  virtual ~HostMatrixBase() = default;

  const TileLayout &getTileLayout(size_t proci) const {
    return tileLayout_[proci];
  }

  std::tuple<poplar::Graph::TileToTensorMapping, std::vector<size_t>>
  getVectorTileMappingAndShape(bool withHalo = false) const;

  virtual matrix::Matrix<Type> copyToTile() = 0;
  virtual MatrixFormat getFormat() const = 0;
  virtual size_t numTiles() const { return numTiles_; }

  bool multicolorRecommended() const { return multicolorRecommended_; }

  virtual HostValue<Type> loadVectorFromFile(std::string fileName,
                                             bool withHalo = false,
                                             std::string name = "vector") const;

  virtual HostValue<Type> decomposeVector(const std::vector<Type> &vector,
                                          bool includeHaloCells,
                                          std::string name = "vector") const;
};
}  // namespace host
}  // namespace graphene::matrix