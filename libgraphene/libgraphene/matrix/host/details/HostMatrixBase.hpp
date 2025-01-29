#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"
namespace graphene::matrix {
class Matrix;

namespace host {

class HostMatrixBase : public Runtime::HostResource,
                       public std::enable_shared_from_this<HostMatrixBase> {
 protected:
  size_t numTiles_;
  std::string name_;

  std::vector<TileLayout> tileLayout_;
  Partitioning partitioning_;

  bool multicolorRecommended_ = false;

 public:
  HostMatrixBase(size_t numTiles, std::string name)
      : numTiles_(numTiles), name_(name) {}
  HostMatrixBase(const HostMatrixBase &other) = delete;
  HostMatrixBase(HostMatrixBase &&other) = delete;

  virtual ~HostMatrixBase() = default;

  const TileLayout &getTileLayout(size_t proci) const {
    return tileLayout_[proci];
  }

  DistributedShape getVectorShape(bool withHalo = false,
                                  size_t width = 0) const;

  virtual matrix::Matrix copyToTile() = 0;
  virtual MatrixFormat getFormat() const = 0;
  virtual size_t numTiles() const { return numTiles_; }

  bool multicolorRecommended() const { return multicolorRecommended_; }

  HostTensor loadVectorFromFile(TypeRef type, std::string fileName,
                                bool withHalo = false,
                                std::string name = "vector") const;

  template <DataType Type>
  HostTensor decomposeVector(const std::vector<Type> &vector,
                             bool includeHaloCells,
                             TypeRef destType = getType<Type>(),
                             std::string name = "vector") const;
};
}  // namespace host
}  // namespace graphene::matrix