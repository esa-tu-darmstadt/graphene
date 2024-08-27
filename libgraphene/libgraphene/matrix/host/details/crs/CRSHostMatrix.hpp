#pragma once

#include <filesystem>
#include <memory>
#include <poplar/Graph.hpp>
#include <set>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/dsl/tensor/HostTensorVariant.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
namespace graphene::matrix::host::crs {
template <DataType Type>
class CRSHostMatrix : public HostMatrixBase<Type> {
  struct CRSAddressing {
    std::vector<size_t> rowPtr;
    std::vector<size_t> colInd;
  };
  struct CRSMatrix {
    CRSAddressing addressing;
    std::vector<Type> offDiagValues;
    std::vector<Type> diagValues;
  };

  static CRSMatrix convertToCRS(TripletMatrix<Type> tripletMatrix);
  static Partitioning calculatePartitioning(size_t numTiles,
                                            const CRSMatrix &crs);
  static std::vector<TileLayout> calculateTileLayouts(
      const Partitioning &partitioning, const CRSMatrix &matrix);

  void calculateLocalAddressings();
  void calculateRowColors();
  void calculateColorAddressings();
  void decomposeValues();

  CRSMatrix globalMatrix_;
  std::vector<CRSAddressing> localAddressings_;

  // For each tile, the color of each row
  std::vector<std::vector<size_t>> rowColors_;
  // For each tile, the number of colors
  std::vector<size_t> numColors_;

  AnyUIntHostValue rowPtr_;
  AnyUIntHostValue colInd_;

  AnyUIntHostValue colorSortAddr;
  AnyUIntHostValue colorSortStartPtr;

  HostTensor<Type> offDiagValues_;
  HostTensor<Type> diagValues_;

 public:
  explicit CRSHostMatrix(TripletMatrix<Type> tripletMatrix, size_t numTiles,
                         std::string name = "matrix");
  CRSHostMatrix(const CRSHostMatrix &other) = delete;
  CRSHostMatrix(CRSHostMatrix &&other) = default;

  virtual matrix::Matrix<Type> copyToTile() override;

  HostTensor<Type> decomposeOffDiagCoefficients(
      const std::vector<Type> &values) const;

  MatrixFormat getFormat() const override { return MatrixFormat::CRS; }
};
}  // namespace graphene::matrix::host::crs