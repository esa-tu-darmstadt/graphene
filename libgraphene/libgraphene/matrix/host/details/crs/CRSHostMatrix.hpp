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

#include <filesystem>
#include <memory>
#include <poplar/Graph.hpp>
#include <set>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
namespace graphene::matrix::host::crs {
class CRSHostMatrix : public HostMatrixBase {
  struct CRSAddressing {
    std::vector<size_t> rowPtr;
    std::vector<size_t> colInd;
  };
  template <FloatDataType Type>
  struct CRSMatrixValues {
    std::vector<Type> offDiagValues;
    std::vector<Type> diagValues;
  };

  template <FloatDataType Type>
  static std::tuple<CRSAddressing, CRSMatrixValues<Type>> convertToCRS(
      TripletMatrix<Type> tripletMatrix);
  template <FloatDataType Type>
  static Partitioning calculatePartitioning(size_t &numTiles,
                                            const CRSMatrixValues<Type> &crs,
                                            const CRSAddressing &addressing);
  static std::vector<TileLayout> calculateTileLayouts(
      const Partitioning &partitioning, const CRSAddressing &matrix);
  static std::vector<CRSAddressing> calculateLocalAddressings(
      const CRSAddressing &globalAddressing,
      const std::vector<TileLayout> &tileLayouts);

  template <FloatDataType Type>
  void decomposeValues(const CRSMatrixValues<Type> &globalMatrix);

  void calculateRowColors();
  void calculateColorAddressings();

  // the original addressing of the matrix
  CRSAddressing globalAddressing_;

  // the decomposed addressings of the matrix
  std::vector<CRSAddressing> localAddressings_;

  // For each tile, the color of each row
  std::vector<std::vector<size_t>> rowColors_;
  // For each tile, the number of colors
  std::vector<size_t> numColors_;

  HostTensor rowPtr_;
  HostTensor colInd_;

  HostTensor colorSortAddr;
  HostTensor colorSortStartPtr;

  HostTensor offDiagValues_;
  HostTensor diagValues_;

 public:
  template <FloatDataType Type>
  explicit CRSHostMatrix(TripletMatrix<Type> tripletMatrix, size_t numTiles,
                         std::string name = "matrix");
  CRSHostMatrix(const CRSHostMatrix &other) = delete;
  CRSHostMatrix(CRSHostMatrix &&other) = delete;

  virtual matrix::Matrix copyToTile() override;

  template <FloatDataType Type>
  HostTensor decomposeOffDiagCoefficients(
      const std::vector<Type> &values,
      TypeRef targetType = getType<Type>()) const;

  MatrixFormat getFormat() const override { return MatrixFormat::CRS; }
};
}  // namespace graphene::matrix::host::crs