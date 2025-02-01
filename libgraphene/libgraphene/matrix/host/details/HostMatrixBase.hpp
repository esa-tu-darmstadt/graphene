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