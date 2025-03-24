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

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"

namespace graphene::matrix::host {
/** Represents a matrix in host memory. */

class HostMatrix {
  std::shared_ptr<HostMatrixBase> pimpl;

 public:
  HostMatrix() = default;

  template <DataType Type>
  HostMatrix(MatrixFormat format, TripletMatrix<Type> tripletMatrix,
             size_t numTiles, std::string name = "matrix");

  template <typename Impl>
  HostMatrix(std::shared_ptr<Impl> impl) : pimpl(std::move(impl)) {}

  template <typename T = HostMatrixBase>
  const T &getImpl() const {
    return static_cast<T &>(*pimpl);
  }

  bool multicolorRecommended() const { return pimpl->multicolorRecommended(); }

  const TileLayout &getTileLayout(size_t proci) const {
    return pimpl->getTileLayout(proci);
  }

  matrix::Matrix copyToTile() const;

  /// Returns the distributed shape of a vector that is compatible with the
  /// matrix. The vector can include or exclude halo cells based on the
  /// withHalo parameter. The width parameter specifies the number of elements
  /// per row in the vector. If width is 0, the resulting shape will be rank 1.
  /// If width is greater than 0, the resulting shape will be rank 2, with the
  /// second dimension having the specified width.
  DistributedShape getVectorShape(bool withHalo = false,
                                  size_t width = 0) const {
    return pimpl->getVectorShape(withHalo, width);
  }

  MatrixFormat getFormat() const { return pimpl->getFormat(); }
  size_t numTiles() const { return pimpl->numTiles(); }

  /// Loads a vector from a file, and decomposes it into a \ref HostTensor with
  /// the given data type.
  HostTensor loadVectorFromFile(TypeRef type, std::string fileName,
                                bool withHalo = false,
                                std::string name = "vector") const {
    return pimpl->loadVectorFromFile(type, fileName, withHalo, name);
  }

  /// Decomposes a vector into a \ref HostTensor with the given data type
  /// according to the tile layout of the matrix.
  template <DataType Type>
  HostTensor decomposeVector(const std::span<Type> &vector,
                             bool includeHaloCells,
                             TypeRef destType = getType<Type>(),
                             std::string name = "vector") const {
    return pimpl->decomposeVector(vector, includeHaloCells, destType, name);
  }
};

HostMatrix loadMatrixFromFile(TypeRef type, std::filesystem::path path,
                              size_t numTiles,
                              MatrixFormat format = MatrixFormat::CRS,
                              std::string name = "matrix");

HostMatrix generate3DPoissonMatrix(TypeRef type, size_t nx, size_t ny,
                                   size_t nz, size_t numTiles,
                                   MatrixFormat format = MatrixFormat::CRS,
                                   std::string name = "poisson");

}  // namespace graphene::matrix::host