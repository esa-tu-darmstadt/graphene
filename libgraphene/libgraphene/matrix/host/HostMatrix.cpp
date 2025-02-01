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

#include "libgraphene/matrix/host/HostMatrix.hpp"

#include <filesystem>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/details/Poisson.hpp"
#include "libgraphene/matrix/host/details/crs/CRSHostMatrix.hpp"
namespace graphene::matrix::host {

matrix::Matrix HostMatrix::copyToTile() const { return pimpl->copyToTile(); }

template <DataType Type>
HostMatrix::HostMatrix(MatrixFormat format, TripletMatrix<Type> tripletMatrix,
                       size_t numTiles, std::string name) {
  switch (format) {
    case MatrixFormat::CRS: {
      auto model = Runtime::instance().createResource<crs::CRSHostMatrix>(
          std::move(tripletMatrix), numTiles, name);
      pimpl = std::move(model);
      break;
    }
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

HostMatrix loadMatrixFromFile(TypeRef type, std::filesystem::path path,
                              size_t numTiles, MatrixFormat format,
                              std::string name) {
  assert(type->isFloat() && "Only floating point types are supported");
  return typeSwitch(type, [&]<FloatDataType T>() {
    TripletMatrix<T> tripletMatrix = loadTripletMatrixFromFile<T>(path);
    return HostMatrix(format, std::move(tripletMatrix), numTiles, name);
  });
}

HostMatrix generate3DPoissonMatrix(TypeRef type, size_t nx, size_t ny,
                                   size_t nz, size_t numTiles,
                                   MatrixFormat format, std::string name) {
  assert(type->isFloat() && "Only floating point types are supported");
  return typeSwitch(type, [&]<FloatDataType T>() {
    TripletMatrix<T> tripletMatrix =
        generate3DPoissonTripletMatrix<T>(nx, ny, nz);
    return HostMatrix(format, std::move(tripletMatrix), numTiles, name);
  });
}

// Template instantiations

}  // namespace graphene::matrix::host