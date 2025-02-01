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


#include "libgraphene/matrix/host/details/Poisson.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/common/Concepts.hpp"
namespace graphene::matrix::host {

template <FloatDataType Type>
TripletMatrix<Type> generate3DPoissonTripletMatrix(size_t nx, size_t ny,
                                                   size_t nz) {
  const size_t stencil = 7;
  TripletMatrix<Type> tripletMatrix;
  tripletMatrix.nrows = nx * ny * nz;
  tripletMatrix.ncols = nx * ny * nz;
  tripletMatrix.entries.reserve(stencil * tripletMatrix.nrows);

  spdlog::debug("Generating 7-point 3D Poisson matrix with dimensions {}x{}x{}",
                nx, ny, nz);

  for (size_t k = 0; k < nz; ++k) {
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        size_t row = k * nx * ny + j * nx + i;
        tripletMatrix.entries.push_back({row, row, 6.0f});
        if (i > 0) {
          tripletMatrix.entries.push_back({row, row - 1, -1.0f});
        }
        if (i < nx - 1) {
          tripletMatrix.entries.push_back({row, row + 1, -1.0f});
        }
        if (j > 0) {
          tripletMatrix.entries.push_back({row, row - nx, -1.0f});
        }
        if (j < ny - 1) {
          tripletMatrix.entries.push_back({row, row + nx, -1.0f});
        }
        if (k > 0) {
          tripletMatrix.entries.push_back({row, row - nx * ny, -1.0f});
        }
        if (k < nz - 1) {
          tripletMatrix.entries.push_back({row, row + nx * ny, -1.0f});
        }
      }
    }
  }

  spdlog::trace("Generated matrix has {} rows and {} non-zero entries",
                tripletMatrix.nrows, tripletMatrix.entries.size());

  return tripletMatrix;
}

// Template instantiations
template TripletMatrix<float> generate3DPoissonTripletMatrix(size_t nx,
                                                             size_t ny,
                                                             size_t nz);
template TripletMatrix<double> generate3DPoissonTripletMatrix(size_t nx,
                                                              size_t ny,
                                                              size_t nz);
template TripletMatrix<doubleword> generate3DPoissonTripletMatrix(size_t nx,
                                                                  size_t ny,
                                                                  size_t nz);

}  // namespace graphene::matrix::host