
#include "libgraphene/matrix/host/details/Poisson.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/common/Concepts.hpp"
namespace graphene::matrix::host {

template <DataType Type>
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

}  // namespace graphene::matrix::host