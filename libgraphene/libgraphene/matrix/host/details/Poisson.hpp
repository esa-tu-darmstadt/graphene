#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"

namespace graphene::matrix::host {

/** Generates a 7-point 3D Poisson matrix in triplet format. */
template <FloatDataType Type>
TripletMatrix<Type> generate3DPoissonTripletMatrix(size_t nx, size_t ny,
                                                   size_t nz);
}  // namespace graphene::matrix::host