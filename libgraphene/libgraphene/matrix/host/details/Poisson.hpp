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
#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"

namespace graphene::matrix::host {

/** Generates a 7-point 3D Poisson matrix in triplet format. */
template <FloatDataType Type>
TripletMatrix<Type> generate3DPoissonTripletMatrix(size_t nx, size_t ny,
                                                   size_t nz);
}  // namespace graphene::matrix::host