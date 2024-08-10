#pragma once

#include <filesystem>
#include <vector>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"

namespace graphene::matrix::host {

/** Loads a matrix from a file in MatrixMarket format. */
template <DataType Type>
TripletMatrix<Type> loadTripletMatrixFromFile(std::filesystem::path fileName);

/** Loads a vector from a file in MatrixMarket format. */
template <DataType Type>
DoubletVector<Type> loadDoubletVectorFromFile(std::filesystem::path fileName);

}  // namespace graphene::matrix::host