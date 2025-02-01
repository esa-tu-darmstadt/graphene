#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"

#include <spdlog/spdlog.h>

#include <execution>

#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::host {

template <DataType Type>
void sortTripletMatrx(TripletMatrix<Type> &tripletMatrix) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Sorting COO matrix in parallel");

  auto compareMatrixEntry = [](const auto &a, const auto &b) {
    return std::tie(a.row, a.col) < std::tie(b.row, b.col);
  };

  std::sort(std::execution::par_unseq, tripletMatrix.entries.begin(),
            tripletMatrix.entries.end(), compareMatrixEntry);
}

template void sortTripletMatrx(TripletMatrix<float> &tripletMatrix);
template void sortTripletMatrx(TripletMatrix<double> &tripletMatrix);
template void sortTripletMatrx(TripletMatrix<doubleword> &tripletMatrix);
}  // namespace graphene::matrix::host