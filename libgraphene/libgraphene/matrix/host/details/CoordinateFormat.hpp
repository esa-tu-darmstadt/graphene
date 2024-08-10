#pragma once

#include <filesystem>
#include <vector>

#include "libgraphene/common/Concepts.hpp"

namespace graphene::matrix::host {

/** A partitioning of the rows of a matrix. */
struct Partitioning {
  std::vector<size_t> rowToTile;
};

/** A matrix in triplet (coordinate) format. */
template <DataType Type>
struct TripletMatrix {
  struct Entry {
    size_t row, col;
    Type val;
  };
  std::vector<Entry> entries;
  size_t nrows = 0, ncols = 0;
};

/** A vector in doublet (coordinate) format. */
template <DataType Type>
struct DoubletVector {
  size_t nrows = 0;
  std::vector<size_t> indices;
  std::vector<Type> values;
};

/** Sorts the matrix in coordinate / triplet format. */
template <DataType Type>
void sortTripletMatrx(TripletMatrix<Type> &tripletMatrix);

}  // namespace graphene::matrix::host