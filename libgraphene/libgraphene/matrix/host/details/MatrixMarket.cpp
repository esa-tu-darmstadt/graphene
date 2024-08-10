#include "libgraphene/matrix/host/details/MatrixMarket.hpp"

#include <spdlog/spdlog.h>

#include <execution>
#include <fstream>

#include "fast_matrix_market/app/doublet.hpp"
#include "fast_matrix_market/app/triplet.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::host {

template <DataType Type>
TripletMatrix<Type> loadTripletMatrixFromFile(std::filesystem::path path) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Loading COO matrix from {}", path.string());

  std::ifstream matrixFile(path);

  if (!matrixFile.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  struct triplet_matrix {
    size_t nrows = 0, ncols = 0;
    std::vector<size_t> rows, cols;
    std::vector<Type> vals;
  } mat;
  fast_matrix_market::read_matrix_market_triplet(
      matrixFile, mat.nrows, mat.ncols, mat.rows, mat.cols, mat.vals);
  matrixFile.close();

  spdlog::info("Matrix has {} rows, {} columns and {} non-zero entries",
               mat.nrows, mat.ncols, mat.vals.size());
  TripletMatrix<Type> coo;
  coo.ncols = mat.ncols;
  coo.nrows = mat.nrows;

  coo.entries.reserve(mat.vals.size());
  for (size_t i = 0; i < mat.vals.size(); i++) {
    coo.entries.push_back({mat.rows[i], mat.cols[i], mat.vals[i]});
  }

  return coo;
}

template <DataType Type>
DoubletVector<Type> loadDoubletVectorFromFile(std::filesystem::path path) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Loading COO vector from {}", path.string());
  std::ifstream fileStream(path);
  if (!fileStream.is_open()) {
    throw std::runtime_error("Failed to open file");
  }

  DoubletVector<Type> vector;

  fast_matrix_market::matrix_market_header header;
  fast_matrix_market::read_header(fileStream, header);
  if (header.object == fast_matrix_market::matrix) {
    spdlog::debug("Matrix market file {} is a matrix, reading first column",
                  path.string());
    fileStream.close();
    TripletMatrix<Type> matrix = loadTripletMatrixFromFile<Type>(path);
    // Copy the first column of the matrix to a vector
    vector.nrows = matrix.nrows;
    vector.indices.reserve(matrix.nrows);
    vector.values.reserve(matrix.nrows);
    for (size_t i = 0; i < matrix.entries.size(); ++i) {
      if (matrix.entries[i].col != 0) continue;
      vector.indices.push_back(matrix.entries[i].row);
      vector.values.push_back(matrix.entries[i].val);
    }
  } else if (header.object == fast_matrix_market::vector) {
    fileStream.seekg(0);
    spdlog::debug("Matrix market file {} is a vector, reading it",
                  path.string());
    fast_matrix_market::read_matrix_market_doublet(
        fileStream, vector.nrows, vector.indices, vector.values);
    fileStream.close();
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported object type of matrix market file: {}",
                    fast_matrix_market::object_map.at(header.object)));
  }
  return vector;
}

template TripletMatrix<float> loadTripletMatrixFromFile(
    std::filesystem::path path);
template DoubletVector<float> loadDoubletVectorFromFile(
    std::filesystem::path path);
}  // namespace graphene::matrix::host