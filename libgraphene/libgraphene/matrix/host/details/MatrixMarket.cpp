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

#include "libgraphene/matrix/host/details/MatrixMarket.hpp"

#include <spdlog/spdlog.h>

#include <execution>
#include <fstream>

#include "fast_matrix_market/app/doublet.hpp"
#include "fast_matrix_market/app/triplet.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::host {

template <FloatDataType Type>
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

template <FloatDataType Type>
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

template <>
DoubletVector<doubleword> loadDoubletVectorFromFile(
    std::filesystem::path path) {
  DoubletVector<double> doubleVector = loadDoubletVectorFromFile<double>(path);
  DoubletVector<doubleword> doublewordVector;
  doublewordVector.nrows = doubleVector.nrows;
  doublewordVector.indices = std::move(doubleVector.indices);
  doublewordVector.values.reserve(doubleVector.values.size());
  for (auto val : doubleVector.values) {
    doublewordVector.values.push_back(doubleword::from(val));
  }
  return doublewordVector;
}

template <>
TripletMatrix<doubleword> loadTripletMatrixFromFile(
    std::filesystem::path path) {
  TripletMatrix<double> doubleMatrix = loadTripletMatrixFromFile<double>(path);
  TripletMatrix<doubleword> doublewordMatrix;
  doublewordMatrix.ncols = doubleMatrix.ncols;
  doublewordMatrix.nrows = doubleMatrix.nrows;
  doublewordMatrix.entries.reserve(doubleMatrix.entries.size());
  for (auto entry : doubleMatrix.entries) {
    doublewordMatrix.entries.push_back(
        {entry.row, entry.col, doubleword::from(entry.val)});
  }
  return doublewordMatrix;
}

template TripletMatrix<float> loadTripletMatrixFromFile(
    std::filesystem::path path);
template DoubletVector<float> loadDoubletVectorFromFile(
    std::filesystem::path path);
template TripletMatrix<double> loadTripletMatrixFromFile(
    std::filesystem::path path);
template DoubletVector<double> loadDoubletVectorFromFile(
    std::filesystem::path path);
}  // namespace graphene::matrix::host