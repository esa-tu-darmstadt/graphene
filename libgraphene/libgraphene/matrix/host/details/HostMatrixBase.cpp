#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"

namespace graphene::matrix::host {
template <DataType Type>
std::tuple<poplar::Graph::TileToTensorMapping, std::vector<size_t>>
HostMatrixBase<Type>::getVectorTileMappingAndShape(bool withHalo) const {
  // Calculate the tile mapping and the number of rows. The number of rows may
  // differ from vector.size() if halo cells are included.
  size_t numRows = 0;
  size_t numElements = 0;
  poplar::Graph::TileToTensorMapping mapping(tileLayout_.size());
  for (const auto &tile : tileLayout_) {
    size_t rowsOnThisTile = 0;
    rowsOnThisTile += tile.numInteriorRows();
    rowsOnThisTile += tile.numSeperatorRows();
    if (withHalo) rowsOnThisTile += tile.numHaloRows();
    mapping[tile.tileId].emplace_back(numElements,
                                      numElements + rowsOnThisTile);
    numElements += rowsOnThisTile;
    numRows += rowsOnThisTile;
  }

  return {mapping, {numRows}};
}

template <DataType Type>
HostTensor<Type> HostMatrixBase<Type>::decomposeVector(
    const std::vector<Type> &vector, bool includeHaloCells,
    std::string name) const {
  auto [mapping, shape] = getVectorTileMappingAndShape(includeHaloCells);

  // Decompose the vector
  std::vector<Type> decomposedVector;
  decomposedVector.reserve(shape[0]);
  for (const auto &tile : tileLayout_) {
    for (size_t localRow = 0; localRow < tile.localToGlobalRow.size();
         ++localRow) {
      size_t globalRow = tile.localToGlobalRow[localRow];
      if (!includeHaloCells && tile.isHalo(localRow)) continue;
      // CHANGEME: Set halo cells to zero to check if halo exchange is working
      decomposedVector.push_back(vector[globalRow]);
    }
  }

  return HostTensor<Type>(std::move(decomposedVector), std::move(shape),
                          std::move(mapping), std::move(name));
}

template <DataType Type>
HostTensor<Type> HostMatrixBase<Type>::loadVectorFromFile(
    std::string fileName, bool withHalo, std::string name) const {
  GRAPHENE_TRACEPOINT();
  DoubletVector<Type> vector = loadDoubletVectorFromFile<Type>(fileName);

  // Convert the doublet vector to a contiguous vector. This is necessary
  // because a doublet vector is not guaranteed to be sorted by index and may
  // contain missing indices.
  std::vector<Type> contiguousValues(vector.nrows);
  for (size_t i = 0; i < vector.values.size(); i++) {
    contiguousValues[vector.indices[i]] = vector.values[i];
  }

  return decomposeVector(contiguousValues, withHalo, name);
}

// Template instantiations
template class HostMatrixBase<float>;
}  // namespace graphene::matrix::host