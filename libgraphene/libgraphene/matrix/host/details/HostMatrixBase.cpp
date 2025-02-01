#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"

namespace graphene::matrix::host {
DistributedShape HostMatrixBase::getVectorShape(bool withHalo,
                                                size_t width) const {
  FirstDimDistribution firstDimDistribution;
  firstDimDistribution.reserve(tileLayout_.back().tileId);
  size_t numRows = 0;
  for (const auto &tile : tileLayout_) {
    size_t rowsOnThisTile = 0;
    rowsOnThisTile += tile.numInteriorRows();
    rowsOnThisTile += tile.numSeperatorRows();
    if (withHalo) rowsOnThisTile += tile.numHaloRows();
    firstDimDistribution[tile.tileId] = rowsOnThisTile;
    numRows += rowsOnThisTile;
  }

  TensorShape globalShape = {numRows};
  if (width > 0) {
    globalShape.push_back(width);
  }
  DistributedShape shape =
      DistributedShape::onTiles(globalShape, firstDimDistribution);
  return shape;
}

template <DataType Type>
HostTensor HostMatrixBase::decomposeVector(const std::vector<Type> &vector,
                                           bool includeHaloCells,
                                           TypeRef destType,
                                           std::string name) const {
  auto shape = getVectorShape(includeHaloCells);

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

  TileMapping mapping = TileMapping::linearMappingWithShape(shape);
  return HostTensor::createPersistent(std::move(decomposedVector),
                                      std::move(shape), std::move(mapping),
                                      std::move(name));
}
HostTensor HostMatrixBase::loadVectorFromFile(TypeRef type,
                                              std::string fileName,
                                              bool withHalo,
                                              std::string name) const {
  GRAPHENE_TRACEPOINT();
  assert(type->isFloat() && "Only float types are supported");

  return typeSwitch(type, [&]<FloatDataType Type>() -> HostTensor {
    DoubletVector<Type> vector = loadDoubletVectorFromFile<Type>(fileName);

    // Convert the doublet vector to a contiguous vector. This is necessary
    // because a doublet vector is not guaranteed to be sorted by index and
    // may contain missing indices.
    std::vector<Type> contiguousValues(vector.nrows);
    for (size_t i = 0; i < vector.values.size(); i++) {
      contiguousValues[vector.indices[i]] = vector.values[i];
    }

    HostTensor tensor = decomposeVector(contiguousValues, withHalo, type, name);
    return tensor;
  });
}

}  // namespace graphene::matrix::host