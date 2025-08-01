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

#include "libgraphene/matrix/host/DistributedTileLayout.hpp"

#include <algorithm>
#include <cstddef>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Traits.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"

namespace graphene::matrix::host {

double Partitioning::calcImbalance() {
  if (rowToTile.empty()) return 0.0;
  std::map<size_t, size_t> tileRowCount;

  for (size_t tile : rowToTile) {
    tileRowCount[tile]++;
  }

  size_t minRows =
      std::ranges::min_element(tileRowCount, [](const auto &a, const auto &b) {
        return a.second < b.second;
      })->second;
  size_t maxRows =
      std::ranges::max_element(tileRowCount, [](const auto &a, const auto &b) {
        return a.second < b.second;
      })->second;

  for (const auto &[tile, count] : tileRowCount) {
    spdlog::trace("Tile {} has {} rows", tile, count);
  }

  spdlog::trace(
      "Partitioning imbalance: minRows = {}, maxRows = {}, "
      "imbalance = {}",
      minRows, maxRows,
      static_cast<double>(maxRows - minRows) / static_cast<double>(minRows));

  return static_cast<double>(maxRows - minRows) / static_cast<double>(minRows);
}

TilePartition::SeperatorRegion &TilePartition::getSeperatorRegionTo(
    std::set<size_t> dstProcs) {
  assert(!dstProcs.empty());
  // Try to find an existing region
  for (auto &region : seperatorRegions) {
    if (region->dstProcs == dstProcs) return *region;
  }

  // Create a new region
  seperatorRegions.push_back(
      make_unique<SeperatorRegion>(dstProcs, tileId, std::vector<size_t>()));
  return *seperatorRegions.back();
}

TilePartition::HaloRegion &TilePartition::getHaloRegionFrom(
    SeperatorRegion &srcRegion) {
  // Try to find an existing region from the processor that owns the
  // seperator
  for (auto &region : haloRegions) {
    if (&region->srcRegion == &srcRegion) return *region;
  }

  // Create a new region
  haloRegions.push_back(
      make_unique<HaloRegion>(srcRegion, std::vector<size_t>()));
  return *haloRegions.back();
}

void TilePartition::calculateRowMapping() {
  // First, determine the number of rows (interior, seperator,
  // and halo) to reserve enough space
  size_t numRows = interiorRows.size();
  for (auto &seperatorRegion : seperatorRegions) {
    numRows += seperatorRegion->cells.size();
  }
  for (auto &haloRegion : haloRegions) {
    numRows += haloRegion->cells.size();
  }

  // Reserve space for the local to global row mapping
  localToGlobalRow.reserve(numRows);

  // Add the interior rows first
  for (size_t i = 0; i < interiorRows.size(); i++) {
    localToGlobalRow.push_back(interiorRows[i]);
  }
  // Add the seperator rows
  for (auto &seperatorRegion : seperatorRegions) {
    for (size_t i = 0; i < seperatorRegion->cells.size(); i++) {
      localToGlobalRow.push_back(seperatorRegion->cells[i]);
    }
  }
  // Add the halo rows
  for (auto &haloRegion : haloRegions) {
    for (size_t i = 0; i < haloRegion->cells.size(); i++) {
      localToGlobalRow.push_back(haloRegion->cells[i]);
    }
  }

  // Create the global to local row mapping
  for (size_t i = 0; i < localToGlobalRow.size(); i++) {
    globalToLocalRow[localToGlobalRow[i]] = i;
  }
}

//
// DistributedTileLayout implementation
//

DistributedShape DistributedTileLayout::getVectorShape(bool withHalo,
                                                       size_t width) const {
  FirstDimDistribution firstDimDistribution;
  firstDimDistribution.reserve(numTiles());
  size_t numRows = 0;
  for (size_t i = 0; i < numTiles(); ++i) {
    const TilePartition &tile = getTilePartition(i);
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
HostTensor DistributedTileLayout::decomposeVector(
    const std::vector<Type> &vector, bool includeHaloCells, TypeRef destType,
    std::string name, size_t width) const {
  assert(width == 0 || width == 1 && "Only 1D vectors are supported yet");
  auto shape = getVectorShape(includeHaloCells, width);

  // Decompose the vector and cast it to the destination type
  return typeSwitch(destType, [&]<DataType DestType>() {
    std::vector<DestType> decomposedVector;
    decomposedVector.reserve(shape[0]);
    for (size_t i = 0; i < numTiles(); ++i) {
      const TilePartition &tile = getTilePartition(i);
      for (size_t localRow = 0; localRow < tile.localToGlobalRow.size();
           ++localRow) {
        size_t globalRow = tile.localToGlobalRow[localRow];
        if (!includeHaloCells && tile.isHalo(localRow)) continue;
        // CHANGEME: Set halo cells to zero to check if halo exchange is working
        decomposedVector.push_back((DestType)vector[globalRow]);
      }
    }

    TileMapping mapping = TileMapping::linearMappingWithShape(shape);
    return HostTensor::createPersistent(std::move(decomposedVector),
                                        std::move(shape), std::move(mapping),
                                        std::move(name));
  });
}

HostTensor DistributedTileLayout::loadVectorFromFile(TypeRef type,
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

// Template instantiations for common types
template HostTensor DistributedTileLayout::decomposeVector<float>(
    const std::vector<float> &vector, bool includeHaloCells, TypeRef destType,
    std::string name, size_t width) const;

template HostTensor DistributedTileLayout::decomposeVector<double>(
    const std::vector<double> &vector, bool includeHaloCells, TypeRef destType,
    std::string name, size_t width) const;

template HostTensor DistributedTileLayout::decomposeVector<twofloat::two<float>>(
    const std::vector<twofloat::two<float>> &vector, bool includeHaloCells, TypeRef destType,
    std::string name, size_t width) const;

}  // namespace graphene::matrix::host
