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

#include "libgraphene/matrix/host/details/crs/CRSHostMatrix.hpp"

#include <metis.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <fast_matrix_market/app/triplet.hpp>
#include <fstream>
#include <poplar/Graph.hpp>
#include <ranges>
#include <stdexcept>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSAddressing.hpp"
#include "libgraphene/matrix/host/DistributedTileLayout.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/host/details/CoordinateFormat.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"
#include "libgraphene/util/Tracepoint.hpp"
#include "libtwofloat/operators.hpp"

using namespace graphene;

namespace {
template <DataType Type>
static id_t getEdgeWeight(Type edgeCoeff, Type maxEdgeCoeff,
                          int maxEdgeWeight) {
  int absLinearInterp =
      (int)(abs(edgeCoeff) * Type(maxEdgeWeight) / Type(maxEdgeCoeff));
  return std::max<int>(1,
                       absLinearInterp);  // Each edge weight must be at least 1
}

}  // namespace

namespace graphene::matrix::host::crs {
template <FloatDataType Type>
CRSHostMatrix::CRSHostMatrix(TripletMatrix<Type> tripletMatrix, size_t numTiles,
                             std::string name)
    : HostMatrixBase(numTiles, name) {
  sortTripletMatrx(tripletMatrix);

  auto [globalAddressing, globalMatrixValues] =
      convertToCRS(std::move(tripletMatrix));

  this->globalAddressing_ = std::move(globalAddressing);
  this->partitioning_ = std::move(
      calculatePartitioning(numTiles_, globalMatrixValues, globalAddressing_));

  this->tilePartitions_ =
      std::move(calculateTileLayouts(this->partitioning_, globalAddressing_));
  this->localAddressings_ =
      calculateLocalAddressings(globalAddressing_, this->tilePartitions_);

  calculateRowColors();
  calculateColorAddressings();

  decomposeValues(globalMatrixValues);
}

template <FloatDataType Type>
std::tuple<CRSHostMatrix::CRSAddressing, CRSHostMatrix::CRSMatrixValues<Type>>
CRSHostMatrix::convertToCRS(TripletMatrix<Type> tripletMatrix) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Converting COO matrix to CRS");

  CRSMatrixValues<Type> crsValues;
  CRSAddressing crsAddressing;

  // Copy the diagonal values to the CRS matrix
  crsValues.diagValues.resize(tripletMatrix.nrows);
  for (size_t i = 0; i < tripletMatrix.entries.size(); i++) {
    if (tripletMatrix.entries[i].row == tripletMatrix.entries[i].col) {
      crsValues.diagValues[tripletMatrix.entries[i].row] =
          tripletMatrix.entries[i].val;
    }
  }

  crsAddressing.rowPtr.reserve(tripletMatrix.nrows + 1);
  crsAddressing.colInd.reserve(tripletMatrix.entries.size());
  crsValues.offDiagValues.reserve(tripletMatrix.nrows);

  crsAddressing.rowPtr.push_back(0);

  size_t lastRow = 0;

  for (size_t i = 0; i < tripletMatrix.entries.size(); i++) {
    // Ignore diagonal entries
    if (tripletMatrix.entries[i].row == tripletMatrix.entries[i].col) {
      continue;
    }

    // Increase the row pointer if we have moved to the next row
    while (lastRow != tripletMatrix.entries[i].row) {
      crsAddressing.rowPtr.push_back(crsAddressing.colInd.size());
      lastRow++;
    }

    crsAddressing.colInd.push_back(tripletMatrix.entries[i].col);
    crsValues.offDiagValues.push_back(tripletMatrix.entries[i].val);
  }

  // Add the last row pointers. The loop is required if the matrix ends with
  // rows with no off-diagonal values
  while (crsAddressing.rowPtr.size() <= tripletMatrix.nrows)
    crsAddressing.rowPtr.push_back(crsAddressing.colInd.size());

  assert(crsAddressing.rowPtr.size() == tripletMatrix.nrows + 1);
  assert(crsValues.diagValues.size() == crsAddressing.rowPtr.size() - 1);
  return {crsAddressing, crsValues};
}

template <FloatDataType Type>
Partitioning CRSHostMatrix::calculatePartitioning(
    size_t &numTiles, const CRSMatrixValues<Type> &matrixValues,
    const CRSAddressing &addressing) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Distributing matrix to {} processors", numTiles);

  // number of vertices
  idx_t nvtxs = (idx_t)addressing.rowPtr.size() - 1;

  // number of processors
  idx_t nparts = (idx_t)numTiles;
  if (nparts == 1) {
    spdlog::warn("Only one processor, skipping matrix partitioning");
    Partitioning partitioning;
    partitioning.numTiles = 1;
    numTiles = 1;
    partitioning.rowToTile.resize(nvtxs, 0);
    return partitioning;
  }

  // number of balancing constraints
  idx_t ncon = 1;

  std::vector<idx_t> xadj;  // rowPtr
  xadj.reserve(addressing.rowPtr.size());
  std::vector<idx_t> adjncy;  // colInd
  adjncy.reserve(addressing.colInd.size());
  std::vector<idx_t> edgeWeights;  // edge weights
  edgeWeights.reserve(matrixValues.offDiagValues.size());
  std::vector<idx_t> vertexWeights;  // vertex weights
  vertexWeights.reserve(addressing.rowPtr.size() - 1);

  // Find the maximum edge coefficient so that we can scale the edge
  // weights to integers
  Type maxEdgeCoeff = *std::max_element(
      matrixValues.offDiagValues.begin(), matrixValues.offDiagValues.end(),
      [](auto a, auto b) { return std::abs(a) < std::abs(b); });
  maxEdgeCoeff = abs(maxEdgeCoeff);
  const int maxEdgeWeight = 1000;

  // We want METIS to balance two constraints:
  // The storage for the off-diagonal values and the storage for the
  // diagonal values. We assume that the storage for the diagonal values

  // Construct the adjacency list and edge weights
  for (size_t i = 0; i < addressing.rowPtr.size() - 1; i++) {
    size_t numEdges = addressing.rowPtr[i + 1] - addressing.rowPtr[i];
    xadj.push_back(adjncy.size());
    vertexWeights.push_back(numEdges + 1);
    for (size_t j = addressing.rowPtr[i]; j < addressing.rowPtr[i + 1]; j++) {
      adjncy.push_back(addressing.colInd[j]);
      edgeWeights.push_back(getEdgeWeight(matrixValues.offDiagValues[j],
                                          maxEdgeCoeff, maxEdgeWeight));
    }
  }
  xadj.push_back(adjncy.size());

  //   if (spdlog::get_level() <= spdlog::level::trace)
  //     dumpEdgeWeightStatistics(edgeWeights);

  // is filled with the total communication volume
  idx_t edgecut;

  // The partitioning of the vertices
  std::vector<idx_t> part(nvtxs);

  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_NUMBERING] = 0;                // C-style numbering
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // Minimize edgecut

  // METIS_PartGraphKway may yield inplausible results, for example processors
  // that are assigned no vertices. METIS_PartGraphRecursive is more robust in
  // this case.
  if (int error =
          METIS_PartGraphRecursive(
              &nvtxs,                // num vertices in graph
              &ncon,                 // num balancing constraints
              xadj.data(),           // indexing into adjncy
              adjncy.data(),         // neighbour info
              vertexWeights.data(),  // vertexWeights.data(),  // vertex wts
              nullptr,               // vsize: total communication vol
              nullptr,               // edgeWeights.data(),  // edge wts
              &nparts,               // nParts
              nullptr,               // tpwgts
              nullptr,               // ubvec: processor imbalance (default)
              options, &edgecut, part.data()) != METIS_OK) {
    throw std::runtime_error(
        fmt::format("METIS error {} during "
                    "partitioning",
                    error));
  }

  spdlog::trace("Total edgecut after partitioning: {}", edgecut);

  /// When the number of requested tiles is large compared to the matrix, some
  /// tiles might end up empty. We need to shift the partitioning to remove
  /// these empty tiles.
  std::vector<size_t> verticesPerProc(numTiles, 0);
  for (size_t i = 0; i < nvtxs; i++) {
    verticesPerProc[part[i]]++;
  }

  size_t numEmptyProcs = 0;

  // For each proc, the mapping from the original proc id to the shifted proc
  // id
  std::vector<ssize_t> procToShiftedProcMapping(numTiles);
  for (size_t i = 0; i < numTiles; i++) {
    if (verticesPerProc[i] == 0) {
      numEmptyProcs++;
      procToShiftedProcMapping[i] = -1;
      continue;
    }
    procToShiftedProcMapping[i] = i - numEmptyProcs;
  }
  numTiles = numTiles - numEmptyProcs;

  // Copy partitioning to the output, accounting for empty processors
  Partitioning partitioning;
  partitioning.numTiles = numTiles;
  partitioning.rowToTile.reserve(nvtxs);
  for (size_t i = 0; i < nvtxs; i++) {
    partitioning.rowToTile.push_back(procToShiftedProcMapping[part[i]]);
  }
  spdlog::info(
      "Tried to partition matrix to {} procs. Resulted in {} non-empty "
      "partitions.",
      numTiles + numEmptyProcs, numTiles);

  return partitioning;
}

std::vector<TilePartition> CRSHostMatrix::calculateTileLayouts(
    const Partitioning &partitioning, const CRSAddressing &addressing) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Calculating regions for each tile in parallel");

  // Get the number of tiles and rows
  size_t numTiles = *std::max_element(partitioning.rowToTile.begin(),
                                      partitioning.rowToTile.end()) +
                    1;
  size_t numRows = addressing.rowPtr.size() - 1;

  // Initialize the regions for each tile
  std::vector<TilePartition> tiles;
  tiles.reserve(numTiles);
  for (size_t i = 0; i < numTiles; i++) {
    tiles.emplace_back(i);
  }

  // A lambda that returns the tiles that own rows adjacent to the
  // given row
  auto neighbourTilesOfRow = [&](size_t row) {
    std::set<size_t> neighbours;
    size_t ownTileId = partitioning.rowToTile[row];
    for (size_t i = addressing.rowPtr[row]; i < addressing.rowPtr[row + 1];
         i++) {
      if (partitioning.rowToTile[addressing.colInd[i]] != ownTileId)
        neighbours.insert(partitioning.rowToTile[addressing.colInd[i]]);
    }
    return neighbours;
  };

  // Mutexes for accessing the halo regions of the tiles. This is needed
  // because due to the parallel execution of the loop below, multiple threads
  // might try to add halo regions to the same tile at the same time.
  std::vector<std::mutex> haloMutexes(numTiles);

  // Determine the regions for each tile in parallel
  std::for_each(
      std::execution::par, tiles.begin(), tiles.end(),
      [&](TilePartition &tile) {
        // Find all cells assigned to this tile and
        // categorize them into interior and separator cells
        for (size_t row = 0; row < numRows; ++row) {
          // Check whether this cell is assigned to this processor
          if (partitioning.rowToTile[row] == tile.tileId) {
            auto neighbourTileIDs = neighbourTilesOfRow(row);
            if (neighbourTileIDs.empty()) {
              // This is an interior cell, meaning it is not needed
              // on any other processor
              tile.interiorRows.push_back(row);
            } else {
              // This is a seperator cell
              // Add it to its seperator region
              auto &ownSeperatorRegion =
                  tile.getSeperatorRegionTo(neighbourTileIDs);
              ownSeperatorRegion.cells.push_back(row);

              // Add it to the halo regions of the neighbour
              // processors
              for (auto neighbourTileID : neighbourTileIDs) {
                // Lock the neighbour processor's halo mutex because
                // other threads might also try to add cells to the
                // same halo region
                std::unique_lock<std::mutex> lock(haloMutexes[neighbourTileID]);

                auto &foreignHaloRegion =
                    tiles[neighbourTileID].getHaloRegionFrom(
                        ownSeperatorRegion);
                foreignHaloRegion.cells.push_back(row);
              }
            }
          }
        }
      });

  // Now that all regions are set, determine the local row to global row
  // mapping
  std::for_each(std::execution::par, tiles.begin(), tiles.end(),
                [&](TilePartition &tile) { tile.calculateRowMapping(); });

  return tiles;
}

matrix::Matrix CRSHostMatrix::copyToTile() {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Copying matrix to tiles");

  // Get the number of tiles and rows
  size_t numTiles = this->numTiles();
  size_t numRows = this->partitioning_.rowToTile.size();

  auto offDiagonalValues = offDiagValues_.copyToRemote().copyToTile();
  auto diagValues = diagValues_.copyToRemote().copyToTile();
  auto rowPtr = rowPtr_.copyToRemote().copyToTile();
  auto colInd = colInd_.copyToRemote().copyToTile();

  Coloring coloring(colorSortAddr.copyToRemote().copyToTile(),
                    colorSortStartPtr.copyToRemote().copyToTile());

  auto crsAddressing = std::make_shared<matrix::crs::CRSAddressing>(
      std::move(rowPtr), std::move(colInd), std::move(coloring));

  return matrix::Matrix(matrix::crs::CRSMatrix(
      this->shared_from_this(), std::move(crsAddressing),
      std::move(offDiagonalValues), std::move(diagValues)));
}
std::vector<CRSHostMatrix::CRSAddressing>
CRSHostMatrix::calculateLocalAddressings(
    const CRSAddressing &globalAddressing,
    const std::vector<TilePartition> &tilePartitions) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Calculating local addressings in parallel");
  size_t numTiles = tilePartitions.size();

  // Holds the decomposed rowPtr and colInd for each tile
  std::vector<CRSHostMatrix::CRSAddressing> localAddressings(numTiles);

  // Calculate the local addressings
  std::for_each(std::execution::par, tilePartitions.begin(),
                tilePartitions.end(), [&](const TilePartition &tile) {
                  // Calculate rowPtr and colInd for this tile
                  auto &rowPtr = localAddressings[tile.tileId].rowPtr;
                  auto &colInd = localAddressings[tile.tileId].colInd;
                  for (size_t localRow = 0;
                       localRow < tile.localToGlobalRow.size(); ++localRow) {
                    // Do not include halo rows in the local addressing
                    if (tile.isHalo(localRow)) continue;

                    rowPtr.push_back(colInd.size());

                    // The columns must be sorted. They are not automatically
                    // sorted even if the global matrix is sorted!
                    // For this, first collect the columns in a set and then
                    // insert them (sorted) into the colInd vector

                    size_t globalRow = tile.localToGlobalRow[localRow];
                    size_t globalStart = globalAddressing.rowPtr[globalRow];
                    size_t globalEnd = globalAddressing.rowPtr[globalRow + 1];
                    std::set<size_t> cols;
                    for (size_t i = globalAddressing.rowPtr[globalRow];
                         i < globalAddressing.rowPtr[globalRow + 1]; ++i) {
                      size_t globalCol = globalAddressing.colInd[i];
                      size_t localCol = tile.globalToLocalRow.at(globalCol);
                      cols.insert(localCol);
                    }
                    // Insert the sorted columns into the colInd vector
                    colInd.insert(colInd.end(), cols.begin(), cols.end());
                  }
                  rowPtr.push_back(colInd.size());

                  // It is possible that the column indices is empty if a tile
                  // has only isolated rows In this case, add a dummy "0" to the
                  // colInd vector to avoid an empty vector
                  if (colInd.empty()) {
                    colInd.push_back(0);
                  }
                });

  return localAddressings;
}
template <FloatDataType Type>
void CRSHostMatrix::decomposeValues(const CRSMatrixValues<Type> &globalMatrix) {
  GRAPHENE_TRACEPOINT();

  // Decompose diagonal coefficients. This can be done with decomposeVector
  // because there is exactly one value per row.
  spdlog::trace("Decomposing diagonal coefficients");
  diagValues_ = this->decomposeVector(globalMatrix.diagValues, false);

  // Decompose off-diagonal coefficients
  spdlog::trace("Decomposing off-diagonal coefficients");
  offDiagValues_ = decomposeOffDiagCoefficients(globalMatrix.offDiagValues);

  // Decompose rowPtr
  spdlog::trace("Decomposing rowPtrs");
  auto decomposedRowPtrs = localAddressings_ | std::ranges::views::transform(
                                                   [](const auto &addressing) {
                                                     return addressing.rowPtr;
                                                   });
  rowPtr_ = constructSmallestIntegerHostValue(std::move(decomposedRowPtrs),
                                              this->name() + "_rowPtr");

  // Decompose colInd
  spdlog::trace("Decomposing colInds");
  auto decomposedColInds = localAddressings_ | std::ranges::views::transform(
                                                   [](const auto &addressing) {
                                                     return addressing.colInd;
                                                   });
  colInd_ = constructSmallestIntegerHostValue(std::move(decomposedColInds),
                                              this->name() + "_colInd");
}

template <FloatDataType Type>
HostTensor CRSHostMatrix::decomposeOffDiagCoefficients(
    const std::vector<Type> &values, TypeRef targetType) const {
  return typeSwitch(targetType, [&]<FloatDataType TargetType>() -> HostTensor {
    std::vector<TargetType> decomposedValues;
    decomposedValues.reserve(colInd_.numElements());

    FirstDimDistribution firstDimDistribution;
    firstDimDistribution.reserve(this->numTiles());

    // Decompose and create the tile mapping
    size_t numElements = 0;
    for (auto &tile : this->tilePartitions_) {
      size_t numElementsOnThisTile = 0;
      const CRSAddressing &localAddressing = localAddressings_[tile.tileId];
      // Iterate over the rows of this tile
      for (size_t localRow = 0; localRow < tile.localToGlobalRow.size();
           ++localRow) {
        if (tile.isHalo(localRow)) continue;
        size_t globalRow = tile.localToGlobalRow[localRow];

        // Remember that the columns of the local matrix are sorted, so the
        // values must be sorted by column as well.

        // For this, collect all off-diagonal values of this row as a pair of
        // its column and value. Then sort them by column and insert them into
        // the decomposedValues vector.

        std::set<std::pair<size_t, TargetType>> offDiagValues;
        for (size_t i = globalAddressing_.rowPtr[globalRow];
             i < globalAddressing_.rowPtr[globalRow + 1]; ++i) {
          size_t globalCol = globalAddressing_.colInd[i];
          if (globalCol == globalRow) continue;
          size_t localCol = tile.globalToLocalRow.at(globalCol);
          TargetType castedValue = static_cast<TargetType>(values[i]);
          offDiagValues.insert({localCol, castedValue});
        }

        for (const auto &pair : offDiagValues) {
          decomposedValues.push_back(pair.second);
          numElementsOnThisTile++;
        }
      }
      // Add a dummy "0" if the tile has no off-diagonal values
      if (numElementsOnThisTile == 0) {
        decomposedValues.push_back(0);
        numElementsOnThisTile++;
      }
      firstDimDistribution[tile.tileId] = numElementsOnThisTile;
      numElements += numElementsOnThisTile;
    }

    TensorShape shape = {numElements};
    DistributedShape distributedShape =
        DistributedShape::onTiles(shape, firstDimDistribution);
    TileMapping mapping = TileMapping::linearMappingWithShape(distributedShape);
    return HostTensor::createPersistent(
        std::move(decomposedValues), std::move(distributedShape),
        std::move(mapping), this->name() + "_offDiag");
  });
}

void CRSHostMatrix::calculateRowColors() {
  assert(!localAddressings_.empty());
  rowColors_.resize(this->numTiles());
  numColors_.resize(this->numTiles(), 0);

  spdlog::info("Calculating row colors in parallel");

  // Calculate the color of each row
  std::for_each(
      std::execution::par, this->tilePartitions_.begin(),
      this->tilePartitions_.end(), [&](TilePartition &tile) {
        size_t numRows = tile.numInteriorRows() + tile.numSeperatorRows();
        auto &tileColors = rowColors_[tile.tileId];
        auto &localAddressing = localAddressings_[tile.tileId];
        tileColors.resize(numRows, std::numeric_limits<size_t>::max());

        // A set of rows that are free to be calculated
        std::queue<size_t> freeRowsInCurrentColor;

        // A set of rows that just got calculated and are now free to
        // be calculated in the next color
        std::queue<size_t> freeRowsInNextColor;

        // For each row, keep track of the number adjacent rows that
        // have not been calculated yet
        std::vector<size_t> numAdjacentUncalculatedRows(numRows, 0);

        // Initialize the number of adjacent uncalculated rows. We
        // only need to consider the lower triangle.
        for (size_t localRow = 0; localRow < numRows; ++localRow) {
          size_t start = localAddressing.rowPtr[localRow];
          size_t end = localAddressing.rowPtr[localRow + 1];
          for (size_t i = start; i < end; ++i) {
            size_t localCol = localAddressing.colInd[i];

            // Stop when we reach the diagonal, only consider the
            // lower triangle
            if (localCol > localRow) break;
            if (localCol >= numRows) continue;
            ++numAdjacentUncalculatedRows[localRow];
          }
        }

        // Add all rows with no adjacent uncalculated cells to the
        // queue
        for (size_t localRow = 0; localRow < numRows; ++localRow) {
          if (numAdjacentUncalculatedRows[localRow] == 0)
            freeRowsInNextColor.push(localRow);
        }

        size_t currentColor = 0;
        while (!freeRowsInNextColor.empty()) {
          // Move the rows from the next color to the current color
          // and clear the next color
          std::swap(freeRowsInCurrentColor, freeRowsInNextColor);
          freeRowsInNextColor = std::queue<size_t>();

          while (!freeRowsInCurrentColor.empty()) {
            size_t localRow = freeRowsInCurrentColor.front();
            freeRowsInCurrentColor.pop();

            // Assign the current color to this cell
            tileColors[localRow] = currentColor;

            // Now that this cell is calculated, reduce the number of
            // adjacent uncalculated cells of its neighbours that
            // depend on it. Only consider the upper triangle
            ssize_t start = localAddressing.rowPtr[localRow];
            ssize_t end = localAddressing.rowPtr[localRow + 1];
            for (ssize_t i = end - 1; i >= start; --i) {
              size_t localCol = localAddressing.colInd[i];
              // Stop when we reach the diagonal, only consider the
              // upper triangle
              if (localCol < localRow) break;
              if (localCol >= numRows) continue;
              --numAdjacentUncalculatedRows[localCol];

              // If this cell has no more adjacent uncalculated cells,
              // add it to the queue
              if (numAdjacentUncalculatedRows[localCol] == 0)
                freeRowsInNextColor.push(localCol);
            }
          }
          currentColor++;
        }
        numColors_[tile.tileId] = currentColor;
      });
}

void CRSHostMatrix::calculateColorAddressings() {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Calculating color addressings in parallel");

  std::vector<std::vector<size_t>> colorSortAddr(this->numTiles());
  std::vector<std::vector<size_t>> colorSortStartPtr(this->numTiles());

  // Calculate the color of each row
  std::for_each(
      std::execution::par, this->tilePartitions_.begin(),
      this->tilePartitions_.end(), [&](TilePartition &tile) {
        size_t numColors = numColors_[tile.tileId];
        size_t numRows = tile.numSeperatorRows() + tile.numInteriorRows();
        auto &localAddressing = localAddressings_[tile.tileId];
        auto &tileColors = rowColors_[tile.tileId];
        auto &tileColorSortAddr = colorSortAddr[tile.tileId];
        auto &tileColorSortStartPtr = colorSortStartPtr[tile.tileId];

        // Calculate the color sort and its start addresses
        tileColorSortAddr.reserve(numRows);
        tileColorSortStartPtr.reserve(numColors + 1);
        for (size_t color = 0; color < numColors; ++color) {
          tileColorSortStartPtr.push_back(tileColorSortAddr.size());
          for (size_t localCelli = 0; localCelli < numRows; ++localCelli) {
            if (tileColors[localCelli] == color) {
              tileColorSortAddr.push_back(localCelli);
            }
          }
        }
        tileColorSortStartPtr.push_back(tileColorSortAddr.size());
      });

  // Recommend the use of multicolor if at least 90% of the rows are
  // in colors with more than 12 rows
  size_t numParallizableRows = 0;
  size_t numTotalRows = 0;
  for (size_t tileID = 0; tileID < this->numTiles(); ++tileID) {
    for (size_t color = 0; color < numColors_[tileID]; ++color) {
      size_t colorSize = colorSortStartPtr[tileID][color + 1] -
                         colorSortStartPtr[tileID][color];
      numTotalRows += colorSize;
      if (colorSize > 12) numParallizableRows += colorSize;
    }
  }

  spdlog::trace(
      "{} ({:.2f} %) of the rows are in colors with more than 12 rows",
      numParallizableRows, 100.0 * numParallizableRows / numTotalRows);
  if (numParallizableRows > 0.9 * numTotalRows) {
    this->multicolorRecommended_ = true;
    spdlog::info("Multicolor is recommended for this matrix");
  } else {
    spdlog::info("Multicolor is not recommended for this matrix");
  }

  this->colorSortAddr = constructSmallestIntegerHostValue(
      std::move(colorSortAddr), this->name() + "_colorSortAddr");
  this->colorSortStartPtr = constructSmallestIntegerHostValue(
      std::move(colorSortStartPtr), this->name() + "_colorSortStartPtr");
}

// Explicit instantiation of the CRSHostMatrix constructor for the supported
// floating point types
template CRSHostMatrix::CRSHostMatrix(TripletMatrix<float> tripletMatrix,
                                      size_t numTiles, std::string name);
template CRSHostMatrix::CRSHostMatrix(TripletMatrix<double> tripletMatrix,
                                      size_t numTiles, std::string name);
template CRSHostMatrix::CRSHostMatrix(TripletMatrix<doubleword> tripletMatrix,
                                      size_t numTiles, std::string name);

}  // namespace graphene::matrix::host::crs