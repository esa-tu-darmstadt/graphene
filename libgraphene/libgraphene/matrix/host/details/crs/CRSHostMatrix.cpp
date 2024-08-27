#include "libgraphene/matrix/host/details/crs/CRSHostMatrix.hpp"

#include <metis.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <fast_matrix_market/app/triplet.hpp>
#include <fstream>
#include <poplar/Graph.hpp>
#include <stdexcept>

#include "libgraphene/dsl/HostTensor.hpp"
#include "libgraphene/dsl/HostTensorVariant.hpp"
#include "libgraphene/dsl/RemoteTensor.hpp"
#include "libgraphene/dsl/RemoteTensorVariant.hpp"
#include "libgraphene/dsl/Tensor.hpp"
#include "libgraphene/dsl/TensorVariant.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSAddressing.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"
#include "libgraphene/util/Tracepoint.hpp"

template <graphene::DataType Type>
static id_t getEdgeWeight(Type edgeCoeff, Type maxEdgeCoeff,
                          int maxEdgeWeight) {
  Type absLinearInterp = abs(edgeCoeff) * maxEdgeWeight / maxEdgeCoeff;
  return std::max<int>(1,
                       absLinearInterp);  // Each edge weight must be at least 1
}

// Determines the number of bytes needed to store an unsigned integer value,
// rounded up to the next power of two.
constexpr size_t getUnsignedIntegerWidthForValue(size_t value) {
  if (value == 0) return 0;
  size_t width = 8;  // in bits
  while ((1UL << width) < value) {
    width *= 2;
  }
  return width / 8;
}

template <graphene::DataType ConcreteType, graphene::DataType... Types>
void constructHostValueForType(graphene::HostValueVariant<Types...> &hostValue,
                               std::vector<std::vector<size_t>> data,
                               std::string name) {
  size_t numTiles = data.size();
  // Determine the shape and tile mapping
  poplar::Graph::TileToTensorMapping tileMapping(numTiles);

  size_t numValues = 0;
  for (size_t tileID = 0; tileID < numTiles; ++tileID) {
    tileMapping[tileID].emplace_back(numValues,
                                     numValues + data[tileID].size());
    numValues += data[tileID].size();
  }
  std::vector<size_t> shape = {numValues};

  // Combine the data to a single vector with the concrete data type
  std::vector<ConcreteType> concreteData;
  concreteData.reserve(numValues);
  for (auto &tile : data) {
    std::copy(tile.begin(), tile.end(), std::back_inserter(concreteData));
  }
  std::string typeName =
      (graphene::Traits<ConcreteType>::PoplarType).toString();
  spdlog::trace(
      "Constructing host value for {} of type {} for {} tiles with {} datums.",
      name, typeName, numTiles, numValues);
  hostValue = graphene::HostTensor<ConcreteType>(
      std::move(concreteData), std::move(shape), std::move(tileMapping), name);
}

template <graphene::DataType... Types>
void constructHostValue(graphene::HostValueVariant<Types...> &hostValue,
                        std::vector<std::vector<size_t>> data,
                        std::string name) {
  // Determine the required data type
  size_t maxValue = 0;
  for (auto &tile : data) {
    auto maxTileValue = std::max_element(tile.begin(), tile.end());
    if (maxTileValue != tile.end()) {
      maxValue = std::max(maxValue, *maxTileValue);
    }
  }
  size_t width = getUnsignedIntegerWidthForValue(maxValue);

  switch (width) {
    case 0:
    case 1:
      constructHostValueForType<uint8_t>(hostValue, data, name);
      break;
    case 2:
      constructHostValueForType<uint16_t>(hostValue, data, name);
      break;
    case 4:
      constructHostValueForType<uint32_t>(hostValue, data, name);
      break;
    default:
      throw std::runtime_error(fmt::format(
          "Due to the size of the matrix, the CRS addressing requires {} bits, "
          "but the IPU only support 8, 16, and 32 bit integers.",
          width * 32));
  }
}

namespace graphene::matrix::host::crs {
template <DataType Type>
CRSHostMatrix<Type>::CRSHostMatrix(TripletMatrix<Type> tripletMatrix,
                                   size_t numTiles, std::string name)
    : HostMatrixBase<Type>(numTiles, name) {
  sortTripletMatrx(tripletMatrix);

  globalMatrix_ = convertToCRS(std::move(tripletMatrix));

  this->partitioning_ =
      std::move(calculatePartitioning(numTiles, globalMatrix_));
  this->tileLayout_ =
      std::move(calculateTileLayouts(this->partitioning_, globalMatrix_));

  calculateLocalAddressings();

  calculateRowColors();
  calculateColorAddressings();

  decomposeValues();
}

template <DataType Type>
CRSHostMatrix<Type>::CRSMatrix CRSHostMatrix<Type>::convertToCRS(
    TripletMatrix<Type> tripletMatrix) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Converting COO matrix to CRS");

  CRSMatrix crs;

  // Copy the diagonal values to the CRS matrix
  crs.diagValues.resize(tripletMatrix.nrows);
  for (size_t i = 0; i < tripletMatrix.entries.size(); i++) {
    if (tripletMatrix.entries[i].row == tripletMatrix.entries[i].col) {
      crs.diagValues[tripletMatrix.entries[i].row] =
          tripletMatrix.entries[i].val;
    }
  }

  crs.addressing.rowPtr.reserve(tripletMatrix.nrows + 1);
  crs.addressing.colInd.reserve(tripletMatrix.entries.size());
  crs.offDiagValues.reserve(tripletMatrix.nrows);

  crs.addressing.rowPtr.push_back(0);

  size_t lastRow = 0;

  for (size_t i = 0; i < tripletMatrix.entries.size(); i++) {
    // Ignore diagonal entries
    if (tripletMatrix.entries[i].row == tripletMatrix.entries[i].col) {
      continue;
    }

    // Increase the row pointer if we have moved to the next row
    while (lastRow != tripletMatrix.entries[i].row) {
      crs.addressing.rowPtr.push_back(crs.addressing.colInd.size());
      lastRow++;
    }

    crs.addressing.colInd.push_back(tripletMatrix.entries[i].col);
    crs.offDiagValues.push_back(tripletMatrix.entries[i].val);
  }

  // Add the last row pointers. The loop is required if the matrix ends with
  // rows with no off-diagonal values
  while (crs.addressing.rowPtr.size() <= tripletMatrix.nrows)
    crs.addressing.rowPtr.push_back(crs.addressing.colInd.size());

  assert(crs.addressing.rowPtr.size() == tripletMatrix.nrows + 1);
  assert(crs.diagValues.size() == crs.addressing.rowPtr.size() - 1);
  return crs;
}

template <DataType Type>
Partitioning CRSHostMatrix<Type>::calculatePartitioning(size_t numTiles,
                                                        const CRSMatrix &crs) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Distributing matrix to {} processors", numTiles);

  Partitioning partitioning;

  // number of vertices
  idx_t nvtxs = (idx_t)crs.addressing.rowPtr.size() - 1;

  // number of processors
  idx_t nparts = (idx_t)numTiles;
  if (nparts == 1) {
    spdlog::warn("Only one processor, skipping matrix partitioning");
    partitioning.rowToTile.resize(nvtxs, 0);
    return partitioning;
  }

  // number of balancing constraints
  idx_t ncon = 1;

  std::vector<idx_t> xadj;  // rowPtr
  xadj.reserve(crs.addressing.rowPtr.size());
  std::vector<idx_t> adjncy;  // colInd
  adjncy.reserve(crs.addressing.colInd.size());
  std::vector<idx_t> edgeWeights;  // edge weights
  edgeWeights.reserve(crs.offDiagValues.size());
  std::vector<idx_t> vertexWeights;  // vertex weights
  vertexWeights.reserve(crs.addressing.rowPtr.size() - 1);

  // Find the maximum edge coefficient so that we can scale the edge
  // weights to integers
  float maxEdgeCoeff = *std::max_element(
      crs.offDiagValues.begin(), crs.offDiagValues.end(),
      [](float a, float b) { return std::abs(a) < std::abs(b); });
  maxEdgeCoeff = abs(maxEdgeCoeff);
  const int maxEdgeWeight = 1000;

  // We want METIS to balance two constraints:
  // The storage for the off-diagonal values and the storage for the
  // diagonal values. We assume that the storage for the diagonal values

  // Construct the adjacency list and edge weights
  for (size_t i = 0; i < crs.addressing.rowPtr.size() - 1; i++) {
    size_t numEdges = crs.addressing.rowPtr[i + 1] - crs.addressing.rowPtr[i];
    xadj.push_back(adjncy.size());
    vertexWeights.push_back(numEdges + 1);
    for (size_t j = crs.addressing.rowPtr[i]; j < crs.addressing.rowPtr[i + 1];
         j++) {
      adjncy.push_back(crs.addressing.colInd[j]);
      edgeWeights.push_back(
          getEdgeWeight(crs.offDiagValues[j], maxEdgeCoeff, maxEdgeWeight));
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

  // Copy partitioning to the output
  partitioning.rowToTile.reserve(nvtxs);
  std::copy(part.begin(), part.end(),
            std::back_inserter(partitioning.rowToTile));

  return partitioning;
}

template <DataType Type>
std::vector<TileLayout> CRSHostMatrix<Type>::calculateTileLayouts(
    const Partitioning &partitioning, const CRSMatrix &matrix) {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Calculating regions for each tile in parallel");

  // Get the number of tiles and rows
  size_t numTiles = *std::max_element(partitioning.rowToTile.begin(),
                                      partitioning.rowToTile.end()) +
                    1;
  size_t numRows = matrix.addressing.rowPtr.size() - 1;

  // Initialize the regions for each tile
  std::vector<TileLayout> tiles;
  tiles.reserve(numTiles);
  for (size_t i = 0; i < numTiles; i++) {
    tiles.emplace_back(i);
  }

  // A lambda that returns the tiles that own rows adjacent to the
  // given row
  auto neighbourTilesOfRow = [&](size_t row) {
    std::set<size_t> neighbours;
    size_t ownTileId = partitioning.rowToTile[row];
    for (size_t i = matrix.addressing.rowPtr[row];
         i < matrix.addressing.rowPtr[row + 1]; i++) {
      if (partitioning.rowToTile[matrix.addressing.colInd[i]] != ownTileId)
        neighbours.insert(partitioning.rowToTile[matrix.addressing.colInd[i]]);
    }
    return neighbours;
  };

  // Mutexes for accessing the halo regions of the tiles. This is needed
  // because due to the parallel execution of the loop below, multiple threads
  // might try to add halo regions to the same tile at the same time.
  std::vector<std::mutex> haloMutexes(numTiles);

  // Determine the regions for each tile in parallel
  std::for_each(
      std::execution::par, tiles.begin(), tiles.end(), [&](TileLayout &tile) {
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
                [&](TileLayout &tile) { tile.calculateRowMapping(); });

  return tiles;
}

template <DataType Type>
matrix::Matrix<Type> CRSHostMatrix<Type>::copyToTile() {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Copying matrix to tiles");

  // Get the number of tiles and rows
  size_t numTiles = this->tileLayout_.size();
  size_t numRows = this->partitioning_.rowToTile.size();

  auto offDiagonalValues = offDiagValues_.copyToRemote().copyToTile();
  auto diagValues = diagValues_.copyToRemote().copyToTile();
  auto rowPtr = rowPtr_.copyToRemote().copyToTile();
  auto colInd = colInd_.copyToRemote().copyToTile();

  Coloring coloring(colorSortAddr.copyToRemote().copyToTile(),
                    colorSortStartPtr.copyToRemote().copyToTile());

  auto crsAddressing = std::make_shared<matrix::crs::CRSAddressing>(
      std::move(rowPtr), std::move(colInd), std::move(coloring));

  return matrix::Matrix<Type>(matrix::crs::CRSMatrix<Type>(
      this->shared_from_this(), std::move(crsAddressing),
      std::move(offDiagonalValues), std::move(diagValues)));
}

template <DataType Type>
void CRSHostMatrix<Type>::calculateLocalAddressings() {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Calculating local addressings in parallel");
  size_t numTiles = this->tileLayout_.size();

  localAddressings_.resize(numTiles);

  // For each tile, stores their local row pointers and column indices
  std::vector<std::vector<size_t>> rowPtrs(numTiles);
  std::vector<std::vector<size_t>> colInds(numTiles);

  // Calculate the local addressings
  std::for_each(
      std::execution::par, this->tileLayout_.begin(), this->tileLayout_.end(),
      [&](TileLayout &tile) {
        // Calculate rowPtr and colInd for this tile
        auto &rowPtr = rowPtrs[tile.tileId];
        auto &colInd = colInds[tile.tileId];
        for (size_t localRow = 0; localRow < tile.localToGlobalRow.size();
             ++localRow) {
          // Do not include halo rows in the local addressing
          if (tile.isHalo(localRow)) continue;

          rowPtr.push_back(colInd.size());

          // The columns must be sorted. They are not automatically
          // sorted even if the global matrix is sorted!
          // For this, first collect the columns in a set and then
          // insert them (sorted) into the colInd vector

          size_t globalRow = tile.localToGlobalRow[localRow];
          size_t globalStart = globalMatrix_.addressing.rowPtr[globalRow];
          size_t globalEnd = globalMatrix_.addressing.rowPtr[globalRow + 1];
          std::set<size_t> cols;
          for (size_t i = globalMatrix_.addressing.rowPtr[globalRow];
               i < globalMatrix_.addressing.rowPtr[globalRow + 1]; ++i) {
            size_t globalCol = globalMatrix_.addressing.colInd[i];
            size_t localCol = tile.globalToLocalRow[globalCol];
            cols.insert(localCol);
          }
          // Insert the sorted columns into the colInd vector
          colInd.insert(colInd.end(), cols.begin(), cols.end());
        }
        rowPtr.push_back(colInd.size());

        // It is possible that the column indices is empty if a tile has only
        // isolated rows
        // In this case, add a dummy "0" to the colInd vector to avoid an
        // empty vector
        if (colInd.empty()) {
          colInd.push_back(0);
        }

        localAddressings_[tile.tileId].rowPtr = rowPtr;
        localAddressings_[tile.tileId].colInd = colInd;
      });

  spdlog::trace("Constructing host values");
  constructHostValue(rowPtr_, std::move(rowPtrs), this->name_ + "_rowPtr");
  constructHostValue(colInd_, std::move(colInds), this->name_ + "_colInd");
}
template <DataType Type>
void CRSHostMatrix<Type>::decomposeValues() {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Decomposing matrix values");
  colInd_.get<uint32_t>({1});
  // Decompose diagonal values. This can be done with decomposeVector because
  // there is exactly one value per row.
  diagValues_ = this->decomposeVector(globalMatrix_.diagValues, false);
  offDiagValues_ = decomposeOffDiagCoefficients(globalMatrix_.offDiagValues);
}

template <DataType Type>
HostTensor<Type> CRSHostMatrix<Type>::decomposeOffDiagCoefficients(
    const std::vector<Type> &values) const {
  std::vector<Type> decomposedValues;
  decomposedValues.reserve(colInd_.numElements());

  poplar::Graph::TileToTensorMapping mapping(this->tileLayout_.size());

  // Decompose and create the tile mapping
  size_t numElements = 0;
  for (auto &tile : this->tileLayout_) {
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

      std::set<std::pair<size_t, Type>> offDiagValues;
      for (size_t i = globalMatrix_.addressing.rowPtr[globalRow];
           i < globalMatrix_.addressing.rowPtr[globalRow + 1]; ++i) {
        size_t globalCol = globalMatrix_.addressing.colInd[i];
        if (globalCol == globalRow) continue;
        size_t localCol = tile.globalToLocalRow.at(globalCol);
        offDiagValues.insert({localCol, globalMatrix_.offDiagValues[i]});
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
    mapping[tile.tileId].emplace_back(numElements,
                                      numElements + numElementsOnThisTile);
    numElements += numElementsOnThisTile;
  }

  std::vector<size_t> shape = {numElements};
  return HostTensor<Type>(std::move(decomposedValues), std::move(shape),
                          std::move(mapping), this->name_ + "_offDiag");
}

template <DataType Type>
void CRSHostMatrix<Type>::calculateRowColors() {
  assert(!localAddressings_.empty());
  rowColors_.resize(this->numTiles_);
  numColors_.resize(this->numTiles_, 0);

  spdlog::info("Calculating row colors in parallel");

  // Calculate the color of each row
  std::for_each(
      std::execution::par, this->tileLayout_.begin(), this->tileLayout_.end(),
      [&](TileLayout &tile) {
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

template <DataType Type>
void CRSHostMatrix<Type>::calculateColorAddressings() {
  GRAPHENE_TRACEPOINT();
  spdlog::info("Calculating color addressings in parallel");

  std::vector<std::vector<size_t>> colorSortAddr(this->numTiles_);
  std::vector<std::vector<size_t>> colorSortStartPtr(this->numTiles_);

  // Calculate the color of each row
  std::for_each(
      std::execution::par, this->tileLayout_.begin(), this->tileLayout_.end(),
      [&](TileLayout &tile) {
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
  for (size_t tileID = 0; tileID < this->numTiles_; ++tileID) {
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

  constructHostValue(this->colorSortAddr, std::move(colorSortAddr),
                     this->name_ + "_colorSortAddr");
  constructHostValue(this->colorSortStartPtr, std::move(colorSortStartPtr),
                     this->name_ + "_colorSortStartPtr");
}

// Template instantiations
template class CRSHostMatrix<float>;
}  // namespace graphene::matrix::host::crs