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

#include "libgraphene/matrix/details/MatrixBase.hpp"

#include <cstddef>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <stdexcept>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/DistributedTileLayout.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"

namespace graphene::matrix {
void MatrixBase::exchangeHaloCells(Tensor &value) const {
  DebugInfo di("MatrixBase");

  struct ExchangeHaloCellsMetadata {
    poplar::Function function;
  };

  // Check if an exchange program has already been generated
  if (value.template hasMetadata<ExchangeHaloCellsMetadata>()) {
    spdlog::trace("Using cached exchange program");
    auto metadata = value.template getMetadata<ExchangeHaloCellsMetadata>();
    Context::program().add(poplar::program::Call(metadata.function, di));
    return;
  }

  // Make sure that the vector is compatible with the matrix layout
  if (!this->isVectorCompatible(value, true))
    throw std::runtime_error(
        "Vector x is not compatible with the layout of the matrix");

  std::vector<poplar::Tensor> srcTensors;
  std::vector<poplar::Tensor> destTensors;

  for (size_t tile = 0; tile < tileLayout.numTiles(); ++tile) {
    poplar::Tensor destTileTensor = value.tensorOnTile(tile);
    const host::TilePartition &destPartition =
        tileLayout.getTilePartition(tile);
    for (auto &haloRegion : destPartition.haloRegions) {
      poplar::Tensor srcTileTensor = value.tensorOnTile(haloRegion->srcProc());
      const host::TilePartition &srcPartition =
          tileLayout.getTilePartition(haloRegion->srcProc());

      // Transfer this region from the source to the destination blockwise
      size_t srcLocalStartCelli =
          srcPartition.globalToLocalRow.at(haloRegion->cells.front());
      size_t srcLocalEndCelli =
          srcPartition.globalToLocalRow.at(haloRegion->cells.back());
      size_t destLocalStartCelli =
          destPartition.globalToLocalRow.at(haloRegion->cells.front());
      size_t destLocalEndCelli =
          destPartition.globalToLocalRow.at(haloRegion->cells.back());

      // Make sure that the cells are in the correct order for a blockwise copy
      // This validates the algorithm proposed in our paper.
      // This is not a good place to put this check though
      //   if (haloRegion->srcRegion.dstProcs.size() != 1) {
      //     for (size_t i = 0; i < haloRegion->cells.size(); ++i) {
      //       gcelli_t globalCelli = haloRegion->cells[i];
      //       celli_t srcLocalCelli = srcHostMeshTile.localCelli(globalCelli);
      //       celli_t destLocalCelli =
      //       destHostMeshTile.localCelli(globalCelli); assert(srcLocalCelli ==
      //       srcLocalStartCelli + i); assert(destLocalCelli ==
      //       destLocalStartCelli + i);
      //     }
      //   }

      poplar::Tensor srcRegionTensor =
          srcTileTensor.slice(srcLocalStartCelli, srcLocalEndCelli + 1, 0);
      poplar::Tensor destRegionTensor =
          destTileTensor.slice(destLocalStartCelli, destLocalEndCelli + 1, 0);

      srcTensors.push_back(srcRegionTensor);
      destTensors.push_back(destRegionTensor);
    }
  }
  // Create a function for the exchange program and add it to the
  // program
  poplar::Tensor srcTensor = poplar::concat(srcTensors);
  poplar::Tensor destTensor = poplar::concat(destTensors);
  poplar::program::Program program =
      poplar::program::Copy(srcTensor, destTensor, false, di);
  poplar::Function function = Context::graph().addFunction(program);
  value.setMetadata(ExchangeHaloCellsMetadata{function});
  Context::program().add(poplar::program::Call(function, di));
  spdlog::trace("Creating new exchange program");
}

bool MatrixBase::isVectorCompatible(const Tensor &value, bool withHalo) const {
  DistributedShape shape = tileLayout.getVectorShape(withHalo);
  TileMapping linearMapping = TileMapping::linearMappingWithShape(shape);

  if (value.shape().firstDimDistribution() != shape.firstDimDistribution())
    return false;

  // We expect a linear mapping for the vector
  if (linearMapping != value.tileMapping()) return false;

  return true;
}

Tensor MatrixBase::stripHaloCellsFromVector(const Tensor &x) const {
  DistributedShape shapeWithoutHalo = tileLayout.getVectorShape(false);

  // If the shape is already correct, return the input
  if (x.shape() == shapeWithoutHalo) return x.same();

  DistributedShape shapeWithHalo = tileLayout.getVectorShape(true);

  if (x.shape() != shapeWithHalo) {
    throw std::invalid_argument(
        "The vector must be compatible with the matrix layout, either with or "
        "without halo cells");
  }

  std::vector<poplar::Tensor> tensors;
  tensors.reserve(numTiles());

  for (auto [tile, rowsWithoutHalo] : shapeWithoutHalo.firstDimDistribution()) {
    poplar::Tensor tileTensor = x.tensorOnTile(tile);
    tensors.push_back(tileTensor.slice(0, rowsWithoutHalo));
  }

  return Tensor::fromPoplar(poplar::concat(tensors), x.type());
}

Tensor MatrixBase::vectorNorm(VectorNorm norm, const Tensor &x) const {
  Tensor stripped = stripHaloCellsFromVector(x);
  return stripped.norm(norm);
}

Tensor MatrixBase::createUninitializedVector(TypeRef type,
                                             bool withHalo) const {
  DistributedShape shape = tileLayout.getVectorShape(withHalo);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);
  return Tensor::uninitialized(type, shape, mapping, "vector");
}

}  // namespace graphene::matrix