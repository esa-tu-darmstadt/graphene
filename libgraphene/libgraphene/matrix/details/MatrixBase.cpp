#include "libgraphene/matrix/details/MatrixBase.hpp"

#include <cstddef>
#include <poplar/Interval.hpp>

#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"

namespace graphene::matrix {
template <DataType Type>
template <typename VectorType>
void MatrixBase<Type>::exchangeHaloCells(Value<VectorType> &value) const {
  DebugInfo di("MatrixBase");

  // Get the mapping
  auto [mapping, shape] = hostMatrix.getVectorTileMappingAndShape(true);

  // Make s
  if (!this->isVectorCompatible(value, true, false))
    throw std::runtime_error(
        "Vector x is not compatible with the layout of the matrix");

  for (size_t tile = 0; tile < mapping.size(); ++tile) {
    poplar::Tensor destTileTensor = value.tensorOnTile(tile);
    const host::TileLayout &destLayout = hostMatrix.getTileLayout(tile);

    for (auto &haloRegion : destLayout.haloRegions) {
      poplar::Tensor srcTileTensor = value.tensorOnTile(haloRegion->srcProc());
      const host::TileLayout &srcLayout =
          hostMatrix.getTileLayout(haloRegion->srcProc());

      // Transfer this region from the source to the destination blockwise
      size_t srcLocalStartCelli =
          srcLayout.globalToLocalRow.at(haloRegion->cells.front());
      size_t srcLocalEndCelli =
          srcLayout.globalToLocalRow.at(haloRegion->cells.back());
      size_t destLocalStartCelli =
          destLayout.globalToLocalRow.at(haloRegion->cells.front());
      size_t destLocalEndCelli =
          destLayout.globalToLocalRow.at(haloRegion->cells.back());

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
      Context::program().add(
          poplar::program::Copy(srcRegionTensor, destRegionTensor, false, di));
    }
  }
}

template <DataType Type>
template <typename VectorType>
bool MatrixBase<Type>::isVectorCompatible(const Value<VectorType> &value,
                                          bool withHalo,
                                          bool tileMappingMustMatch) const {
  auto [mapping, shape] = hostMatrix.getVectorTileMappingAndShape(withHalo);
  if (value.shape() != shape) return false;

  if (!tileMappingMustMatch) return true;

  // Check that the tile mapping is the same. Allow for empty mappings on both
  // sides.
  for (size_t tile = 0;
       tile < std::max(value.tileMapping().size(), mapping.size()); ++tile) {
    bool valueHasMappingOnThisTile =
        value.tileMapping().size() > tile && !value.tileMapping()[tile].empty();
    bool matrixHasMappingOnThisTile =
        mapping.size() > tile && !mapping[tile].empty();

    if (valueHasMappingOnThisTile != matrixHasMappingOnThisTile) return false;
  }
  return true;
}

template <DataType Type>
template <typename VectorType>
Value<VectorType> MatrixBase<Type>::stripHaloCellsFromVector(
    const Value<VectorType> &x) const {
  auto [mappingWithoutHalo, shapeWithoutHalo] =
      hostMatrix.getVectorTileMappingAndShape(false);

  // If the shape is already correct, return the input
  if (x.shape() == shapeWithoutHalo) return x;

  auto [mappingWithHalo, shapeWithHalo] =
      hostMatrix.getVectorTileMappingAndShape(true);

  if (x.shape() != shapeWithHalo) {
    throw std::runtime_error(
        "The vector must be compatible with the matrix layout, either with or "
        "without halo cells");
  }

  std::vector<poplar::Tensor> tensors(mappingWithoutHalo.size());
  for (size_t tile = 0; tile < mappingWithoutHalo.size(); ++tile) {
    if (mappingWithoutHalo[tile].empty()) continue;
    std::vector<poplar::Interval> &intervalsOnThisTile =
        mappingWithoutHalo[tile];
    if (intervalsOnThisTile.size() > 1) {
      throw std::runtime_error(
          "stripHaloCellsFromVector does not support multiple intervals on a "
          "tile");
    }
    poplar::Tensor tileTensor = x.tensor();
    tensors[tile] = tileTensor.slice(intervalsOnThisTile[0].begin(),
                                     intervalsOnThisTile[0].end());
  }
  return Value<VectorType>(poplar::concat(tensors));
}

template <DataType Type>
template <typename VectorType>
Value<VectorType> MatrixBase<Type>::vectorNorm(
    VectorNorm norm, const Value<VectorType> &x) const {
  Value<VectorType> stripped = stripHaloCellsFromVector(x);
  return stripped.norm(norm);
}

template <DataType Type>
template <typename VectorType>
Value<VectorType> MatrixBase<Type>::createUninitializedVector(
    bool withHalo) const {
  auto [mapping, shape] = hostMatrix.getVectorTileMappingAndShape(withHalo);
  return Value<VectorType>(shape, mapping);
}

// Template instantiations
#define INSTANTIATE(T)                                                         \
  template class MatrixBase<T>;                                                \
  template void MatrixBase<T>::exchangeHaloCells(Value<T> &value) const;       \
  template bool MatrixBase<T>::isVectorCompatible(                             \
      const Value<T> &value, bool withHalo, bool tileMappingMustMatch) const;  \
  template Value<T> MatrixBase<T>::stripHaloCellsFromVector(const Value<T> &x) \
      const;                                                                   \
  template Value<T> MatrixBase<T>::vectorNorm(VectorNorm norm,                 \
                                              const Value<T> &x) const;        \
  template Value<T> MatrixBase<T>::createUninitializedVector(bool withHalo)    \
      const;

INSTANTIATE(float)

// Additionally, instantiate the template for mixed precision
template void MatrixBase<float>::exchangeHaloCells(Value<double> &value) const;
template bool MatrixBase<float>::isVectorCompatible(
    const Value<double> &value, bool withHalo, bool tileMappingMustMatch) const;
template Value<double> MatrixBase<float>::stripHaloCellsFromVector(
    const Value<double> &x) const;

}  // namespace graphene::matrix