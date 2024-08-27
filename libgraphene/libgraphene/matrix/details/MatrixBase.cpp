#include "libgraphene/matrix/details/MatrixBase.hpp"

#include <cstddef>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>

#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/TileLayout.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"

namespace graphene::matrix {
template <DataType Type>
template <typename VectorType>
void MatrixBase<Type>::exchangeHaloCells(Tensor<VectorType> &value) const {
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

  // Get the mapping
  auto [mapping, shape] = hostMatrix.getVectorTileMappingAndShape(true);

  // Make sure that the vector is compatible with the matrix layout
  if (!this->isVectorCompatible(value, true, false))
    throw std::runtime_error(
        "Vector x is not compatible with the layout of the matrix");

  std::vector<poplar::Tensor> srcTensors;
  std::vector<poplar::Tensor> destTensors;

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

template <DataType Type>
template <typename VectorType>
bool MatrixBase<Type>::isVectorCompatible(const Tensor<VectorType> &value,
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
Tensor<VectorType> MatrixBase<Type>::stripHaloCellsFromVector(
    const Tensor<VectorType> &x) const {
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
  return Tensor<VectorType>(poplar::concat(tensors));
}

template <DataType Type>
template <typename VectorType>
Tensor<VectorType> MatrixBase<Type>::vectorNorm(
    VectorNorm norm, const Tensor<VectorType> &x) const {
  Tensor<VectorType> stripped = stripHaloCellsFromVector(x);
  return stripped.norm(norm);
}

template <DataType Type>
template <typename VectorType>
Tensor<VectorType> MatrixBase<Type>::createUninitializedVector(
    bool withHalo) const {
  auto [mapping, shape] = hostMatrix.getVectorTileMappingAndShape(withHalo);
  return Tensor<VectorType>(shape, mapping);
}

// Template instantiations
#define INSTANTIATE(T)                                                         \
  template class MatrixBase<T>;                                                \
  template void MatrixBase<T>::exchangeHaloCells(Tensor<T> &value) const;      \
  template bool MatrixBase<T>::isVectorCompatible(                             \
      const Tensor<T> &value, bool withHalo, bool tileMappingMustMatch) const; \
  template Tensor<T> MatrixBase<T>::stripHaloCellsFromVector(                  \
      const Tensor<T> &x) const;                                               \
  template Tensor<T> MatrixBase<T>::vectorNorm(VectorNorm norm,                \
                                               const Tensor<T> &x) const;      \
  template Tensor<T> MatrixBase<T>::createUninitializedVector(bool withHalo)   \
      const;

INSTANTIATE(float)

// Additionally, instantiate the template for mixed precision
template void MatrixBase<float>::exchangeHaloCells(
    Tensor<doubleword> &value) const;
template bool MatrixBase<float>::isVectorCompatible(
    const Tensor<doubleword> &value, bool withHalo,
    bool tileMappingMustMatch) const;
template Tensor<doubleword> MatrixBase<float>::stripHaloCellsFromVector(
    const Tensor<doubleword> &x) const;
// Additionally, instantiate the template for mixed precision
template void MatrixBase<float>::exchangeHaloCells(Tensor<double> &value) const;
template bool MatrixBase<float>::isVectorCompatible(
    const Tensor<double> &value, bool withHalo,
    bool tileMappingMustMatch) const;
template Tensor<double> MatrixBase<float>::stripHaloCellsFromVector(
    const Tensor<double> &x) const;

}  // namespace graphene::matrix