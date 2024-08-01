#pragma once

#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
namespace graphene::matrix::host {
/** Represents a matrix in host memory. */
template <DataType Type>
class HostMatrix {
  std::shared_ptr<HostMatrixBase<Type>> pimpl;

 public:
  HostMatrix() = default;

  template <typename Impl>
  HostMatrix(std::shared_ptr<Impl> impl) : pimpl(std::move(impl)) {}

  template <typename T = HostMatrixBase<Type>>
  const T &getImpl() const {
    return static_cast<T &>(*pimpl);
  }

  bool multicolorRecommended() const { return pimpl->multicolorRecommended(); }

  const TileLayout &getTileLayout(size_t proci) const {
    return pimpl->getTileLayout(proci);
  }

  matrix::Matrix<Type> copyToTile() const { return pimpl->copyToTile(); }

  std::tuple<poplar::Graph::TileToTensorMapping, std::vector<size_t>>
  getVectorTileMappingAndShape(bool withHalo = false) const {
    return pimpl->getVectorTileMappingAndShape(withHalo);
  }

  MatrixFormat getFormat() const { return pimpl->getFormat(); }
  size_t numTiles() const { return pimpl->numTiles(); }

  HostValue<Type> loadVectorFromFile(std::string fileName,
                                     bool withHalo = false,
                                     std::string name = "vector") const {
    return pimpl->loadVectorFromFile(fileName, withHalo, name);
  }

  HostValue<Type> decomposeVector(const std::vector<Type> &vector,
                                  bool includeHaloCells,
                                  std::string name = "vector") const {
    return pimpl->decomposeVector(vector, includeHaloCells, name);
  }
};

template <DataType Type>
HostMatrix<Type> loadMatrixFromFile(std::filesystem::path path, size_t numTiles,
                                    MatrixFormat format = MatrixFormat::CRS,
                                    std::string name = "matrix");

template <DataType Type>
HostValue<Type> loadVectorFromFile(std::filesystem::path path,
                                   const HostMatrix<Type> &matrix,
                                   bool withHalo = false,
                                   std::string name = "vector");

}  // namespace graphene::matrix::host