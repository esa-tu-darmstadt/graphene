#pragma once

#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"
namespace graphene::matrix::host {
/** Represents a matrix in host memory. */
template <DataType Type>
class HostMatrix {
  std::shared_ptr<HostMatrixBase<Type>> pimpl;

 public:
  HostMatrix() = default;

  HostMatrix(MatrixFormat format, TripletMatrix<Type> tripletMatrix,
             size_t numTiles, std::string name = "matrix");

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

  HostTensor<Type> loadVectorFromFile(std::string fileName,
                                      bool withHalo = false,
                                      std::string name = "vector") const {
    return pimpl->loadVectorFromFile(fileName, withHalo, name);
  }

  HostTensor<Type> decomposeVector(const std::vector<Type> &vector,
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
HostTensor<Type> loadVectorFromFile(std::filesystem::path path,
                                    const HostMatrix<Type> &matrix,
                                    bool withHalo = false,
                                    std::string name = "vector");

template <DataType Type>
HostMatrix<Type> generate3DPoissonMatrix(
    size_t nx, size_t ny, size_t nz, size_t numTiles,
    MatrixFormat format = MatrixFormat::CRS, std::string name = "poisson");

}  // namespace graphene::matrix::host