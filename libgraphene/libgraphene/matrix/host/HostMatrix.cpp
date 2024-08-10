#include "libgraphene/matrix/host/HostMatrix.hpp"

#include <filesystem>

#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/details/Poisson.hpp"
#include "libgraphene/matrix/host/details/crs/CRSHostMatrix.hpp"
namespace graphene::matrix::host {
template <DataType Type>
HostMatrix<Type>::HostMatrix(MatrixFormat format,
                             TripletMatrix<Type> tripletMatrix, size_t numTiles,
                             std::string name) {
  switch (format) {
    case MatrixFormat::CRS: {
      auto model = Runtime::instance().createResource<crs::CRSHostMatrix<Type>>(
          std::move(tripletMatrix), numTiles, name);
      pimpl = std::move(model);
      break;
    }
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

template <DataType Type>
HostMatrix<Type> loadMatrixFromFile(std::filesystem::path path, size_t numTiles,
                                    MatrixFormat format, std::string name) {
  TripletMatrix<Type> tripletMatrix = loadTripletMatrixFromFile<Type>(path);
  return HostMatrix<Type>(format, std::move(tripletMatrix), numTiles, name);
}

template <DataType Type>
HostValue<Type> loadVectorFromFile(std::filesystem::path path,
                                   const HostMatrix<Type> &matrix,
                                   bool withHalo, std::string name) {
  return matrix.getImpl().loadVectorFromFile(path, withHalo, name);
}

template <DataType Type>
HostMatrix<Type> generate3DPoissonMatrix(size_t nx, size_t ny, size_t nz,
                                         size_t numTiles, MatrixFormat format,
                                         std::string name) {
  TripletMatrix<Type> tripletMatrix =
      generate3DPoissonTripletMatrix<Type>(nx, ny, nz);
  return HostMatrix<Type>(format, std::move(tripletMatrix), numTiles, name);
}

// Template instantiations
template class HostMatrix<float>;
template HostMatrix<float> loadMatrixFromFile(std::filesystem::path path,
                                              size_t numTiles,
                                              MatrixFormat format,
                                              std::string name);
template HostValue<float> loadVectorFromFile(std::filesystem::path path,
                                             const HostMatrix<float> &matrix,
                                             bool withHalo, std::string name);
template HostMatrix<float> generate3DPoissonMatrix(size_t nx, size_t ny,
                                                   size_t nz, size_t numTiles,
                                                   MatrixFormat format,
                                                   std::string name);
}  // namespace graphene::matrix::host