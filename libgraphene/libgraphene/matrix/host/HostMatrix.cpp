#include "libgraphene/matrix/host/HostMatrix.hpp"

#include <filesystem>

#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/details/crs/CRSHostMatrix.hpp"

namespace graphene::matrix::host {
template <DataType Type>
HostMatrix<Type> loadMatrixFromFile(std::filesystem::path path, size_t numTiles,
                                    MatrixFormat format, std::string name) {
  switch (format) {
    case MatrixFormat::CRS: {
      auto model = Runtime::instance().createResource<crs::CRSHostMatrix<Type>>(
          path, numTiles, name);
      return HostMatrix<Type>(std::move(model));
    }
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

template <DataType Type>
HostValue<Type> loadVectorFromFile(std::filesystem::path path,
                                   const HostMatrix<Type> &matrix,
                                   bool withHalo, std::string name) {
  return matrix.getImpl().loadVectorFromFile(path, withHalo, name);
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
}  // namespace graphene::matrix::host