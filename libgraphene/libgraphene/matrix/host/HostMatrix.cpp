#include "libgraphene/matrix/host/HostMatrix.hpp"

#include <filesystem>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/host/details/Poisson.hpp"
#include "libgraphene/matrix/host/details/crs/CRSHostMatrix.hpp"
namespace graphene::matrix::host {

matrix::Matrix HostMatrix::copyToTile() const { return pimpl->copyToTile(); }

template <DataType Type>
HostMatrix::HostMatrix(MatrixFormat format, TripletMatrix<Type> tripletMatrix,
                       size_t numTiles, std::string name) {
  switch (format) {
    case MatrixFormat::CRS: {
      auto model = Runtime::instance().createResource<crs::CRSHostMatrix>(
          std::move(tripletMatrix), numTiles, name);
      pimpl = std::move(model);
      break;
    }
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

HostMatrix loadMatrixFromFile(TypeRef type, std::filesystem::path path,
                              size_t numTiles, MatrixFormat format,
                              std::string name) {
  assert(type->isFloat() && "Only floating point types are supported");
  return typeSwitch(type, [&]<FloatDataType T>() {
    TripletMatrix<T> tripletMatrix = loadTripletMatrixFromFile<T>(path);
    return HostMatrix(format, std::move(tripletMatrix), numTiles, name);
  });
}

HostMatrix generate3DPoissonMatrix(TypeRef type, size_t nx, size_t ny,
                                   size_t nz, size_t numTiles,
                                   MatrixFormat format, std::string name) {
  assert(type->isFloat() && "Only floating point types are supported");
  return typeSwitch(type, [&]<FloatDataType T>() {
    TripletMatrix<T> tripletMatrix =
        generate3DPoissonTripletMatrix<T>(nx, ny, nz);
    return HostMatrix(format, std::move(tripletMatrix), numTiles, name);
  });
}

// Template instantiations

}  // namespace graphene::matrix::host