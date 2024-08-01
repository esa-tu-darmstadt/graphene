#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/crs/CRSAddressing.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"
namespace graphene::matrix::crs {

template <DataType Type>
struct CRSMatrix : public MatrixBase<Type> {
  std::shared_ptr<CRSAddressing> addressing;
  Value<Type> offDiagonalCoefficients;
  Value<Type> diagonalCoefficients;
  CRSMatrix(std::shared_ptr<host::HostMatrixBase<Type>> hostMatrix,
            std::shared_ptr<CRSAddressing> addressing,
            Value<Type> offDiagonalCoefficients,
            Value<Type> diagonalCoefficients)
      : MatrixBase<Type>(std::move(hostMatrix)),
        addressing(std::move(addressing)),
        offDiagonalCoefficients(std::move(offDiagonalCoefficients)),
        diagonalCoefficients(std::move(diagonalCoefficients)) {}

  Value<Type> operator*(Value<Type> &x) const override;

  Value<Type> residual(Value<Type> &x, const Value<Type> &b,
                       bool withHalo = false) const override;

  Value<float> residual(Value<double> &x, const Value<float> &b) const override;
};

}  // namespace graphene::matrix::crs