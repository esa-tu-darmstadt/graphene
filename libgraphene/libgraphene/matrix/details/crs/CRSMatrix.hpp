#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/crs/CRSAddressing.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"

namespace graphene::matrix::crs {

/**
 * @brief A class representing a Compressed Row Storage (CRS) matrix.
 *
 * @tparam Type The data type of the elements stored in the CRSMatrix.
 */
struct CRSMatrix : public MatrixBase {
  std::shared_ptr<CRSAddressing>
      addressing;                  ///< Addressing scheme for the CRS matrix.
  Tensor offDiagonalCoefficients;  ///< Off-diagonal coefficients of the matrix.
  Tensor diagonalCoefficients;     ///< Diagonal coefficients of the matrix.

  /**
   * @brief Construct a new CRSMatrix object.
   *
   * @param hostMatrix The host matrix this CRS matrix is based on.
   * @param addressing The addressing scheme for the CRS matrix.
   * @param offDiagonalCoefficients The off-diagonal coefficients of the matrix.
   * @param diagonalCoefficients The diagonal coefficients of the matrix.
   */
  CRSMatrix(std::shared_ptr<host::HostMatrixBase> hostMatrix,
            std::shared_ptr<CRSAddressing> addressing,
            Tensor offDiagonalCoefficients, Tensor diagonalCoefficients)
      : MatrixBase(std::move(hostMatrix)),
        addressing(std::move(addressing)),
        offDiagonalCoefficients(std::move(offDiagonalCoefficients)),
        diagonalCoefficients(std::move(diagonalCoefficients)) {}

  Tensor spmv(Tensor &x, TypeRef destType = nullptr,
              TypeRef intermediateType = nullptr,
              bool withHalo = false) const override;

  Tensor residual(Tensor &x, const Tensor &b, TypeRef destType = nullptr,
                  TypeRef intermediateType = nullptr,
                  bool withHalo = false) const override;
};

}  // namespace graphene::matrix::crs