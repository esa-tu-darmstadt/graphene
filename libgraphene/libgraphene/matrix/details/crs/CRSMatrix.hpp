#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/crs/CRSAddressing.hpp"
#include "libgraphene/matrix/host/details/HostMatrixBase.hpp"

namespace graphene::matrix::crs {

/**
 * @brief A class representing a Compressed Row Storage (CRS) matrix.
 *
 * @tparam Type The data type of the elements stored in the CRSMatrix.
 */
template <DataType Type>
struct CRSMatrix : public MatrixBase<Type> {
  std::shared_ptr<CRSAddressing>
      addressing;  ///< Addressing scheme for the CRS matrix.
  Value<Type>
      offDiagonalCoefficients;  ///< Off-diagonal coefficients of the matrix.
  Value<Type> diagonalCoefficients;  ///< Diagonal coefficients of the matrix.

  /**
   * @brief Construct a new CRSMatrix object.
   *
   * @param hostMatrix The host matrix this CRS matrix is based on.
   * @param addressing The addressing scheme for the CRS matrix.
   * @param offDiagonalCoefficients The off-diagonal coefficients of the matrix.
   * @param diagonalCoefficients The diagonal coefficients of the matrix.
   */
  CRSMatrix(std::shared_ptr<host::HostMatrixBase<Type>> hostMatrix,
            std::shared_ptr<CRSAddressing> addressing,
            Value<Type> offDiagonalCoefficients,
            Value<Type> diagonalCoefficients)
      : MatrixBase<Type>(std::move(hostMatrix)),
        addressing(std::move(addressing)),
        offDiagonalCoefficients(std::move(offDiagonalCoefficients)),
        diagonalCoefficients(std::move(diagonalCoefficients)) {}

  /**
   * @brief Multiply the CRS matrix by a vector.
   *
   * @param x The vector to be multiplied.
   * @return Value<Type> The result of the multiplication.
   */
  Value<Type> operator*(Value<Type> &x) const override;

  /**
   * @brief Compute the residual of the matrix equation.
   *
   * @param x The solution vector.
   * @param b The right-hand side vector.
   * @param withHalo Whether to include halo elements in the computation.
   * @return Value<Type> The residual vector.
   */
  Value<Type> residual(Value<Type> &x, const Value<Type> &b,
                       bool withHalo = false) const override;

  /**
   * @brief Compute the mixed-precision residual.
   *
   * @param x The solution vector in double precision.
   * @param b The right-hand side vector with float precision.
   * @return Value<float> The residual vector with float precision.
   */

  Value<float> residual(Value<double> &x, const Value<float> &b) const override;

  /**
   * @brief Compute the mixed-precision residual.
   *
   * @param x The solution vector in double word arithmetic.
   * @param b The right-hand side vector with float precision.
   * @return Value<float> The residual vector with float precision.
   */
  Value<float> residual(Value<doubleword> &x,
                        const Value<float> &b) const override;
};

}  // namespace graphene::matrix::crs