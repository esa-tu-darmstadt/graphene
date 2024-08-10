#pragma once

#include "libgraphene/dsl/Value.hpp"
#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"

namespace graphene::matrix {

namespace host {
template <DataType Type>
class HostMatrixBase;
}

/**
 * @brief A base class representing a matrix.
 *
 * @tparam Type The data type of the elements stored in the matrix.
 */
template <DataType Type>
struct MatrixBase {
  host::HostMatrix<Type> hostMatrix;  ///< The host matrix.

  /**
   * @brief Construct a new MatrixBase object.
   *
   * @param hostMatrix The host matrix to be used.
   */
  MatrixBase(std::shared_ptr<host::HostMatrixBase<Type>> hostMatrix)
      : hostMatrix(std::move(hostMatrix)) {}

  /**
   * @brief Virtual destructor for MatrixBase.
   */
  virtual ~MatrixBase() = default;

  /**
   * @brief Compute the residual of the matrix equation.
   *
   * @param x The solution vector.
   * @param b The right-hand side vector.
   * @param withHalo Whether to include halo elements in the computation.
   * @return Value<Type> The residual vector.
   */
  virtual Value<Type> residual(Value<Type> &x, const Value<Type> &b,
                               bool withHalo = false) const = 0;

  /**
   * @brief Compute the residual of the matrix equation with mixed precision.
   *
   * @param x The solution vector with double precision.
   * @param b The right-hand side vector with float precision.
   * @return Value<float> The residual vector with float precision.
   */
  virtual Value<float> residual(Value<double> &x,
                                const Value<float> &b) const = 0;

  /**
   * @brief Multiply the matrix by a vector.
   *
   * @param x The vector to be multiplied.
   * @return Value<Type> The result of the multiplication.
   */
  virtual Value<Type> operator*(Value<Type> &x) const = 0;

  /**
   * @brief Exchange halo cells of the given value.
   *
   * @tparam VectorType The type of the vector elements.
   * @param value The value whose halo cells are to be exchanged.
   */
  template <typename VectorType>
  void exchangeHaloCells(Value<VectorType> &value) const;

  /**
   * @brief Check if the shape and potentially the tile mapping of the vector is
   * compatible with the matrix.
   *
   * @tparam VectorType The type of the vector elements.
   * @param value The vector to be checked.
   * @param withHalo Whether to include halo elements in the check.
   * @param tileMappingMustMatch Whether the tile mapping must match. If the
   * tile mapping of a vector and the matrix does not match, poplar will
   * automatically rearrange the data.
   * @return true if the vector is compatible, false otherwise.
   */
  template <typename VectorType>
  bool isVectorCompatible(const Value<VectorType> &value, bool withHalo,
                          bool tileMappingMustMatch = true) const;

  /**
   * @brief Get the number of tiles in the matrix.
   *
   * @return size_t The number of tiles.
   */
  size_t numTiles() const { return hostMatrix.numTiles(); }

  /**
   * @brief Returns a view of the vector without the halo cells. Both values
   * have the same underlying tensor, i.e., modifying one will modify the other.
   *
   * @tparam VectorType The type of the vector elements.
   * @param x The vector from which to strip halo cells.
   * @return Value<VectorType> The vector without halo cells.
   */
  template <typename VectorType>
  Value<VectorType> stripHaloCellsFromVector(const Value<VectorType> &x) const;

  /**
   * @brief Returns the specified norm of the given value. In contrast to \ref
   * Value::norm, this strips the halo cells away, which would otherwise be
   * counted multiple times.
   *
   * @tparam VectorType The type of the vector elements.
   * @param norm The type of norm to compute.
   * @param x The vector whose norm is to be computed.
   * @return Value<VectorType> The computed norm.
   */
  template <typename VectorType>
  Value<VectorType> vectorNorm(VectorNorm norm,
                               const Value<VectorType> &x) const;

  /**
   * @brief Get the format of the matrix.
   *
   * @return MatrixFormat The format of the matrix.
   */
  MatrixFormat getFormat() const { return hostMatrix.getFormat(); }

  /**
   * @brief Create an uninitialized vector with the correct shape and tile
   * mapping for a vector that is compatible with the matrix.
   *
   * @tparam VectorType The type of the vector elements.
   * @param withHalo Whether to include halo elements in the vector.
   * @return Value<VectorType> The uninitialized vector.
   */
  template <typename VectorType>
  Value<VectorType> createUninitializedVector(bool withHalo) const;
};

}  // namespace graphene::matrix