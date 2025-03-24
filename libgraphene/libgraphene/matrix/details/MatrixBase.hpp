/*
 * Graphene Linear Algebra Framework for Intelligence Processing Units.
 * Copyright (C) 2025 Embedded Systems and Applications, TU Darmstadt.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"

namespace graphene::matrix {

namespace host {
class HostMatrixBase;
}

/**
 * @brief A base class representing a matrix.
 *
 * @tparam Type The data type of the elements stored in the matrix.
 */
struct MatrixBase {
  host::HostMatrix hostMatrix;  ///< The host matrix.

  /**
   * @brief Construct a new MatrixBase object.
   *
   * @param hostMatrix The host matrix to be used.
   */
  MatrixBase(std::shared_ptr<host::HostMatrixBase> hostMatrix)
      : hostMatrix(std::move(hostMatrix)) {}

  /**
   * @brief Virtual destructor for MatrixBase.
   */
  virtual ~MatrixBase() = default;

  /// -----------------------------------------------------------------------
  /// Format dependent operations
  /// -----------------------------------------------------------------------

  /**
   * @brief Multiply the CRS matrix by a vector.
   *
   * @param x The vector to be multiplied.
   * @param destType The data type of the result vector.
   * @param intermediateType The data type of intermediate computations.
   * @param withHalo Whether to include halo elements in the computation.
   * @return Tensor The result of the multiplication.
   */
  virtual Tensor spmv(Tensor &x, TypeRef destType, TypeRef intermediateType,
                      bool withHalo = false) const = 0;

  /**
   * @brief Compute the residual of the matrix equation.
   *
   * @param x The solution vector.
   * @param b The right-hand side vector.
   * @param destType The data type of the result vector.
    * @param intermediateType The data type of intermediate computations.
   * @param withHalo Whether to include halo elements in the computation.

   * @return Tensor The residual vector.
   */
  virtual Tensor residual(Tensor &x, const Tensor &b, TypeRef destType,
                          TypeRef intermediateType,
                          bool withHalo = false) const = 0;

  /// -----------------------------------------------------------------------
  /// Format independent operations
  /// -----------------------------------------------------------------------

  /**
   * @brief Exchange halo cells of the given value.
   *
   * @param value The value whose halo cells are to be exchanged.
   */
  void exchangeHaloCells(Tensor &value) const;

  /**
   * @brief Check if the shape and potentially the tile mapping of the vector is
   * compatible with the matrix.
   *
   * @param value The vector to be checked.
   * @param withHalo Whether to include halo elements in the check.
   * @return true if the vector is compatible, false otherwise.
   */
  bool isVectorCompatible(const Tensor &value, bool withHalo) const;

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
   * @param x The vector from which to strip halo cells.
   * @return Tensor The vector without halo cells.
   */
  Tensor stripHaloCellsFromVector(const Tensor &x) const;

  /**
   * @brief Returns the specified norm of the given value. In contrast to \ref
   * Value::norm, this strips the halo cells away, which would otherwise be
   * counted multiple times.
   *
   * @param norm The type of norm to compute.
   * @param x The vector whose norm is to be computed.
   * @return Tensor The computed norm.
   */
  Tensor vectorNorm(VectorNorm norm, const Tensor &x) const;

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
   * @param type The data type of the vector.
   * @param withHalo Whether to include halo elements in the vector.
   * @return Tensor<VectorType> The uninitialized vector.
   */
  Tensor createUninitializedVector(TypeRef type, bool withHalo) const;
};

}  // namespace graphene::matrix