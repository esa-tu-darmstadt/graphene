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

#include <any>
#include <memory>
#include <poplar/DebugContext.hpp>
#include <variant>

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/host/DistributedTileLayout.hpp"

namespace graphene::matrix {

namespace host {
class HostMatrix;
class HostMatrixBase;

}  // namespace host
namespace solver {
class Configuration;
}

/** Represents a matrix.
 * This class supports different matrix storage formats, such as the sparse CRS
 * format. The matrix format is hidden behind a concept interface.
 */
class Matrix {
 private:
  std::unique_ptr<MatrixBase> pimpl_;

 public:
  Matrix() = delete;

  /** Constructs a sparse matrix in CRS format */
  Matrix(crs::CRSMatrix storage)
      : pimpl_(std::make_unique<crs::CRSMatrix>(std::move(storage))) {}

  template <typename T = MatrixBase>
  const T &getImpl() const {
    return static_cast<T &>(*pimpl_);
  }

  /**
   * Solves a linear system of equations.
   *
   * This function solves a linear system of equations represented by the
   * equation Ax = b, where A is a matrix, x is the solution vector, and b is
   * the right-hand side vector.
   *
   * @param x The solution vector, which will be modified by this function.
   * @param b The right-hand side vector.
   * @param destType The type of the result vector. If not specified, the type
   * of x is used.
   * @param intermediateType The type of intermediate results. If not specified,
   * \p destType is used.
   * @param config The configuration for the solver.
   */
  void solve(Tensor &x, Tensor &b,
             std::shared_ptr<solver::Configuration> &config);

  /** Computes the residual $ b - Ax $ */
  Tensor residual(Tensor &x, const Tensor &b, TypeRef destType = nullptr,
                  TypeRef intermediateType = nullptr,
                  bool withHalo = false) const {
    if (!destType) destType = x.type();
    if (!intermediateType) intermediateType = destType;
    return pimpl_->residual(x, b, destType, intermediateType, withHalo);
  }

  /** Sparse matrix vector multiplication */
  Tensor operator*(Tensor &x) const {
    return pimpl_->spmv(x, x.type(), x.type());
  }

  /** Returns a view of the vector without the halo cells. Both values have the
   * same underlying tensor, i.e., modifying one will modify the other. */
  Tensor stripHaloCellsFromVector(const Tensor &x) const {
    return pimpl_->stripHaloCellsFromVector(x);
  }

  /**
   * Creates a \ref Value object with the correct shape and tile mapping for a
   * vector that is compatible with the matrix. The vector is uninitialized.
   *
   * @param withHalo A boolean indicating whether to include halo elements in
   * the vector.
   * @return An uninitialized vector of type VectorType.
   */
  Tensor createUninitializedVector(TypeRef type, bool withHalo) const {
    return pimpl_->createUninitializedVector(type, withHalo);
  }

  /** Returns the specified norm of the given vector. In constrast to \ref
   * Expression::norm, this strips the halo cells away, which would otherwise be
   * counted multiple times by the reduce operation. */
  Tensor vectorNorm(VectorNorm norm, const Tensor &x) const {
    return pimpl_->vectorNorm(norm, x);
  }

  /** Returns true if the shape if the given vector is compatible to the matrix,
   * i.e. the product $ A*x $ can be calculated. Optionally, checks that the
   * tile mapping of the vector is as expected. */
  bool isVectorCompatible(const Tensor &value, bool withHalo) const {
    return pimpl_->isVectorCompatible(value, withHalo);
  }

  /**
   * Returns a constant reference to the underlying host matrix.
   */
  const host::DistributedTileLayout &tileLayout() const {
    return pimpl_->tileLayout;
  }

  /**
   * Returns the number of tiles the matrix is distributed over.
   */
  size_t numTiles() const { return pimpl_->numTiles(); }

  /**
   * Retrieves the format of the matrix.
   */
  MatrixFormat getFormat() const { return pimpl_->getFormat(); }
};

}  // namespace graphene::matrix