#pragma once

#include <any>
#include <memory>
#include <poplar/DebugContext.hpp>
#include <variant>

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"

namespace graphene::matrix {

namespace host {
template <DataType Type>
class HostMatrix;
template <DataType Type>
class HostMatrixBase;

}  // namespace host
namespace solver {
class Configuration;
}

/** Represents a matrix.
 * This class supports different matrix storage formats, such as the sparse CRS
 * format. The matrix format is hidden behind a concept interface.
 */
template <DataType Type>
class Matrix {
 private:
  std::unique_ptr<MatrixBase<Type>> pimpl_;

 public:
  Matrix() = delete;

  /** Constructs a sparse matrix in CRS format */
  Matrix(crs::CRSMatrix<Type> storage)
      : pimpl_(std::make_unique<crs::CRSMatrix<Type>>(std::move(storage))) {}

  template <typename T = MatrixBase<Type>>
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
   * @param config The configuration for the solver.
   */
  void solve(Tensor<Type> &x, Tensor<Type> &b,
             std::shared_ptr<solver::Configuration> &config);

  /** Computes the residual $ b - Ax $ */
  Tensor<Type> residual(Tensor<Type> &x, const Tensor<Type> &b) const {
    return pimpl_->residual(x, b);
  }

  /** Computes the mixed-precision residual. */
  Tensor<float> residual(Tensor<doubleword> &x, const Tensor<float> &b) const
    requires std::is_same_v<Type, float>
  {
    return pimpl_->residual(x, b);
  }

  /** Computes the mixed-precision residual. */
  Tensor<float> residual(Tensor<double> &x, const Tensor<float> &b) const
    requires std::is_same_v<Type, float>
  {
    return pimpl_->residual(x, b);
  }

  /** Sparse matrix vector multiplication */
  Tensor<Type> operator*(Tensor<Type> &x) const { return pimpl_->operator*(x); }

  /** Returns a view of the vector without the halo cells. Both values have the
   * same underlying tensor, i.e., modifying one will modify the other. */
  template <typename VectorType>
  Tensor<VectorType> stripHaloCellsFromVector(
      const Tensor<VectorType> &x) const {
    return pimpl_->stripHaloCellsFromVector(x);
  }

  template <typename VectorType>
  /**
   * Creates a \ref Value object with the correct shape and tile mapping for a
   * vector that is compatible with the matrix. The vector is uninitialized.
   *
   * @param withHalo A boolean indicating whether to include halo elements in
   * the vector.
   * @return An uninitialized vector of type VectorType.
   */
  Tensor<VectorType> createUninitializedVector(bool withHalo) const {
    return pimpl_->template createUninitializedVector<VectorType>(withHalo);
  }

  /** Returns the specified norm of the given vector. In constrast to \ref
   * Value::norm, this strips the halo cells away, which would otherwise be
   * counted multiple times by the reduce operation. */
  template <typename VectorType>
  Tensor<VectorType> vectorNorm(VectorNorm norm,
                                const Tensor<VectorType> &x) const {
    return pimpl_->vectorNorm(norm, x);
  }

  /** Returns true if the shape if the given vector is compatible to the matrix,
   * i.e. the product $ A*x $ can be calculated. Optionally, checks that the
   * tile mapping of the vector is as expected. */
  bool isVectorCompatible(const Tensor<Type> &value, bool withHalo,
                          bool tileMappingMustMatch = true) const {
    return pimpl_->isVectorCompatible(value, withHalo, tileMappingMustMatch);
  }

  /**
   * Returns a constant reference to the underlying host matrix.
   */
  const host::HostMatrix<Type> &hostMatrix() const {
    return pimpl_->hostMatrix;
  }

  /**
   * Returns a reference to the underlying host matrix.
   */
  host::HostMatrix<Type> &hostMatrix() { return pimpl_->hostMatrix; }

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