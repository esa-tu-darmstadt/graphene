#pragma once

#include <any>
#include <memory>
#include <poplar/DebugContext.hpp>
#include <variant>

#include "libgraphene/dsl/Value.hpp"
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

  void solve(Value<Type> &x, Value<Type> &b,
             std::shared_ptr<solver::Configuration> &config);

  /** Computes the residual $ b - Ax $ */
  Value<Type> residual(Value<Type> &x, const Value<Type> &b) const {
    return pimpl_->residual(x, b);
  }

  /** Computes the mixed-precision residual. */
  Value<float> residual(Value<double> &x, const Value<float> &b) const
    requires std::is_same_v<Type, float>
  {
    return pimpl_->residual(x, b);
  }

  /** Sparse matrix vector multiplication */
  Value<Type> operator*(Value<Type> &x) const { return pimpl_->operator*(x); }

  /** Returns a view of the vector without the halo cells. Both values have the
   * same underlying tensor, i.e., modifying one will modify the other. */
  template <typename VectorType>
  Value<VectorType> stripHaloCellsFromVector(const Value<VectorType> &x) const {
    return pimpl_->stripHaloCellsFromVector(x);
  }

  template <typename VectorType>
  Value<VectorType> createUninitializedVector(bool withHalo) const {
    return pimpl_->template createUninitializedVector<VectorType>(withHalo);
  }

  /** Returns the specified norm of the given vector. In constrast to \ref
   * Value::norm, this strips the halo cells away, which would otherwise be
   * counted multiple times by the reduce operation. */
  template <typename VectorType>
  Value<VectorType> vectorNorm(VectorNorm norm,
                               const Value<VectorType> &x) const {
    return pimpl_->vectorNorm(norm, x);
  }

  /** Returns true if the shape if the given vector is compatible to the matrix,
   * i.e. the product $ A*x $ can be calculated. Optionally, checks that the
   * tile mapping of the vector is as expected. */
  bool isVectorCompatible(const Value<Type> &value, bool withHalo,
                          bool tileMappingMustMatch = true) const {
    return pimpl_->isVectorCompatible(value, withHalo, tileMappingMustMatch);
  }

  const host::HostMatrix<Type> &hostMatrix() const {
    return pimpl_->hostMatrix;
  }
  host::HostMatrix<Type> &hostMatrix() { return pimpl_->hostMatrix; }

  size_t numTiles() const { return pimpl_->numTiles(); }

  MatrixFormat getFormat() const { return pimpl_->getFormat(); }
};

}  // namespace graphene::matrix