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
/** A concept of a matrix. Captures all format specific operations. */
template <DataType Type>
struct MatrixBase {
  host::HostMatrix<Type> hostMatrix;

  MatrixBase(std::shared_ptr<host::HostMatrixBase<Type>> hostMatrix)
      : hostMatrix(std::move(hostMatrix)) {}
  virtual ~MatrixBase() = default;

  virtual Value<Type> residual(Value<Type> &x, const Value<Type> &b,
                               bool withHalo = false) const = 0;

  virtual Value<float> residual(Value<double> &x,
                                const Value<float> &b) const = 0;

  virtual Value<Type> operator*(Value<Type> &x) const = 0;

  template <typename VectorType>
  void exchangeHaloCells(Value<VectorType> &value) const;

  template <typename VectorType>
  bool isVectorCompatible(const Value<VectorType> &value, bool withHalo,
                          bool tileMappingMustMatch = true) const;

  size_t numTiles() const { return hostMatrix.numTiles(); }

  /** Returns a view of the vector without the halo cells. Both values have the
   * same underlying tensor, i.e., modifying one will modify the other. */
  template <typename VectorType>
  Value<VectorType> stripHaloCellsFromVector(const Value<VectorType> &x) const;

  /** Returns the specified norm of the given value. In constrast to \ref
   * Value::norm, this strips the halo cells away, which would otherwise be
   * counted multiple times. */
  template <typename VectorType>
  Value<VectorType> vectorNorm(VectorNorm norm,
                               const Value<VectorType> &x) const;

  MatrixFormat getFormat() const { return hostMatrix.getFormat(); }

  template <typename VectorType>
  Value<VectorType> createUninitializedVector(bool withHalo) const;
};
}  // namespace graphene::matrix