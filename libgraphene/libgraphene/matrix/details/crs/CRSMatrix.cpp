#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"

#include <iostream>
#include <poplar/GraphElements.hpp>
#include <poplar/Interval.hpp>
#include <poplar/PrintTensor.hpp>
#include <poplar/Program.hpp>
#include <poputil/VertexTemplates.hpp>
#include <type_traits>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/code/Operators.hpp"
#include "libgraphene/dsl/common/details/Expressions.hpp"
#include "libgraphene/dsl/tensor/Execute.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"

namespace graphene::matrix::crs {
Tensor CRSMatrix::spmv(Tensor &x, TypeRef destType, TypeRef intermediateType,
                       bool withHalo) const {
  DebugInfo di("CRSMatrix");
  // Make sure that x contains halo cells.
  if (!this->isVectorCompatible(x, true)) {
    throw std::runtime_error(
        "shape of vector is incompatible with the matrix layout");
  }

  if (!destType) destType = x.type();

  this->exchangeHaloCells(x);

  // Create an uninitialized tensor for the result with the correct shape
  DistributedShape resultShape = this->hostMatrix.getVectorShape(withHalo);
  Tensor result = Tensor::uninitialized(
      destType, resultShape, TileMapping::linearMappingWithShape(resultShape),
      "spmv");

  using namespace codedsl;
  ExecuteThreaded(
      [destType, intermediateType](Value workerID, Value x, Value rowPtr,
                                   Value colInd, Value diag, Value offDiag,
                                   Value res) {
        // For each row in the matrix
        For(workerID, rowPtr.size() - 1, getNumWorkerThreadsPerTile(),
            [&](Value i) {
              Variable sum =
                  diag[i].cast(destType) * x[i].cast(intermediateType);
              // For each non-diagonal element in the row
              For(rowPtr[i], rowPtr[i + 1], 1, [&](Value j) {
                sum += offDiag[j].cast(intermediateType) *
                       x[colInd[j]].cast(intermediateType);
              });
              res[i] = sum.cast(destType);
            });
      },
      In(x), In(addressing->rowPtr), In(addressing->colInd),
      In(diagonalCoefficients), In(offDiagonalCoefficients), Out(result));

  return result;
}

Tensor CRSMatrix::residual(Tensor &x, const Tensor &b, TypeRef destType,
                           TypeRef intermediateType, bool withHalo) const {
  DebugInfo di("CRSMatrix");
  // Make sure that x contains halo cells.
  if (!this->isVectorCompatible(x, true)) {
    throw std::runtime_error(
        "shape of vector is incompatible with the matrix layout");
  }

  if (!destType)
    destType =
        detail::inferType(detail::BinaryOpType::MULTIPLY, x.type(), b.type());

  this->exchangeHaloCells(x);

  // Create an uninitialized tensor for the result with the correct shape
  DistributedShape resultShape = this->hostMatrix.getVectorShape(withHalo);
  Tensor result = Tensor::uninitialized(
      destType, resultShape, TileMapping::linearMappingWithShape(resultShape),
      "spmv");

  using namespace codedsl;
  ExecuteThreaded(
      [destType, intermediateType](Value workerID, Value x, Value b,
                                   Value rowPtr, Value colInd, Value diag,
                                   Value offDiag, Value res) {
        // For each row in the matrix
        For(workerID, rowPtr.size() - 1, getNumWorkerThreadsPerTile(),
            [&](Value i) {
              Variable sum =
                  diag[i].cast(intermediateType) * x[i].cast(intermediateType);
              // For each non-diagonal element in the row
              For(rowPtr[i], rowPtr[i + 1], 1, [&](Value j) {
                sum += offDiag[j].cast(intermediateType) *
                       x[colInd[j]].cast(intermediateType);
              });
              res[i] = (b[i].cast(intermediateType) - sum).cast(destType);
            });
      },
      In(x), In(b), In(addressing->rowPtr), In(addressing->colInd),
      In(diagonalCoefficients), In(offDiagonalCoefficients), Out(result));

  return result;
}
}  // namespace graphene::matrix::crs