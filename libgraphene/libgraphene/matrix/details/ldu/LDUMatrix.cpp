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

#include "libgraphene/matrix/details/ldu/LDUMatrix.hpp"

#include <iostream>
#include <poplar/GraphElements.hpp>
#include <poplar/Interval.hpp>
#include <poplar/PrintTensor.hpp>
#include <poplar/Program.hpp>
#include <poputil/VertexTemplates.hpp>
#include <type_traits>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/code/ControlFlow.hpp"
#include "libgraphene/dsl/code/Operators.hpp"
#include "libgraphene/dsl/common/details/Expressions.hpp"
#include "libgraphene/dsl/tensor/Execute.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"

namespace graphene::matrix::ldu {
Tensor LDUMatrix::spmv(Tensor &x, TypeRef destType, TypeRef intermediateType,
                       bool withHalo) const {
  DebugInfo di("LDUMatrix");
  // Make sure that x contains halo cells.
  if (!this->isVectorCompatible(x, true)) {
    throw std::runtime_error(
        "shape of vector is incompatible with the matrix layout");
  }

  if (!destType) destType = x.type();

  this->exchangeHaloCells(x);

  // Create an uninitialized tensor for the result with the correct shape
  DistributedShape resultShape = this->tileLayout.getVectorShape(withHalo);
  Tensor result = Tensor::uninitialized(
      destType, resultShape, TileMapping::linearMappingWithShape(resultShape),
      "spmv");

  using namespace codedsl;
  ExecuteThreaded(
      [destType, intermediateType](
          Value workerID, Value x, Value diagCoeffs, Value lowerCoeffs,
          Value upperCoeffs, Value lowerAddr, Value upperAddr,
          Value ownerStartAddr, Value neighbourStartPtr, Value neighbourColInd,
          Value res) {
        Value numCells = diagCoeffs.size();
        // For each row in the matrix
        For(workerID, numCells, getNumWorkerThreadsPerTile(), [&](Value celli) {
          Variable sum = diagCoeffs[celli].cast(destType) *
                         x[celli].cast(intermediateType);

          // Iterate over the upper triangular part
          Variable fStart = ownerStartAddr[celli];
          Variable fEnd = ownerStartAddr[celli + 1];
          AssumeHardwareLoop(fEnd - fStart);
          For(fStart, fEnd, 1, [&](Value face) {
            sum += upperCoeffs[face].cast(intermediateType) *
                   x[upperAddr[face]].cast(intermediateType);
          });

          // Iterate over the lower triangular part
          Variable sStart = neighbourStartPtr[celli];
          Variable sEnd = neighbourStartPtr[celli + 1];
          AssumeHardwareLoop(sEnd - sStart);
          For(sStart, sEnd, 1, [&](Value s) {
            Variable face = neighbourColInd[s];
            sum += lowerCoeffs[s].cast(intermediateType) *
                   x[lowerAddr[face]].cast(intermediateType);
          });
          res[celli] = sum.cast(destType);
        });
      },
      In(x), In(diagonalCoefficients), In(lowerCoefficients),
      In(upperCoefficients ? *upperCoefficients : lowerCoefficients),
      In(addressing->lowerAddr), In(addressing->upperAddr),
      In(addressing->ownerStartAddr), In(addressing->neighbourStartPtr),
      In(addressing->neighbourColInd), Out(result));

  return result;
}

Tensor LDUMatrix::residual(Tensor &x, const Tensor &b, TypeRef destType,
                           TypeRef intermediateType, bool withHalo) const {
  DebugInfo di("LDUMatrix");
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
  DistributedShape resultShape = this->tileLayout.getVectorShape(withHalo);
  Tensor result = Tensor::uninitialized(
      destType, resultShape, TileMapping::linearMappingWithShape(resultShape),
      "residual");

  using namespace codedsl;
  ExecuteThreaded(
      [destType, intermediateType](
          Value workerID, Value x, Value b, Value diagCoeffs, Value lowerCoeffs,
          Value upperCoeffs, Value lowerAddr, Value upperAddr,
          Value ownerStartAddr, Value neighbourStartPtr, Value neighbourColInd,
          Value res) {
        Value numCells = diagCoeffs.size();
        // For each row in the matrix
        For(workerID, numCells, getNumWorkerThreadsPerTile(), [&](Value celli) {
          Variable sum = diagCoeffs[celli].cast(intermediateType) *
                         x[celli].cast(intermediateType);

          // Iterate over the upper triangular part
          Variable fStart = ownerStartAddr[celli];
          Variable fEnd = ownerStartAddr[celli + 1];
          AssumeHardwareLoop(fEnd - fStart);
          For(fStart, fEnd, 1, [&](Value face) {
            sum += upperCoeffs[face].cast(intermediateType) *
                   x[upperAddr[face]].cast(intermediateType);
          });

          // Iterate over the lower triangular part
          Variable sStart = neighbourStartPtr[celli];
          Variable sEnd = neighbourStartPtr[celli + 1];
          AssumeHardwareLoop(sEnd - sStart);
          For(sStart, sEnd, 1, [&](Value s) {
            Variable face = neighbourColInd[s];
            sum += lowerCoeffs[s].cast(intermediateType) *
                   x[lowerAddr[face]].cast(intermediateType);
          });

          // Compute residual: b - Ax
          res[celli] = (b[celli].cast(intermediateType) - sum).cast(destType);
        });
      },
      In(x), In(b), In(diagonalCoefficients), In(lowerCoefficients),
      In(upperCoefficients ? *upperCoefficients : lowerCoefficients),
      In(addressing->lowerAddr), In(addressing->upperAddr),
      In(addressing->ownerStartAddr), In(addressing->neighbourStartPtr),
      In(addressing->neighbourColInd), Out(result));

  return result;
}
}  // namespace graphene::matrix::ldu