#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"

#include <iostream>
#include <poplar/GraphElements.hpp>
#include <poplar/Interval.hpp>
#include <poplar/PrintTensor.hpp>
#include <poplar/Program.hpp>
#include <poputil/VertexTemplates.hpp>
#include <type_traits>

#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"

namespace graphene::matrix::crs {
template <DataType Type>
Tensor<Type> CRSMatrix<Type>::operator*(Tensor<Type> &x) const {
  DebugInfo di("CRSMatrix");
  auto &graph = Context::graph();

  // Make sure that x contains halo cells.
  if (!this->isVectorCompatible(x, true, false)) {
    throw std::runtime_error(
        "x must be a vector with halo cells compatible with the matrix");
  }

  this->exchangeHaloCells(x);

  poplar::ComputeSet cs = graph.addComputeSet(di);

  // Create an uninitialized tensor for the result with the correct shape
  bool resultWithHalo = false;
  auto [resultMapping, resultShape] =
      this->hostMatrix.getVectorTileMappingAndShape(resultWithHalo);
  Tensor<Type> result(resultShape, resultMapping);

  // The expected tile mapping of the input tensor
  auto [mappingWithHalo, shapeWithHalo] =
      this->hostMatrix.getVectorTileMappingAndShape(true);

  for (size_t tile = 0; tile < resultMapping.size(); ++tile) {
    if (resultMapping[tile].empty()) continue;

    // Slice the input tensors according to our tile layout. If this layout does
    // not match the tile mapping of the input, poplar will automatically
    // rearrange the data.
    poplar::Tensor xTile =
        sliceTensorToTile(x.tensor(), tile, &mappingWithHalo);
    poplar::Tensor rowPtrTile = addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile = diagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor offDiagCoeffsTile =
        offDiagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor resultTile = result.tensorOnTile(tile);

    // If there are no interior and seperator rows on this tile, we can skip
    if (!diagCoeffsTile.valid()) continue;

    // Choose the codelet based on the data types
    std::string className = "graphene::matrix::crs::MatrixVectorMultiply";
    std::string codeletName = poputil::templateVertex(
        className, Traits<Type>::PoplarType, rowPtrTile.elementType(),
        colIndTile.elementType());

    auto vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["x"], xTile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["diagCoeffs"], diagCoeffsTile);
    graph.connect(vertex["offDiagCoeffs"], offDiagCoeffsTile);
    graph.connect(vertex["result"], resultTile);

    graph.setPerfEstimate(vertex, 100);
  }
  Context::program().add(poplar::program::Execute(cs, di));
  return result;
}

template <DataType Type>
Tensor<Type> CRSMatrix<Type>::residual(Tensor<Type> &x, const Tensor<Type> &b,
                                       bool withHalo) const {
  return b - (*this) * x;
}

template <DataType Type, DataType ExtendedType>
static Tensor<Type> mixedPrecisionResidual(const CRSMatrix<Type> &matrix,
                                           Tensor<ExtendedType> &x,
                                           const Tensor<Type> &b) {
  DebugInfo di("CRSMatrix");
  auto &graph = Context::graph();

  if (!matrix.isVectorCompatible(x, true, false)) {
    throw std::runtime_error(
        "x must be a vector with halo cells compatible with the matrix");
  }
  if (!matrix.isVectorCompatible(b, true, false) &&
      !matrix.isVectorCompatible(b, false, false)) {
    throw std::runtime_error(
        "b must be a vector with or without halo cells compatible with the "
        "matrix");
  }

  matrix.exchangeHaloCells(x);

  poplar::ComputeSet cs = graph.addComputeSet(di);

  // Create an uninitialized tensor for the result with the correct shape
  bool resultWithHalo = false;
  auto [resultMapping, resultShape] =
      matrix.hostMatrix.getVectorTileMappingAndShape(resultWithHalo);
  Tensor<Type> result(resultShape, resultMapping);

  // The expected tile mapping of x
  auto [xMapping, xShape] =
      matrix.hostMatrix.getVectorTileMappingAndShape(true);

  // The expected tile mapping of b
  auto [bMapping, bShape] =
      matrix.hostMatrix.getVectorTileMappingAndShape(false);
  for (size_t tile = 0; tile < resultMapping.size(); ++tile) {
    if (resultMapping[tile].empty()) continue;

    // Slice the input tensors according to our tile layout. If this layout does
    // not match the tile mapping of the input, poplar will automatically
    // rearrange the data.
    poplar::Tensor xTile = sliceTensorToTile(x.tensor(), tile, &xMapping);
    poplar::Tensor bTile = sliceTensorToTile(b.tensor(), tile, &bMapping);

    poplar::Tensor rowPtrTile = matrix.addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = matrix.addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile =
        matrix.diagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor offDiagCoeffsTile =
        matrix.offDiagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor resultTile = result.tensorOnTile(tile);

    // Choose the codelet based on the data types
    std::string className = "graphene::matrix::crs::ResidualMixedPrecision";
    if (std::is_same_v<Type, float> &&
        std::is_same_v<ExtendedType, doubleword>) {
      className += "DoublewordToFloat";
    } else if (std::is_same_v<Type, float> &&
               std::is_same_v<ExtendedType, double>) {
      className += "DoubleToFloat";
    } else {
      throw std::runtime_error("Invalid data types");
    }
    std::string codeletName = poputil::templateVertex(
        className, rowPtrTile.elementType(), colIndTile.elementType());

    auto vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["x"], xTile);
    graph.connect(vertex["b"], bTile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["diagCoeffs"], diagCoeffsTile);
    graph.connect(vertex["offDiagCoeffs"], offDiagCoeffsTile);
    graph.connect(vertex["result"], resultTile);

    graph.setPerfEstimate(vertex, 100);
  }
  Context::program().add(poplar::program::Execute(cs, di));
  return result;
}

template <>
Tensor<float> CRSMatrix<float>::residual(Tensor<double> &x,
                                         const Tensor<float> &b) const {
  return mixedPrecisionResidual<float, double>(*this, x, b);
}

template <>
Tensor<float> CRSMatrix<float>::residual(Tensor<doubleword> &x,
                                         const Tensor<float> &b) const {
  return mixedPrecisionResidual<float, doubleword>(*this, x, b);
}

// Template instantiations
template class CRSMatrix<float>;
}  // namespace graphene::matrix::crs