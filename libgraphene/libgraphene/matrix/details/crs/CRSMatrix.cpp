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
Value<Type> CRSMatrix<Type>::operator*(Value<Type> &x) const {
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
  Value<Type> result(resultShape, resultMapping);

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
Value<Type> CRSMatrix<Type>::residual(Value<Type> &x, const Value<Type> &b,
                                      bool withHalo) const {
  return b - (*this) * x;
}

template <DataType Type>
Value<float> CRSMatrix<Type>::residual(Value<double> &x,
                                       const Value<float> &b) const {
  DebugInfo di("CRSMatrix");
  auto &graph = Context::graph();

  if (!std::is_same_v<Type, float>) {
    throw std::runtime_error(
        "Mixed precision residual is only supported for float matrices "
        "currently");
  }
  if (!this->isVectorCompatible(x, true, false)) {
    throw std::runtime_error(
        "x must be a vector with halo cells compatible with the matrix");
  }
  if (!this->isVectorCompatible(b, true, false) &&
      !this->isVectorCompatible(b, false, false)) {
    throw std::runtime_error(
        "b must be a vector with or without halo cells compatible with the "
        "matrix");
  }

  this->exchangeHaloCells(x);

  poplar::ComputeSet cs = graph.addComputeSet(di);

  // Create an uninitialized tensor for the result with the correct shape
  bool resultWithHalo = false;
  auto [resultMapping, resultShape] =
      this->hostMatrix.getVectorTileMappingAndShape(resultWithHalo);
  Value<Type> result(resultShape, resultMapping);

  // The expected tile mapping of x
  auto [xMapping, xShape] = this->hostMatrix.getVectorTileMappingAndShape(true);

  // The expected tile mapping of b
  auto [bMapping, bShape] =
      this->hostMatrix.getVectorTileMappingAndShape(false);
  for (size_t tile = 0; tile < resultMapping.size(); ++tile) {
    if (resultMapping[tile].empty()) continue;

    // Slice the input tensors according to our tile layout. If this layout does
    // not match the tile mapping of the input, poplar will automatically
    // rearrange the data.
    poplar::Tensor xTile = sliceTensorToTile(x.tensor(), tile, &xMapping);
    poplar::Tensor bTile = sliceTensorToTile(b.tensor(), tile, &bMapping);

    poplar::Tensor rowPtrTile = addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile = diagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor offDiagCoeffsTile =
        offDiagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor resultTile = result.tensorOnTile(tile);

    // Choose the codelet based on the data types
    std::string className = "graphene::matrix::crs::ResidualDoubleWord";
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

// Template instantiations
template class CRSMatrix<float>;
}  // namespace graphene::matrix::crs