#include "libgraphene/matrix/solver/ilu/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/ControlFlow.hpp"
#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::ilu {

template <DataType Type>
Solver<Type>::Solver(const Matrix<Type>& matrix,
                     std::shared_ptr<Configuration> config)
    : solver::Solver<Type>(matrix),
      config_(config),
      solveMulticolor_(this->shouldUseMulticolor(config_->solveMulticolor)),
      factorizeMulticolor_(
          this->shouldUseMulticolor(config_->factorizeMulticolor)) {
  factorize();
}

template <DataType Type>
void Solver<Type>::factorizeCRS() {
  const crs::CRSMatrix<Type>& A =
      this->matrix().template getImpl<crs::CRSMatrix<Type>>();
  GRAPHENE_TRACEPOINT();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  CRSFactorization factorization;
  factorization.factorizedInverseDiag =
      std::make_unique<Value<Type>>(A.diagonalCoefficients);
  if (!config_->diagonalBased) {
    factorization.factorizedOffDiag = Value<Type>(A.offDiagonalCoefficients);
  }

  if (factorizeMulticolor_) {
    if (!A.addressing->coloring.has_value()) {
      throw std::runtime_error(
          "Matrix has no coloring. Cannot use multicolor factorization");
    }
    spdlog::trace("Using multicolor factorization");
  } else {
    spdlog::trace("Using sequential factorization");
  }
  spdlog::trace("Factorizing in {}-mode", this->name());

  poplar::ComputeSet cs = graph.addComputeSet(di);

  for (size_t tile = 0; tile < A.numTiles(); ++tile) {
    poplar::Tensor rowPtrTile = A.addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = A.addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile =
        factorization.factorizedInverseDiag->tensorOnTile(tile);
    poplar::Tensor offDiagCoeffsTile;
    if (config_->diagonalBased)
      offDiagCoeffsTile = A.offDiagonalCoefficients.tensorOnTile(tile);
    else
      offDiagCoeffsTile = factorization.factorizedOffDiag->tensorOnTile(tile);
    poplar::Tensor colorSortAddrTile, colorSortStartPtrTile;
    if (factorizeMulticolor_) {
      colorSortAddrTile =
          A.addressing->coloring->colorSortAddr.tensorOnTile(tile);
      colorSortStartPtrTile =
          A.addressing->coloring->colorSortStartPtr.tensorOnTile(tile);
    }

    std::string codeletName = "graphene::matrix::solver::ilu::ILUFactorizeCRS";
    if (config_->diagonalBased) {
      codeletName += "Diagonal";
    }
    if (factorizeMulticolor_) {
      codeletName += "Multicolor";
      codeletName = poputil::templateVertex(
          codeletName, Traits<Type>::PoplarType, rowPtrTile.elementType(),
          colIndTile.elementType(), colorSortAddrTile.elementType(),
          colorSortStartPtrTile.elementType());
    } else {
      codeletName = poputil::templateVertex(
          codeletName, Traits<Type>::PoplarType, rowPtrTile.elementType(),
          colIndTile.elementType());
    }

    auto vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["diagCoeffs"], diagCoeffsTile);
    graph.connect(vertex["offDiagCoeffs"], offDiagCoeffsTile);
    if (factorizeMulticolor_) {
      graph.connect(vertex["colorSortAddr"], colorSortAddrTile);
      graph.connect(vertex["colorSortStartAddr"], colorSortStartPtrTile);
    }

    graph.setPerfEstimate(vertex, 100);
  }

  program.add(poplar::program::Execute(cs, di));

  // Invert the diagonal
  *factorization.factorizedInverseDiag =
      (Type)1 / *factorization.factorizedInverseDiag;

  factorization_ = std::move(factorization);
}

template <DataType Type>
void Solver<Type>::solveCRS(Value<Type>& x, Value<Type>& b) {
  const auto& A = this->matrix().template getImpl<crs::CRSMatrix<Type>>();
  GRAPHENE_TRACEPOINT();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  CRSFactorization& factorization = std::get<CRSFactorization>(factorization_);

  if (solveMulticolor_) {
    if (!A.addressing->coloring.has_value()) {
      throw std::runtime_error(
          "Matrix has no coloring. Cannot use multicolor substitution");
    }
    spdlog::trace("Using multicolor substitution");
  } else {
    spdlog::trace("Using sequential substitution");
  }
  spdlog::trace("Solving in {}-mode", this->name());

  poplar::ComputeSet cs = graph.addComputeSet(di);

  for (size_t tile = 0; tile < A.numTiles(); ++tile) {
    poplar::Tensor xTile = x.tensorOnTile(tile);
    poplar::Tensor bTile = b.tensorOnTile(tile);
    poplar::Tensor rowPtrTile = A.addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = A.addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile =
        factorization.factorizedInverseDiag->tensorOnTile(tile);

    poplar::Tensor offDiagCoeffsTile;
    if (config_->diagonalBased)
      offDiagCoeffsTile = A.offDiagonalCoefficients.tensorOnTile(tile);
    else
      offDiagCoeffsTile = factorization.factorizedOffDiag->tensorOnTile(tile);
    poplar::Tensor colorSortAddrTile, colorSortStartPtrTile;
    if (solveMulticolor_) {
      colorSortAddrTile =
          A.addressing->coloring->colorSortAddr.tensorOnTile(tile);
      colorSortStartPtrTile =
          A.addressing->coloring->colorSortStartPtr.tensorOnTile(tile);
    }

    std::string codeletName = "graphene::matrix::solver::ilu::ILUSolveCRS";
    if (solveMulticolor_) {
      codeletName += "Multicolor";
      codeletName = poputil::templateVertex(
          codeletName, Traits<Type>::PoplarType, rowPtrTile.elementType(),
          colIndTile.elementType(), colorSortAddrTile.elementType(),
          colorSortStartPtrTile.elementType(), config_->diagonalBased);
    } else {
      codeletName = poputil::templateVertex(
          codeletName, Traits<Type>::PoplarType, rowPtrTile.elementType(),
          colIndTile.elementType(), config_->diagonalBased);
    }

    auto vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["x"], xTile);
    graph.connect(vertex["b"], bTile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["inverseDiagCoeffs"], diagCoeffsTile);
    graph.connect(vertex["offDiagCoeffs"], offDiagCoeffsTile);
    if (solveMulticolor_) {
      graph.connect(vertex["colorSortAddr"], colorSortAddrTile);
      graph.connect(vertex["colorSortStartAddr"], colorSortStartPtrTile);
    }

    graph.setPerfEstimate(vertex, 100);
  }

  program.add(poplar::program::Execute(cs, di));
}

template <DataType Type>
void Solver<Type>::factorize() {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      factorizeCRS();
      break;
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

template <DataType Type>
SolverStats Solver<Type>::solve(Value<Type>& x, Value<Type>& b) {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      solveCRS(x, b);
      break;
    default:
      throw std::runtime_error("Unsupported matrix format");
  }

  SolverStats stats(this->name(), VectorNorm::None, this->matrix().numTiles());
  stats.iterations = 1;
  return stats;
}

template class Solver<float>;
}  // namespace graphene::matrix::solver::ilu