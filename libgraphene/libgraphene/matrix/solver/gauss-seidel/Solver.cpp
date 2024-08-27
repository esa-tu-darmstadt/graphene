#include "libgraphene/matrix/solver/gauss-seidel/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/ControlFlow.hpp"
#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::gaussseidel {

template <DataType Type>
Solver<Type>::Solver(const Matrix<Type>& matrix,
                     std::shared_ptr<const gaussseidel::Configuration> config)
    : solver::Solver<Type>(matrix),
      config_(std::move(config)),
      solveMulticolor_(this->shouldUseMulticolor(config_->solveMulticolor)) {}

template <DataType Type>
void Solver<Type>::solveIterationCSR(Tensor<Type>& x, Tensor<Type>& b) const {
  const auto& A = this->matrix().template getImpl<crs::CRSMatrix<Type>>();
  DebugInfo di("GaussSeidelSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  if (solveMulticolor_) {
    if (!A.addressing->coloring.has_value()) {
      throw std::runtime_error(
          "Matrix has no coloring. Cannot use multicolor gauss-seidel");
    }
    spdlog::trace("Using multicolor gauss-seidel");
  } else {
    spdlog::trace("Using sequential gauss-seidel");
  }

  A.exchangeHaloCells(x);

  poplar::ComputeSet cs = graph.addComputeSet(di);

  for (size_t tile = 0; tile < A.numTiles(); ++tile) {
    poplar::Tensor xTile = x.tensorOnTile(tile);
    poplar::Tensor bTile = b.tensorOnTile(tile);
    poplar::Tensor rowPtrTile = A.addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = A.addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile = A.diagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor offDiagCoeffsTile =
        A.offDiagonalCoefficients.tensorOnTile(tile);
    poplar::Tensor colorSortAddrTile, colorSortStartPtrTile;
    if (solveMulticolor_) {
      colorSortAddrTile =
          A.addressing->coloring->colorSortAddr.tensorOnTile(tile);
      colorSortStartPtrTile =
          A.addressing->coloring->colorSortStartPtr.tensorOnTile(tile);
    }

    std::string codeletName =
        "graphene::matrix::solver::gaussseidel::GausSeidelSolveSmoothCRS";
    if (solveMulticolor_) {
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
    graph.connect(vertex["x"], xTile);
    graph.connect(vertex["b"], bTile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["diagCoeffs"], diagCoeffsTile);
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
void gaussseidel::Solver<Type>::solveIteration(Tensor<Type>& x,
                                               Tensor<Type>& b) const {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      solveIterationCSR(x, b);
      break;
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

template <DataType Type>
SolverStats Solver<Type>::solve(Tensor<Type>& x, Tensor<Type>& b) {
  GRAPHENE_TRACEPOINT();
  spdlog::trace("Solving with Gauss-Seidel");

  DebugInfo di("GaussSeidelSolver");
  const Matrix<Type>& A = this->matrix();

  if (!A.isVectorCompatible(x, true, true))
    throw std::runtime_error("x must be vector with halo cells.");

  if (!A.isVectorCompatible(b, false, true))
    throw std::runtime_error("b must be vector without halo cells.");

  auto& program = Context::program();

  SolverStats stats(name(), config_->norm, A.numTiles());

  if (config_->numFixedIterations > 0) {
    // A fixed number of iterations is requested
    // Do not check for convergence
    spdlog::trace("Using a fixed number of iterations for Gauss-Seidel: {}",
                  config_->numFixedIterations);

    cf::Repeat(config_->numFixedIterations, [&]() { solveIteration(x, b); });

    stats.iterations = config_->numFixedIterations;
    return stats;
  }

  // Calculate the norm of b if required
  if (stats.requiresBNorm(config_->relTolerance))
    stats.bNorm = A.vectorNorm(config_->norm, b);

  // Calculate the initial residual
  Tensor<Type> initialResidual = A.residual(x, b);
  stats.initialResidual = A.vectorNorm(config_->norm, initialResidual);
  stats.finalResidual = stats.initialResidual;

  auto terminate =
      (stats.converged && stats.iterations >= config_->minIterations) ||
      (stats.iterations >= config_->maxIterations);

  cf::While(!terminate, [&]() {
    // Sweep the solver
    cf::Repeat(config_->numSweeps, [&]() {
      solveIteration(x, b);
      stats.iterations = stats.iterations + 1;
    });

    // Calculate the residual and check for convergence
    auto currentResidual = A.residual(x, b);
    stats.finalResidual = A.vectorNorm(config_->norm, currentResidual);
    stats.checkConvergence(config_->absTolerance, config_->relTolerance,
                           config_->relResidual);

    if (config_->printPerformanceEachIteration) stats.print();
  });

  if (config_->printPerformanceAfterSolve) stats.print();

  return stats;
}

template class Solver<float>;
}  // namespace graphene::matrix::solver::gaussseidel