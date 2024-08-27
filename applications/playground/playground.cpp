#include <spdlog/spdlog.h>

#include <cstddef>
#include <memory>

#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"
#include "libgraphene/matrix/solver/gauss-seidel/Configuration.hpp"
#include "libgraphene/matrix/solver/ilu/Configuration.hpp"
#include "libgraphene/matrix/solver/iterative-refinement/Configuration.hpp"
#include "libgraphene/matrix/solver/pbicgstab/Configuration.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"

using namespace graphene;
using namespace matrix;

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);
  //   Runtime::enableProfiling();

  Runtime runtime(1);

  spdlog::info("Building data flow graph");

  //   auto host_A = host::loadMatrixFromFile<float>(
  //       "data/matrices/diffusor_5mio/A.mtx", 1472 * 1);
  //   auto host_x = host::loadVectorFromFile<float>(
  //       "data/matrices/diffusor_5mio/x.mtx", host_A, true);
  //   auto host_b = host::loadVectorFromFile<float>(
  //       "data/matrices/diffusor_5mio/b.mtx", host_A);

  auto host_A = host::loadMatrixFromFile<float>(
      "data/matrices/laplacian_8x8x8/A.mtx", 1 * 1);
  auto host_x = host::loadVectorFromFile<float>(
      "data/matrices/laplacian_8x8x8/b.mtx", host_A, true);
  auto host_b = host::loadVectorFromFile<float>(
      "data/matrices/laplacian_8x8x8/b.mtx", host_A);

  auto A = host_A.copyToTile();
  auto x = host_x.copyToRemote().copyToTile();
  auto b = host_b.copyToRemote().copyToTile();

  //   auto& impl = A.getImpl<crs::CRSMatrix<float>>();

  //   impl.diagonalCoefficients.print("diag coeffs");
  //   impl.offDiagonalCoefficients.print("off-diag coeffs");
  //   impl.addressing->colInd.print("col indices");
  //   impl.addressing->rowPtr.print("row ptr");

  auto gaussSeidelConfig =
      std::make_shared<solver::gaussseidel::Configuration>();
  gaussSeidelConfig->printPerformanceAfterSolve = true;
  gaussSeidelConfig->solveMulticolor = solver::MultiColorMode::Auto;
  // gaussSeidelConfig->numFixedIterations = 100;
  gaussSeidelConfig->numSweeps = 3;
  gaussSeidelConfig->maxIterations = 10;
  gaussSeidelConfig->absTolerance = 1e-9;
  gaussSeidelConfig->norm = VectorNorm::L2;

  auto iluConfig = std::make_shared<solver::ilu::Configuration>();
  iluConfig->solveMulticolor = solver::MultiColorMode::On;
  iluConfig->factorizeMulticolor = solver::MultiColorMode::On;
  iluConfig->diagonalBased = false;

  auto pbicgstabConfig = std::make_shared<solver::pbicgstab::Configuration>();
  //   pbicgstabConfig->printPerformanceAfterSolve = true;
  //   pbicgstabConfig->printPerformanceEachIteration = true;
  pbicgstabConfig->relResidual = 1e-5;
  pbicgstabConfig->norm = VectorNorm::L2;
  pbicgstabConfig->maxIterations = 50;
  //   pbicgstabConfig->verbose = true;
  pbicgstabConfig->preconditioner = iluConfig;

  auto irConfig =
      std::make_shared<solver::iterativerefinement::Configuration>();
  irConfig->printPerformanceAfterSolve = true;
  irConfig->absTolerance = 1e-12;
  irConfig->norm = VectorNorm::L2;
  irConfig->maxIterations = 100;
  irConfig->innerSolver = iluConfig;
  irConfig->mixedPrecision = true;

  auto config1 = std::static_pointer_cast<solver::Configuration>(irConfig);

  A.solve(x, b, config1);

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  spdlog::info("Done!");
  return 0;
}