#include "libgraphene/matrix/solver/SolverStats.hpp"

#include <spdlog/spdlog.h>

#include "libgraphene/dsl/Operators.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Runtime.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver {
void SolverStats::checkConvergence(float absTolerance, float relTolerance,
                                   float relResidual) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("SolverStats");

  if (!bNorm && relTolerance > 0) {
    throw std::runtime_error(
        "Using the relative tolerance (||res||/||b||) as a "
        "convergence criterium requires calculating the norm of b beforehand, "
        "which was not done by the solver.");
  }

  if (relTolerance > 0) {
    converged = finalResidual < absTolerance ||
                finalResidual < (initialResidual * relResidual) ||
                finalResidual < (bNorm.value() * relTolerance);
    bNorm.value().print("bNorm");
  } else {
    // Do not consider the relative tolerance because ||b|| may have not been
    // calculated
    converged = finalResidual < absTolerance ||
                finalResidual < (initialResidual * relResidual);
  }
}

void SolverStats::checkSingularity(Tensor<float> wApA, float tolerance) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("SolverStats");

  singular = singular || wApA < tolerance;
}

void SolverStats::print() const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("SolverStats");
  auto &program = Context::program();
  auto &graph = Context::graph();
  auto &runtime = Runtime::instance();

  // Register custom print function
  std::string handle = runtime.registerHandle("PrintSolverStats");
  auto func = graph.addHostFunction(handle,
                                    {/* initialResidual */ {poplar::FLOAT, 1},
                                     /* finalResidual */ {poplar::FLOAT, 1},
                                     /* norm factor */ {poplar::FLOAT, 1},
                                     /* num interations*/ {poplar::INT, 1},
                                     /* converged */ {poplar::BOOL, 1},
                                     /* singular */ {poplar::BOOL, 1}},
                                    {});

  // Capture the values we want to print because this SolverPerformance object
  // will be destroyed when the host function is called
  std::string solverNameCopy = solverName;
  VectorNorm normCopy = norm;
  size_t tilesCopy = numTiles;

  runtime.registerHostFunction(handle, [solverNameCopy, normCopy, tilesCopy](
                                           poplar::ArrayRef<const void *> ins,
                                           poplar::ArrayRef<void *> /*outs*/) {
    float initialResidual = *(float *)ins[0];
    float finalResidual = *(float *)ins[1];
    float normFactor = *(float *)ins[2];
    int iterations = *(uint32_t *)ins[3];
    bool converged = *(bool *)ins[4];
    bool singular = *(bool *)ins[5];

    std::string executionTime = Runtime::instance().getCurrentExecutionTime();
    std::string normString = normToString(normCopy);

    // Print message for unscaled norms
    spdlog::info(
        "{}: Solved on {} IPU tiles, {} norm, Initial residual = {}, "
        "Final residual = {}, Iterations = {}, Converged = {}, Singular = "
        "{}, Current Execution Time = {}",
        solverNameCopy, tilesCopy, normString, initialResidual, finalResidual,
        iterations, converged, singular, executionTime);
  });
  program.add(poplar::program::Call(
      func,
      {initialResidual.tensor(true), finalResidual.tensor(true),
       normFactor.tensor(true), iterations.tensor(true), converged.tensor(true),
       singular.tensor(true)},
      {}, di));
  // converged.print("converged");
  // FIXME: converged is always false if we do not use it after the host
  // function call. Maybe a bug in the liveness analysis of poplar?
}

}  // namespace graphene::matrix::solver