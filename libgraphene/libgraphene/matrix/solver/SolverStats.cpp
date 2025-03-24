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

#include "libgraphene/matrix/solver/SolverStats.hpp"

#include <spdlog/spdlog.h>

#include <poplar/Type.hpp>

#include "libgraphene/dsl/tensor/Operators.hpp"
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

void SolverStats::checkSingularity(Tensor wApA, float tolerance) {
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
  auto func =
      graph.addHostFunction(handle,
                            {/* initialResidual */ {poplar::FLOAT, 1},
                             /* finalResidual */ {poplar::FLOAT, 1},
                             /* norm factor */ {poplar::FLOAT, 1},
                             /* num interations*/ {poplar::UNSIGNED_INT, 1},
                             /* converged */ {poplar::BOOL, 1},
                             /* singular */ {poplar::BOOL, 1}},
                            {});

  // Capture the values we want to print by value  because this
  // SolverPerformance object will be destroyed when the host function is called
  runtime.registerHostFunction(handle, [solverName = solverName, norm = norm,
                                        tiles = numTiles](
                                           poplar::ArrayRef<const void *> ins,
                                           poplar::ArrayRef<void *> /*outs*/) {
    float initialResidual = *(float *)ins[0];
    float finalResidual = *(float *)ins[1];
    float normFactor = *(float *)ins[2];
    int iterations = *(uint32_t *)ins[3];
    bool converged = *(bool *)ins[4];
    bool singular = *(bool *)ins[5];

    std::string executionTime = Runtime::instance().getCurrentExecutionTime();
    std::string normString = normToString(norm);

    // Print message for unscaled norms
    spdlog::info(
        "{}: Solved on {} IPU tiles, {} norm, Initial residual = {}, "
        "Final residual = {}, Iterations = {}, Converged = {}, Singular = "
        "{}, Current Execution Time = {}",
        solverName, tiles, normString, initialResidual, finalResidual,
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