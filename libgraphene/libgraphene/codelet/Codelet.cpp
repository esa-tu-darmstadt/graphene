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

#include "libgraphene/codelet/Codelet.hpp"

#include <dlfcn.h>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/codelets.hpp>

using namespace graphene;

void graphene::addCodelets(poplar::Graph &graph,
                           poplar::program::Sequence &prog) {
  popops::addCodelets(graph);

#ifndef GRAPHENE_PRECOMPILED_CODELETS_PATH
#error "GRAPHENE_PRECOMPILED_CODELETS_PATH not defined"
#endif

  std::string path = GRAPHENE_PRECOMPILED_CODELETS_PATH;

  graph.addCodelets(path, poplar::CodeletFileType::Object);

  if (graph.getTarget().getTargetType() == poplar::TargetType::IPU) {
    // Add codelets for stack reservation
    poplar::ComputeSet cs = graph.addComputeSet("stackReservation");
    for (size_t proc = 0; proc < graph.getTarget().getNumTiles(); ++proc) {
      poplar::VertexRef v1 =
          graph.addVertex(cs, "sim::codelet::StackReservationWorker");
      graph.setTileMapping(v1, proc);
      poplar::VertexRef v2 =
          graph.addVertex(cs, "sim::codelet::StackReservationSupervisor");
      graph.setTileMapping(v2, proc);
    }
    prog.add(poplar::program::Execute(cs));
  }
}