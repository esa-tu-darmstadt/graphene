#include "libgraphene/codelet/Codelet.hpp"

#include <dlfcn.h>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/codelets.hpp>

using namespace graphene;

void graphene::addCodelets(poplar::Graph &graph,
                           poplar::program::Sequence &prog) {
  popops::addCodelets(graph);

  Dl_info dlInfo;
  if (dladdr((void *)::graphene::addCodelets, &dlInfo) == 0) {
    throw std::runtime_error("Could not find path to libGrapheneCodelet.so");
  }

  std::string path(dlInfo.dli_fname);
  path = path.substr(0, path.find_last_of('/') + 1);
  path = path + "ipu/GrapheneCodeletsIPU.gp";

  graph.addCodelets(path, poplar::CodeletFileType::Object);

  // Add codelets for stack reservation
  poplar::ComputeSet cs = graph.addComputeSet("stackReservation");
  for (size_t proc = 0; proc < graph.getTarget().getNumTiles(); ++proc) {
    poplar::VertexRef v = graph.addVertex(cs, "sim::codelet::StackReservation");
    graph.setTileMapping(v, proc);
  }
  prog.add(poplar::program::Execute(cs));
}