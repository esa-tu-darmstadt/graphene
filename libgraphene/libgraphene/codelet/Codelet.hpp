#pragma once

namespace poplar {
class Graph;
namespace program {
class Sequence;
}  // namespace program
}  // namespace poplar
namespace graphene {
/// \brief Adds the codelets required for the graphene library to the given
/// graph.
void addCodelets(poplar::Graph &graph, poplar::program::Sequence &prog);
}  // namespace graphene