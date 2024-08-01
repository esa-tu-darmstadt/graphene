#include "libgraphene/util/Context.hpp"

std::deque<poplar::Graph *> graphene::Context::graphs_ =
    std::deque<poplar::Graph *>();
std::deque<poplar::program::Sequence *> graphene::Context::programs_ =
    std::deque<poplar::program::Sequence *>();
poplar::program::Sequence *graphene::Context::preludeProgram_;