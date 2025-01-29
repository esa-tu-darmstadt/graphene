#include "libgraphene/dsl/tensor/RemoteTensor.hpp"

#include <poplar/DebugContext.hpp>

#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Traits.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene {

Tensor RemoteTensor::copyToTile() const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RemoteValue");
  di.add("debugStr", debugStr_);

  poplar::Graph &graph = Context::graph();

  poplar::Tensor tensor = graph.addVariable(
      type_->poplarType(), shape_.globalShape().toPoplar(), debugStr_);
  graph.setTileMapping(tensor, tileMapping_.toPoplar());
  di.addOutput(tensor);

  TileMapping ipuMapping =
      tileMapping_.translateToIPUMapping(graph.getTarget().getTilesPerIPU());

  size_t numIPUs = ipuMapping.maxTile() + 1;
  for (size_t ipu = 0; ipu < numIPUs; ++ipu) {
    size_t numElements = ipuMapping.numElementsOnTile(ipu);
    if (numElements == 0) continue;

    poplar::RemoteBuffer buffer = buffers_.at(ipu);
    Context::program().add(poplar::program::Copy(
        buffer, sliceTensorToIPU(tensor, ipu, tileMapping_), di));
  }

  return Tensor::fromPoplar(tensor, type_);
}
}  // namespace graphene