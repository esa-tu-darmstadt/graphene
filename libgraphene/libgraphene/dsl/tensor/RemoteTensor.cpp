#include "libgraphene/dsl/tensor/RemoteTensor.hpp"

#include <poplar/DebugContext.hpp>

#include "libgraphene/common/Traits.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene {

template <DataType Type>
Tensor<Type> RemoteTensor<Type>::copyToTile() const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RemoteValue");
  di.add("debugStr", debugStr_);

  poplar::Graph &graph = Context::graph();
  poplar::Graph::TileToTensorMapping tileMapping = tileMapping_;

  poplar::Tensor tensor =
      graph.addVariable(Traits<Type>::PoplarType, shape_, debugStr_);
  graph.setTileMapping(tensor, tileMapping);
  di.addOutput(tensor);

  auto ipuIntervals = calculateIPUIntervals(tileMapping);

  for (size_t ipu = 0; ipu < buffers_.size(); ++ipu) {
    auto &buffer = buffers_[ipu];
    poplar::Tensor ipuTensor = tensor.flatten().slice(ipuIntervals[ipu].begin(),
                                                      ipuIntervals[ipu].end());
    Context::program().add(poplar::program::Copy(buffer, ipuTensor, di));
  }

  return Tensor<Type>(tensor);
}

// Explicit instantiation
#define INSTANTIATE(T) template class RemoteTensor<T>;

INSTANTIATE(float)
INSTANTIATE(bool)
INSTANTIATE(uint8_t)
INSTANTIATE(int8_t)
INSTANTIATE(uint16_t)
INSTANTIATE(int16_t)
INSTANTIATE(uint32_t)
INSTANTIATE(int32_t)

#undef INSTANTIATE
}  // namespace graphene