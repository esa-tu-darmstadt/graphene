#include "libgraphene/dsl/HostValue.hpp"

#include <spdlog/spdlog.h>

#include <poplar/DataStream.hpp>

#include "libgraphene/dsl/RemoteValue.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"

namespace graphene {
template <DataType Type>
RemoteValue<Type> HostValue<Type>::copyToRemote() const {
  // Make sure that the number of elements in the data vector is equal to the
  // product of the shape vector
  size_t numElementsInShape = std::accumulate(shape().begin(), shape().end(), 1,
                                              std::multiplies<size_t>());
  if (data().size() != numElementsInShape) {
    throw std::runtime_error(
        fmt::format("The number of elements in the data vector ({}) is not "
                    "equal to the product of the shape vector ({})",
                    data().size(), numElementsInShape));
  }

  std::vector<poplar::RemoteBuffer> buffers;
  auto ipuIntervals = calculateIPUIntervals(mapping());
  for (size_t ipu = 0; ipu < ipuIntervals.size(); ++ipu) {
    std::string handle = Runtime::instance().registerHandle(
        "hostToRemote_ipu" + std::to_string(ipu));
    poplar::RemoteBuffer buffer = Context::graph().addRemoteBuffer(
        handle, Traits<Type>::PoplarType, ipuIntervals[ipu].size(), 1, true);
    Runtime::instance().registerCopyToRemoteBuffer(
        buffer, data().data() + ipuIntervals[ipu].begin());
    buffers.push_back(buffer);
  }

  storage_->wasCopiedToRemote = true;
  return RemoteValue<Type>(buffers, mapping(), shape(), storage_->name);
};

// Template instantiations
template class HostValue<uint8_t>;
template class HostValue<int8_t>;
template class HostValue<uint16_t>;
template class HostValue<int16_t>;
template class HostValue<uint32_t>;
template class HostValue<int32_t>;
template class HostValue<float>;
}  // namespace graphene