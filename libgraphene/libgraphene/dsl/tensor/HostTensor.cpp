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

#include "libgraphene/dsl/tensor/HostTensor.hpp"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <poplar/DataStream.hpp>
#include <unordered_map>

#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"

namespace graphene {
RemoteTensor HostTensor::copyToRemote() const {
  if (!storage_->ownsData()) {
    throw std::runtime_error(
        "Copying data to remote that is not owned by the runtime. This is not "
        "a problem per se, but it may yield a use-after-free if not done "
        "correctly. Feel free to remove this exception if you know what you "
        "are doing.");
  }

  // Buffers per IPU
  std::unordered_map<size_t, poplar::RemoteBuffer> buffers;
  TileMapping ipuMapping = mapping().translateToIPUMapping(
      Context::graph().getTarget().getTilesPerIPU());

  size_t numIPUs = ipuMapping.maxTile() + 1;
  for (size_t ipu = 0; ipu < numIPUs; ++ipu) {
    size_t numElementsOnIPU = ipuMapping.numElementsOnTile(ipu);
    if (numElementsOnIPU == 0) continue;
    // Create one remote buffer for all data mapped to this IPU
    std::string handle = Runtime::instance().registerHandle(
        "hostToRemote_ipu" + std::to_string(ipu));
    poplar::RemoteBuffer buffer = Context::graph().addRemoteBuffer(
        handle, type()->poplarEquivalentType()->poplarType(), numElementsOnIPU);
    buffers[ipu] = buffer;

    auto intervalsOnIPU = ipuMapping[ipu];
    if (intervalsOnIPU.size() > 1) {
      // Not supported, because poplar requires us to copy to the complete
      // remote buffer from host at once. Offsetting the buffer is not
      // supported (see Runtime::copyToRemoteBuffers etc)
      throw std::runtime_error(
          "Copying to remote buffer with non-contiguous intervals is not "
          "supported");
    }

    Interval interval = *intervalsOnIPU.begin();

    // Register the copy of this interval to the remote buffer
    char* intervalData =
        reinterpret_cast<char*>(data()) + interval.start * type()->size();
    Runtime::instance().registerCopyToRemoteBuffer(type(), buffer, intervalData,
                                                   interval.size());
  }

  storage_->wasCopiedToRemote = true;
  return RemoteTensor(type(), buffers, mapping(), shape(), name_);
};

}  // namespace graphene