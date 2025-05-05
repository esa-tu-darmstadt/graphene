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

  poplar::Tensor tensor =
      graph.addVariable(type_->poplarEquivalentType()->poplarType(),
                        shape_.globalShape().toPoplar(), debugStr_);
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