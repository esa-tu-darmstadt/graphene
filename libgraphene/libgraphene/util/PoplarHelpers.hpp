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

#pragma once

#include <poplar/Tensor.hpp>
#include <vector>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"

namespace graphene {

/**
 * Returns the slice of a tensor that is mapped to a specific IPU. The tensor
 * has the same rank but is sliced along the first dimension.
 *
 * @param tensor The tensor to slice.
 * @param ipu The IPU to slice the tensor for.
 */
poplar::Tensor sliceTensorToIPU(poplar::Tensor tensor, size_t ipu,
                                const TileMapping &mapping);

/**
 * Returns the slice of a tensor that is mapped to a specific tile. The tensor
 * has the same rank but is sliced along the first dimension.
 *
 * @param tensor The tensor to slice.
 * @param tile The tile to slice the tensor for.
 * @param cache The mapping of the tensor to tiles. If not provided, the mapping
 * will be queried from the graph, which can take a significant amount of time.
 */
poplar::Tensor sliceTensorToTile(poplar::Tensor tensor, size_t tile,
                                 const TileMapping &mapping);

/**
 * @brief Converts an element index to its corresponding coordinates in its
 * tensor.
 *
 * Given the shape of a tensor and an element index, this function calculates
 * the coordinates of the element in the tensor
 *
 * @param shape The shape of the tensor.
 * @param elementIndex The index of the element.
 * @return The coordinates of the element in the tensor.
 */
std::vector<size_t> getCoordinateFromElementIndex(const DistributedShape &shape,
                                                  size_t elementIndex);

}  // namespace graphene