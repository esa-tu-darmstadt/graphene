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

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <poplar/Graph.hpp>
#include <set>
#include <stdexcept>
#include <vector>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/VectorMap.hpp"

namespace graphene {

/// Represents a contiguous interval of elements mapped to a tile/IPU: [start,
/// end).
struct Interval {
  size_t start;
  size_t end;
  size_t size() const { return end - start; }
  bool empty() const { return size() == 0; }

  bool operator==(const Interval &other) const {
    return start == other.start && end == other.end;
  }
};

/// Represents a mapping of tiles to (tensor) intervals.
class TileMapping {
  VectorMap<std::vector<Interval>> mapping_;

  /// Constructs a tile mapping from a poplar tile mapping.
  TileMapping(const poplar::Graph::TileToTensorMapping &poplarMapping) {
    reserve(poplarMapping.size());
    for (size_t tile = 0; tile < poplarMapping.size(); ++tile) {
      for (const poplar::Interval &interval : poplarMapping[tile]) {
        map(tile, interval.begin(), interval.end());
      }
    }
  }

 public:
  using iterator = decltype(mapping_)::iterator;
  using const_iterator = decltype(mapping_)::const_iterator;

  TileMapping() = default;

  /// Reserves space for the given number of tiles. This can be used to avoid
  /// reallocations when inserting many intervals.
  void reserve(size_t maxTile) { mapping_.reserve(maxTile); }

  /// Returns the intervals mapped to the given tile.
  const std::vector<Interval> &operator[](size_t tile) const {
    return mapping_[tile];
  }

  /// Inserts the given interval into the tile mapping.
  void map(size_t tile, Interval interval) {
    mapping_[tile].push_back(interval);
  }

  /// Maps the interval [start, end) to the given tile.
  void map(size_t tile, size_t start, size_t end) {
    map(tile, Interval{start, end});
  }

  /// Returns an iterator to the beginning of the tile mapping.
  iterator begin() { return mapping_.begin(); }

  /// Returns an iterator to the end of the tile mapping.
  iterator end() { return mapping_.end(); }

  /// Returns a const iterator to the beginning of the tile mapping.
  const_iterator begin() const { return mapping_.begin(); }

  /// Returns a const iterator to the end of the tile mapping.
  const_iterator end() const { return mapping_.end(); }

  /// True if the tile mapping is empty.
  bool empty() const { return mapping_.empty(); }

  /// Returns the maximum tile number in the tile mapping.
  size_t maxTile() const { return mapping_.maxKey(); }

  size_t numElementsOnTile(size_t tile) const {
    return std::accumulate(mapping_[tile].begin(), mapping_[tile].end(), 0,
                           [](size_t sum, const Interval &interval) {
                             return sum + interval.end - interval.start;
                           });
  }

  /// Conversion to a vector of vectors of intervals, as used by
  /// poplar.
  poplar::Graph::TileToTensorMapping toPoplar() const {
    // TileToTensorMapping := std::vector<std::vector<poplar::Interval>>
    poplar::Graph::TileToTensorMapping poplarMapping(maxTile() + 1);
    for (auto [tile, intervals] : mapping_) {
      for (const Interval &interval : intervals) {
        poplarMapping[tile].push_back(
            poplar::Interval(interval.start, interval.end));
      }
    }
    return poplarMapping;
  }

  /// Conversion from a poplar tile mapping.
  static TileMapping fromPoplar(
      const poplar::Graph::TileToTensorMapping &poplarMapping) {
    return TileMapping(poplarMapping).simplify();
  }

  /// Generates a tile mapping where elements are distributed linearly across
  /// tiles according to the dimensions specified in the Shape.
  static TileMapping linearMappingWithShape(const DistributedShape &shape) {
    TileMapping mapping;
    size_t currentElement = 0;
    size_t stride = shape.globalShape().stride(0);
    for (auto [tile, size] : shape.firstDimDistribution()) {
      mapping.map(tile, currentElement, currentElement + size * stride);
      currentElement += size * stride;
    }
    return mapping;
  }

  /// Scale up the tile mapping by the given factor.
  TileMapping scaleUp(size_t factor) const {
    TileMapping scaledMapping;
    for (auto [tile, intervals] : mapping_) {
      for (const Interval &interval : intervals)
        scaledMapping.map(tile, interval.start * factor, interval.end * factor);
    }

    return scaledMapping;
  }

  /// Could be more efficient if we directly compare the intervals
  bool operator==(const TileMapping &other) const {
    return simplify().mapping_ == other.simplify().mapping_;
  }

  /// Translate the tile mapping to an IPU mapping by grouping tiles into
  /// IPUs.
  /// FIXME: We cannot query the number of tiles per IPU from the graph
  /// ourselves because we cannot acces the graph (Context::graph() is in a
  /// library that depends on this library). Maybe this class is better placed
  /// in the DSL library?
  TileMapping translateToIPUMapping(size_t tilesPerIPU) const {
    TileMapping ipuMapping;
    for (auto [tile, intervals] : mapping_) {
      size_t ipu = tile / tilesPerIPU;
      for (const Interval &interval : intervals) {
        ipuMapping.map(ipu, interval);
      }
    }
    return ipuMapping.simplify();
  }

  /// Simplify the tile mapping by merging adjacent intervals on the same tile
  /// and removing empty intervals.
  TileMapping simplify() const {
    TileMapping simplified;

    for (auto [tile, intervals] : mapping_) {
      auto it = intervals.begin();
      // Keep a "current" interval that we'll expand as we find adjacent
      // intervals
      Interval current = *it;
      ++it;

      for (; it != intervals.end(); ++it) {
        const Interval &next = *it;

        // If 'next' can be merged with 'current', do so
        if (next.start == current.end) {
          current.end = next.end;
        } else {
          // Otherwise, push the finished 'current' interval and move on
          if (!current.empty()) simplified.map(tile, current);
          current = next;
        }
      }

      // Don't forget to add the last accumulated interval
      if (!current.empty()) simplified.map(tile, current);
    }

    return simplified;
  }

  /// Returns true if the tile mapping is compatible with the given shape, i.e.,
  /// if the number of elements mapped to each tile matches between the tile
  /// mapping and the distributed shape.
  bool isCompatibleWithShape(DistributedShape shape) const {
    VectorMap<size_t> elementsPerTile;
    elementsPerTile.reserve(maxTile());

    for (auto [tile, intervals] : mapping_) {
      size_t numElements = 0;
      for (const Interval &interval : intervals) {
        numElements += interval.end - interval.start;
      }
      if (numElements != shape.numElementsOnTile(tile)) {
        return false;
      }
    }

    return true;
  }

  void dump() const {
    std::cout << "Tile mapping: " << std::endl;
    for (auto [tile, intervals] : mapping_) {
      std::cout << "  Tile " << tile << ": ";
      for (const Interval &interval : intervals) {
        std::cout << "[" << interval.start << ", " << interval.end << ") ";
      }
      std::cout << std::endl;
    }
  }

 private:
};
}  // namespace graphene