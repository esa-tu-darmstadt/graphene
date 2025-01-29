#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <poplar/Graph.hpp>
#include <set>
#include <stdexcept>
#include <vector>

#include "libgraphene/common/Shape.hpp"

namespace graphene {

/// Represents a contiguous interval of elements mapped to a tile/IPU: [start,
/// end). This class implements a comparison operator that allows intervals to
/// be placed in a set. When doing so, intervals are automatically sorted
/// correctly.
class MappedInterval {
 public:
  /// Constructs an interval mapped to the given resource with the given start
  /// (inclusive) and end (exclusive): [start, end).
  MappedInterval(size_t resource, size_t start, size_t end)
      : resource_(resource), start_(start), end_(end) {
    if (start_ >= end_) {
      throw std::runtime_error("Invalid interval: start >= end");
    }
  }
  size_t tile() const { return resource_; }
  size_t &tile() { return resource_; }

  size_t start() const { return start_; }
  size_t end() const { return end_; }

  size_t &start() { return start_; }
  size_t &end() { return end_; }

  size_t size() const { return end_ - start_; }
  bool empty() const { return size() == 0; }

  /// Returns true if the given interval overlaps with this interval.
  bool overlaps(const MappedInterval &other) const {
    return start_ < other.end_ && end_ > other.start_;
  }

  /// Compares two intervals by their start position. The intervals must not
  /// overlap.
  bool operator<(const MappedInterval &other) const {
    // Make sure the intervals do not overlap
    if (other == *this) {
      return false;
    }
    if (overlaps(other)) {
      throw std::runtime_error("Cannot compare overlapping intervals");
    }

    return start_ < other.start_;
  }

  /// Compares two intervals for equality.
  bool operator==(const MappedInterval &other) const {
    return std::tie(resource_, start_, end_) ==
           std::tie(other.resource_, other.start_, other.end_);
  }

 private:
  size_t resource_;
  size_t start_;
  size_t end_;
};

/// Represents a mapping of (tensor) intervals to tiles.
class TileMapping {
  std::set<MappedInterval> mapping_;

  /// Constructs a tile mapping from a poplar tile mapping.
  TileMapping(const poplar::Graph::TileToTensorMapping &poplarMapping) {
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

  /// Constructs a tile mapping with the given intervals.
  TileMapping(std::initializer_list<MappedInterval> intervals)
      : mapping_(intervals) {}

  /// Constructs a tile mapping with the given intervals.
  TileMapping(std::set<MappedInterval> intervals) : mapping_(intervals) {}

  /// Returns the intervals mapped to the given tile.
  std::set<MappedInterval> operator[](size_t tile) const {
    std::set<MappedInterval> intervals;
    for (const MappedInterval &interval : mapping_) {
      if (interval.tile() == tile) {
        intervals.insert(interval);
      }
    }
    return intervals;
  }

  /// Inserts the given interval into the tile mapping.
  void map(MappedInterval interval) { mapping_.insert(interval); }

  /// Maps the interval [start, end) to the given tile.
  void map(size_t tile, size_t start, size_t end) {
    map(MappedInterval(tile, start, end));
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
  size_t maxTile() const {
    if (mapping_.empty()) {
      return 0;
    }
    return std::max_element(
               mapping_.begin(), mapping_.end(),
               [](const MappedInterval &a, const MappedInterval &b) {
                 return a.tile() < b.tile();
               })
        ->tile();
  }

  size_t numElementsOnTile(size_t tile) const {
    size_t numElements = 0;
    for (const MappedInterval &interval : mapping_) {
      if (interval.tile() == tile) {
        numElements += interval.size();
      }
    }
    return numElements;
  }

  /// Conversion to a vector of vectors of intervals, as used by
  /// poplar.
  poplar::Graph::TileToTensorMapping toPoplar() const {
    // TileToTensorMapping := std::vector<std::vector<poplar::Interval>>
    poplar::Graph::TileToTensorMapping poplarMapping(maxTile() + 1);
    for (const MappedInterval &interval : mapping_) {
      poplarMapping[interval.tile()].push_back(
          poplar::Interval(interval.start(), interval.end()));
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
    for (const MappedInterval &interval : mapping_) {
      scaledMapping.map(interval.tile(), interval.start() * factor,
                        interval.end() * factor);
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
    for (const MappedInterval &interval : mapping_) {
      size_t ipu = interval.tile() / tilesPerIPU;
      ipuMapping.map(ipu, interval.start(), interval.end());
    }
    return ipuMapping.simplify();
  }

  /// Simplify the tile mapping by merging adjacent intervals on the same tile
  /// and removing empty intervals.
  TileMapping simplify() const {
    // If there is less than two intervals, there is nothing to simplify
    if (mapping_.size() < 2) {
      return *this;
    }

    TileMapping simplified;

    auto it = mapping_.begin();
    // Keep a "current" interval that we'll expand as we find adjacent
    // intervals
    MappedInterval current = *it;
    ++it;

    for (; it != mapping_.end(); ++it) {
      const MappedInterval &next = *it;

      // If 'next' can be merged with 'current', do so
      if (next.tile() == current.tile() && next.start() == current.end()) {
        current.end() = next.end();
      } else {
        // Otherwise, push the finished 'current' interval and move on
        if (!current.empty()) simplified.map(current);
        current = next;
      }
    }

    // Don't forget to add the last accumulated interval
    if (!current.empty()) simplified.map(current);

    return simplified;
  }

  /// Returns true if the tile mapping is compatible with the given shape, i.e.,
  /// if the number of elements mapped to each tile matches between the tile
  /// mapping and the distributed shape.
  bool isCompatibleWithShape(DistributedShape shape) const {
    VectorMap<size_t> elementsPerTile;
    elementsPerTile.reserve(maxTile());

    for (const MappedInterval &interval : mapping_) {
      elementsPerTile[interval.tile()] += interval.size();
    }

    for (auto [tile, numElements] : elementsPerTile) {
      if (numElements != shape.numElementsOnTile(tile)) {
        return false;
      }
    }

    return true;
  }

  void dump() const {
    std::cout << "Tile mapping: " << std::endl;
    for (const MappedInterval &interval : mapping_) {
      std::cout << "  Tile " << interval.tile() << ": [" << interval.start()
                << ", " << interval.end() << ")" << std::endl;
    }
  }

 private:
};
}  // namespace graphene