#pragma once

#include <spdlog/spdlog.h>

#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "libgraphene/common/VectorMap.hpp"
#include "poplar/Graph.hpp"

namespace graphene {

/// Represents the shape of a tensor. Shapes are always non-empty. Scalar shapes
/// are represented by a shape with a single dimension of size 1.
class TensorShape : private std::vector<size_t> {
 public:
  using std::vector<size_t>::iterator;
  using std::vector<size_t>::reverse_iterator;
  using std::vector<size_t>::const_iterator;
  using std::vector<size_t>::const_reverse_iterator;

  /// Default constructor, creates a scalar shape
  TensorShape() {
    // Empty shapes are handled as scalar shapes
    this->push_back(1);
  }

  /// Create a shape with the given dimensions
  explicit TensorShape(std::vector<size_t> dims) : std::vector<size_t>(dims) {
    if (this->empty()) {
      // Empty shapes are handled as scalar shapes
      this->push_back(1);
    }
  }

  /// Create a shape with the given (static) dimensions
  TensorShape(std::initializer_list<size_t> dims)
      : TensorShape(std::vector<size_t>(dims)) {}

  /// Make some functions of the base class public
  using std::vector<size_t>::begin;
  using std::vector<size_t>::end;
  using std::vector<size_t>::cbegin;
  using std::vector<size_t>::cend;
  using std::vector<size_t>::rbegin;
  using std::vector<size_t>::rend;
  using std::vector<size_t>::crbegin;
  using std::vector<size_t>::crend;
  using std::vector<size_t>::operator[];
  using std::vector<size_t>::push_back;
  using std::vector<size_t>::pop_back;

  /// Return the rank of the shape
  size_t rank() const { return this->size(); }

  size_t dim(size_t index) const { return this->at(index); }
  size_t &dim(size_t index) { return this->at(index); }

  /// Return the number of elements in the tensor
  size_t numElements() const {
    return std::accumulate(begin(), end(), 1, std::multiplies<>());
  }

  /// Return true if the shape is a scalar
  bool isScalar() const { return rank() == 1 && dim(0) == 1; }

  /// Return the distance in elements between two consecutive elements in the
  /// given dimension.
  size_t stride(size_t dimension) const {
    size_t stride = 1;
    for (size_t i = dimension + 1; i < rank(); ++i) {
      stride *= dim(i);
    }
    return stride;
  }

  bool operator==(const TensorShape &other) const {
    return static_cast<const std::vector<size_t> &>(*this) ==
           static_cast<const std::vector<size_t> &>(other);
  }

  const std::vector<size_t> &toPoplar() const { return *this; }
};

using FirstDimDistribution = VectorMap<size_t>;

/// Represents the shape of a tensor and how it is distributed across tiles
/// along its first dimension. Empty shapes are equivalent to scalar shapes.
///
/// In this framework, tensors are distributed across tiles in the first
/// dimension. Thus, the size of the first dimension is different for each
/// tile. The size of all other dimensions is the same for all tiles.
class DistributedShape {
 public:
 private:
  /// The global shape of the tensor, i.e., the first dimension is accumulated
  TensorShape globalShape_;

  /// The sizes of the first dimension on each tile
  FirstDimDistribution firstDimDistribution_;

 private:
  /// Create a distributed shape with the given global shape and tile sizes
  /// for the first dimension. The user is expected to use factory functions
  /// instead of this constructor.
  DistributedShape(TensorShape globalShape,
                   FirstDimDistribution firstDimDistribution)
      : globalShape_(std::move(globalShape)),
        firstDimDistribution_(std::move(firstDimDistribution)) {
    // Make sure that the sizes of the first dimension on all tiles sum up
    // to the size of the first dimension in the global shape
    size_t sum = std::accumulate(
        firstDimDistribution_.begin(), firstDimDistribution_.end(), 0u,
        [](size_t sum, auto &&pair) { return sum + pair.value; });
    if (sum != globalShape_[0]) {
      throw std::invalid_argument(
          "Sum of first dimension sizes does not match first dimension "
          "of global shape.");
    }

    // Make sure that no dimension is empty
    for (size_t dim : globalShape_) {
      if (dim == 0) {
        throw std::invalid_argument("Shape dimension " + std::to_string(dim) +
                                    " is zero.");
      }
    }
  }

 public:
  DistributedShape() : graphene::DistributedShape(scalar()) {}

  /// Return true if the distributed shape is valid, i.e., the sizes of the
  /// first dimension on all tiles sum up to the size of the first dimension
  /// in the global shape. This should always return true.
  bool valid() {
    size_t sum = std::accumulate(
        firstDimDistribution_.begin(), firstDimDistribution_.end(), 0u,
        [](size_t sum, auto &&pair) { return sum + pair.value; });
    return sum == globalShape_[0];
  }

 public:
  static DistributedShape onSingleTile(TensorShape shape, size_t tile = 0) {
    FirstDimDistribution firstDimDistribution;
    firstDimDistribution[tile] = shape.dim(0);

    return DistributedShape(shape, firstDimDistribution);
  }

  static DistributedShape onTiles(TensorShape shape,
                                  FirstDimDistribution firstDimDistribution) {
    return DistributedShape(std::move(shape), std::move(firstDimDistribution));
  }

  /// Create a shape with the given global shape and a linear distribution of
  /// the first dimension across the given number of tiles.
  static DistributedShape createLinearlyDistributed(TensorShape shape,
                                                    size_t numTiles) {
    FirstDimDistribution firstDimDistribution;
    size_t stride = shape.stride(0);
    size_t tileSize = shape.dim(0) / numTiles;
    for (size_t tile = 0; tile < numTiles; ++tile) {
      firstDimDistribution[tile] = tileSize;
    }
    firstDimDistribution[numTiles - 1] += shape.dim(0) % numTiles;
    return DistributedShape(shape, firstDimDistribution);
  }

  /// Return a scalar shape mapped to the given tile
  static DistributedShape scalar(size_t tile = 0) {
    return onSingleTile({}, tile);
  }

  /// Create a shape from a Poplar shape. In contrast to the constructor, this
  /// function allows empty shapes and treats them as a scalar shape. Poplar
  /// allows empty shapes to represent scalar tensors.
  static DistributedShape fromPoplar(
      const std::vector<size_t> &dims,
      const poplar::Graph::TileToTensorMapping &mapping) {
    if (dims.empty()) return scalar();

    TensorShape globalShape(dims);

    size_t stride = 1;
    for (size_t i = 1; i < dims.size(); ++i) stride *= dims[i];

    FirstDimDistribution firstDimDistribution;
    firstDimDistribution.reserve(mapping.size());
    size_t firstDimSum = 0;  // For verifcation purposes
    for (size_t tile = 0; tile < mapping.size(); ++tile) {
      size_t numElementsOnTile = std::accumulate(
          mapping[tile].begin(), mapping[tile].end(), 0,
          [stride](size_t sum, const poplar::Interval &interval) {
            return sum + interval.size();
          });
      size_t firstDimSizeOnTile = numElementsOnTile / stride;
      firstDimDistribution[tile] = firstDimSizeOnTile;
      firstDimSum += firstDimSizeOnTile;
    }

    if (firstDimSum != dims[0]) {
      throw std::logic_error(
          "Sum of first dimension sizes calculated from tile mapping does "
          "not "
          "match first dimension of given shape.");
    }

    std::vector<size_t> dimsExceptFirst;
    if (dims.size() > 1) {
      dimsExceptFirst = std::vector<size_t>(dims.begin() + 1, dims.end());
    }
    return onTiles(globalShape, firstDimDistribution);
  }

  /// Return the rank of the shape
  size_t rank() const { return globalShape_.rank(); }

  /// Return the number of elements in the tensor
  size_t numElements() const { return globalShape_.numElements(); }

  /// Return the number of elements in the tensor that are mapped to the given
  /// tile
  size_t numElementsOnTile(size_t tile) const {
    return firstDimDistribution_[tile] * globalShape_.stride(0);
  }

  /// Append a dimension to the shape. The tile id is not required because
  /// only the first dimension is distributed across tiles.
  void push_back(size_t size) { globalShape_.push_back(size); }

  /// The shape of the full tensor, with the first dimension accumulated
  const TensorShape &globalShape() const { return globalShape_; }

  /// The shape of the full tensor, with the first dimension accumulated
  TensorShape &globalShape() { return globalShape_; }

  /// The sizes of the first dimension on each tile.
  const FirstDimDistribution &firstDimDistribution() const {
    return firstDimDistribution_;
  }

  /// The sizes of the first dimension on each tile.
  FirstDimDistribution &firstDimDistribution() { return firstDimDistribution_; }

  /// Return the size of the given (global) dimension.
  size_t operator[](size_t index) const { return globalShape_[index]; }

  /// Return the distance in elements between two consecutive elements in the
  /// given dimension. This is always the same for all tiles.
  size_t stride(size_t dimension) const {
    return globalShape_.stride(dimension);
  }

  /// Compare two shapes for equality
  bool operator==(const DistributedShape &other) const {
    // Check if the first dimension is equal on all tiles
    if (globalShape_ != other.globalShape_) return false;
    if (firstDimDistribution_ != other.firstDimDistribution_) return false;
    return true;
  }

  /// Return a string representation of the shape
  std::string str() const {
    bool first = true;
    std::string str;
    for (size_t dim : globalShape_) {
      if (!first) str += "x";
      str += std::to_string(dim);
      first = false;
    }
    return str;
  }

  /// Dump the shape to stdout
  void dump() const { std::cout << str() << std::endl; }

  std::ostream &operator<<(std::ostream &os) const { return os << str(); }

  /**
   * Groups the first dimension of the shape by the specified group size. For
   * example, grouping the first dimension distribution {0: 1, 1: 1, 2: 1, 3: 1}
   * with a group size of 2 will result in {0: 2, 1:, 0, 2: 2, 3: 0}.
   *
   * @param groupSize The size of the group to combine elements of the first
   * dimension. If groupSize is 1, the original shape is returned.
   * @return A new Shape object with the first dimension grouped by the
   * specified size. The other dimensions remain unchanged.
   */
  DistributedShape groupFirstDimension(size_t groupSize) const {
    if (groupSize == 1) return *this;

    size_t maxNonGroupedTile = firstDimDistribution_.maxKey();
    size_t maxGroupedTile = maxNonGroupedTile - maxNonGroupedTile % groupSize;

    FirstDimDistribution groupedFirstDimSizes;
    groupedFirstDimSizes.reserve(maxGroupedTile);
    for (size_t group = 0; group <= maxGroupedTile; group++) {
      for (size_t tile = group * groupSize;
           tile < std::min((group + 1) * groupSize, maxNonGroupedTile + 1);
           tile++) {
        groupedFirstDimSizes[group * groupSize] += firstDimDistribution_[tile];
      }
    }
    return DistributedShape(globalShape_, groupedFirstDimSizes);
  }

  // /// Pads the given shape with 1s at the beginning to increase the rank to
  // the
  // /// given value
  // static Shape withRank(Shape shape, size_t rank) {
  //   if (shape.rank() > rank) {
  //     throw std::runtime_error("Cannot decrease rank of shape");
  //   }
  //   Shape newShape = Shape::withRank(rank);
  //   std::copy(shape.rbegin(), shape.rend(), newShape.rbegin());
  //   return newShape;
  // }

  /// Broadcast two shapes according to NumPy broadcasting rules if possible .
  /// If the shapes are not compatible for broadcasting, return std::nullopt.
  /// The rules for broadcasting are as follows:
  /// Comparing the dimensions from the last to the first:
  /// * Each dimension must be equal, or
  /// * One of the dimensions must be of size 1, or
  /// * The dimension does not exist in one of the tensors
  static std::optional<DistributedShape> broadcast(DistributedShape shape1,
                                                   DistributedShape shape2) {
    DistributedShape &longerShape =
        shape1.rank() > shape2.rank() ? shape1 : shape2;
    DistributedShape &shorterShape =
        shape1.rank() > shape2.rank() ? shape2 : shape1;
    DistributedShape broadcastedShape = longerShape;

    auto firstDimDistributionCompatible = [](DistributedShape &shape1,
                                             DistributedShape &shape2) -> bool {
      if (shape1[0] == 1) return true;
      if (shape2[0] == 1) return true;
      if (shape1.firstDimDistribution() == shape2.firstDimDistribution())
        return true;
      return false;
    };

    // If the shapes have a different rank, then the shorter shape is extended
    // with 1s at the beginning. Thus, its first dimension is set to one, which
    // is always compatible with the first dimension of the longer shape.

    // If the shapes have the same rank but different first dimension
    // distributions, one of the first dimensions must be 1
    if (shape1.rank() == shape2.rank() &&
        shape1.firstDimDistribution() != shape2.firstDimDistribution()) {
      if (shape1[0] == 1)
        broadcastedShape.firstDimDistribution() = shape2.firstDimDistribution();
      else if (shape2[0] == 1)
        broadcastedShape.firstDimDistribution() = shape1.firstDimDistribution();
      else {
        spdlog::trace(
            "Shapes have the same rank but the first dimension does not "
            "match: {} vs {}",
            shape1[0], shape2[0]);
        return std::nullopt;
      }
    }

    // At this point we know that all dimension but the last n-1 are set and
    // compatible, with n being the rank of the shorter shape. We can now
    // compare the last n-1 dimensions.

    for (size_t i = 1; i < shorterShape.rank(); ++i) {
      // Compare the right-aligned dimensions
      size_t longerDim = i + longerShape.rank() - shorterShape.rank();
      size_t shorterDim = i;

      if (longerShape[longerDim] == 1) {
        broadcastedShape.globalShape()[longerDim] = shorterShape[shorterDim];
      } else if (shorterShape[shorterDim] == 1) {
        broadcastedShape.globalShape()[longerDim] = longerShape[longerDim];
      } else if (longerShape[longerDim] == shorterShape[shorterDim]) {
        broadcastedShape.globalShape()[longerDim] = longerShape[longerDim];
      } else {
        spdlog::trace(
            "Shapes are not compatible for broadcasting because they have "
            "different incompatible dimensions: {} vs {}",
            longerShape[longerDim], shorterShape[shorterDim]);
        return std::nullopt;
      }
    }
    return broadcastedShape;
  }
};

}  // namespace graphene