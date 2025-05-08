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

#include <gtest/gtest.h>

#include <stdexcept>

#include "libgraphene/common/Shape.hpp"

using namespace graphene;

class DistributedShapeTest : public ::testing::Test {};

// Construction and Basic Properties
TEST_F(DistributedShapeTest, ScalarConstructionAndProperties) {
  auto shape = DistributedShape::scalar(3);  // Scalar on tile 3
  EXPECT_EQ(shape.rank(), 1);
  EXPECT_EQ(shape.numElements(), 1);
  EXPECT_EQ(shape.numElementsOnTile(3), 1);
  EXPECT_EQ(shape.numElementsOnTile(0), 0);
  EXPECT_EQ(shape.globalShape().isScalar(), true);
}

TEST_F(DistributedShapeTest, SingleTileConstruction) {
  TensorShape tensor_shape({4, 3, 2});  // 4x3x2 tensor
  auto shape = DistributedShape::onSingleTile(tensor_shape, 5);

  EXPECT_EQ(shape.rank(), 3);
  EXPECT_EQ(shape.numElements(), 24);  // 4 * 3 * 2
  EXPECT_EQ(shape.numElementsOnTile(5), 24);
  EXPECT_EQ(shape.numElementsOnTile(0), 0);
  EXPECT_TRUE(shape.valid());
}

TEST_F(DistributedShapeTest, DistributedConstruction) {
  TensorShape tensor_shape({6, 3, 2});  // 6x3x2 tensor
  FirstDimDistribution dist(
      {{0, 2}, {1, 4}});  // 2 elements on tile 0, 4 on tile 1

  auto shape = DistributedShape::onTiles(tensor_shape, dist);
  EXPECT_EQ(shape.rank(), 3);
  EXPECT_EQ(shape.numElements(), 36);         // 6 * 3 * 2
  EXPECT_EQ(shape.numElementsOnTile(0), 12);  // 2 * 3 * 2
  EXPECT_EQ(shape.numElementsOnTile(1), 24);  // 4 * 3 * 2
  EXPECT_TRUE(shape.valid());
}

// Distribution Validation
TEST_F(DistributedShapeTest, InvalidDistributionThrows) {
  TensorShape tensor_shape({4, 2});
  FirstDimDistribution invalid_dist({{0, 2}, {1, 1}});  // Sum = 3, should be 4

  EXPECT_THROW(DistributedShape::onTiles(tensor_shape, invalid_dist),
               std::invalid_argument);
}

TEST_F(DistributedShapeTest, EmptyDistributionBehavior) {
  TensorShape tensor_shape({0, 2});
  FirstDimDistribution empty_dist;

  EXPECT_THROW(DistributedShape::onTiles(tensor_shape, empty_dist),
               std::invalid_argument);
}

// Dimension Manipulation
TEST_F(DistributedShapeTest, PushBackDimension) {
  TensorShape initial_shape({2, 3});
  auto shape = DistributedShape::onSingleTile(initial_shape, 0);

  shape.push_back(4);
  EXPECT_EQ(shape.rank(), 3);
  EXPECT_EQ(shape.numElements(), 24);  // 2 * 3 * 4
  EXPECT_EQ(shape.globalShape().dim(2), 4);
}

// Poplar Integration
TEST_F(DistributedShapeTest, PoplarConversion) {
  TensorShape tensor_shape({4, 3, 2});
  auto shape = DistributedShape::onSingleTile(tensor_shape, 0);

  const auto& poplar_shape = shape.globalShape().toPoplar();
  EXPECT_EQ(poplar_shape.size(), 3);
  EXPECT_EQ(poplar_shape[0], 4);
  EXPECT_EQ(poplar_shape[1], 3);
  EXPECT_EQ(poplar_shape[2], 2);
}

TEST_F(DistributedShapeTest, PoplarShapeConstruction) {
  std::vector<size_t> dims{4, 3, 2};
  poplar::Graph::TileToTensorMapping mapping(2);   // 2 tiles
  mapping[0].push_back(poplar::Interval(0, 12));   // First 12 elements
  mapping[1].push_back(poplar::Interval(12, 24));  // Next 12 elements

  auto shape = DistributedShape::fromPoplar(dims, mapping);
  EXPECT_EQ(shape.rank(), 3);
  EXPECT_EQ(shape.numElements(), 24);
  EXPECT_EQ(shape.numElementsOnTile(0), 12);
  EXPECT_EQ(shape.numElementsOnTile(1), 12);
}

// Grouping Operations
TEST_F(DistributedShapeTest, GroupFirstDimension) {
  TensorShape tensor_shape({4, 2});
  FirstDimDistribution dist({{0, 1}, {1, 1}, {2, 1}, {3, 1}});
  auto shape = DistributedShape::onTiles(tensor_shape, dist);

  auto grouped = shape.groupFirstDimension(2);
  EXPECT_EQ(grouped.firstDimDistribution()[0], 2);  // Tiles 0,1 grouped
  EXPECT_EQ(grouped.firstDimDistribution()[2], 2);  // Tiles 2,3 grouped
  EXPECT_TRUE(grouped.valid());
}

TEST_F(DistributedShapeTest, GroupFirstDimensionIdentity) {
  TensorShape tensor_shape({4, 2});
  FirstDimDistribution dist({{0, 2}, {1, 2}});
  auto shape = DistributedShape::onTiles(tensor_shape, dist);

  auto grouped = shape.groupFirstDimension(1);
  EXPECT_EQ(shape, grouped);  // Should be identical when group size is 1
}

// String Representation
TEST_F(DistributedShapeTest, StringRepresentation) {
  TensorShape tensor_shape({2, 3, 4});
  auto shape = DistributedShape::onSingleTile(tensor_shape, 0);

  EXPECT_EQ(shape.str(), "2x3x4");
}

// Element Counting and Strides
TEST_F(DistributedShapeTest, ElementCountingAndStrides) {
  TensorShape tensor_shape({4, 3, 2});
  FirstDimDistribution dist({{0, 1}, {1, 3}});
  auto shape = DistributedShape::onTiles(tensor_shape, dist);

  EXPECT_EQ(shape.numElementsOnTile(0), 6);     // 1 * 3 * 2
  EXPECT_EQ(shape.numElementsOnTile(1), 18);    // 3 * 3 * 2
  EXPECT_EQ(shape.globalShape().stride(0), 6);  // Stride in first dimension
  EXPECT_EQ(shape.globalShape().stride(1), 2);  // Stride in second dimension
}

// Equality Comparison
TEST_F(DistributedShapeTest, EqualityComparison) {
  TensorShape shape1({4, 3});
  TensorShape shape2({4, 3});
  TensorShape shape3({4, 2});

  FirstDimDistribution dist1({{0, 2}, {1, 2}});
  FirstDimDistribution dist2({{0, 2}, {1, 2}});
  FirstDimDistribution dist3({{0, 1}, {1, 3}});

  auto distributed1 = DistributedShape::onTiles(shape1, dist1);
  auto distributed2 = DistributedShape::onTiles(shape2, dist2);
  auto distributed3 = DistributedShape::onTiles(shape1, dist3);
  auto distributed4 = DistributedShape::onTiles(shape3, dist1);

  EXPECT_EQ(distributed1, distributed2);
  EXPECT_NE(distributed1, distributed3);
  EXPECT_NE(distributed1, distributed4);
}

TEST_F(DistributedShapeTest, DistributedLinearly) {
  TensorShape tensor_shape({7, 3, 2});
  auto shape = DistributedShape::createLinearlyDistributed(tensor_shape, 3);

  EXPECT_EQ(shape.firstDimDistribution()[0], 2);
  EXPECT_EQ(shape.firstDimDistribution()[1], 2);
  EXPECT_EQ(shape.firstDimDistribution()[2], 3);
  EXPECT_EQ(shape.firstDimDistribution().count(), 3);
  EXPECT_TRUE(shape.valid());
}

TEST_F(DistributedShapeTest, Copy) {
  DistributedShape shape1 =
      DistributedShape::createLinearlyDistributed({8, 2, 4}, 4);

  // Copy constructor
  DistributedShape shape2 = shape1;
  EXPECT_EQ(shape1, shape2);

  // Copy assignment
  DistributedShape shape3 = DistributedShape::scalar(0);
  shape3 = shape1;
  EXPECT_EQ(shape1, shape3);
}

TEST_F(DistributedShapeTest, BroadcastSameRankCompatible) {
  // Both shapes have rank=2, matching rest-of-dims, and compatible first
  // dimension
  TensorShape t1({2, 3});
  FirstDimDistribution d1({{0, 2}});
  auto ds1 = DistributedShape::onTiles(t1, d1);

  TensorShape t2({2, 3});
  FirstDimDistribution d2({{0, 2}});
  auto ds2 = DistributedShape::onTiles(t2, d2);

  auto result = DistributedShape::broadcast(ds1, ds2);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->globalShape().dim(0), 2u);
  EXPECT_EQ(result->globalShape().dim(1), 3u);
}

TEST_F(DistributedShapeTest, BroadcastSameRankIncompatible) {
  // Both shapes have rank=2 but different first-dim distribution with no
  // shape=1
  TensorShape t1({2, 3});
  FirstDimDistribution d1({{0, 2}});
  auto ds1 = DistributedShape::onTiles(t1, d1);

  TensorShape t2({2, 3});
  FirstDimDistribution d2({{0, 1}, {1, 1}});
  auto ds2 = DistributedShape::onTiles(t2, d2);

  auto result = DistributedShape::broadcast(ds1, ds2);
  EXPECT_FALSE(result.has_value());
}

TEST_F(DistributedShapeTest, BroadcastDifferentRank) {
  // dsA: rank=2, dsB: rank=3 with first-dim=1 => should allow broadcasting
  TensorShape tA({1, 4});
  FirstDimDistribution dA({{0, 1}});
  auto dsA = DistributedShape::onTiles(tA, dA);

  TensorShape tB({2, 3, 4});
  FirstDimDistribution dB({{0, 2}});
  auto dsB = DistributedShape::onTiles(tB, dB);

  auto result = DistributedShape::broadcast(dsA, dsB);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->globalShape().rank(), 3u);
  EXPECT_EQ(result->globalShape().dim(0), 2u);
  EXPECT_EQ(result->globalShape().dim(1), 3u);
  EXPECT_EQ(result->globalShape().dim(2), 4u);
}

TEST_F(DistributedShapeTest, BroadcastScalarAndNonScalar) {
  // Broadcasting a scalar
  auto dsScalar = DistributedShape::scalar(0);  // rank=1, first-dim=1
  TensorShape tB({3, 2});
  FirstDimDistribution dB({{0, 3}});
  auto dsB = DistributedShape::onTiles(tB, dB);

  auto result = DistributedShape::broadcast(dsScalar, dsB);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->globalShape().rank(), 2u);
  EXPECT_EQ(result->globalShape().dim(0), 3u);
  EXPECT_EQ(result->globalShape().dim(1), 2u);
}

TEST_F(DistributedShapeTest, BroadcastIncompatibleDifferentRanks) {
  // Shape A has rank=1 with dim(0)=10
  // Shape B has rank=2 with dims(0)=1, dims(1)=3
  // After padding A to rank 2, we get {1, 10} which is incompatible with {1, 3}
  TensorShape tA({10});
  FirstDimDistribution dA({{0, 10}});
  auto dsA = DistributedShape::onTiles(tA, dA);

  TensorShape tB({1, 3});
  FirstDimDistribution dB({{0, 1}});
  auto dsB = DistributedShape::onTiles(tB, dB);

  auto result = DistributedShape::broadcast(dsA, dsB);
  EXPECT_FALSE(result.has_value());
}