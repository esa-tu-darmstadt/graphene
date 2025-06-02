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

#include <sstream>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/PrintTensorFormat.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Runtime.hpp"

using namespace graphene;

class TensorTest : public testing::Test {};

template <DataType T>
HostTensor createHostTensor(DistributedShape shape, TypeRef type,
                            std::string name = "test") {
  std::vector<T> data(shape.numElements());
  for (size_t i = 0; i < shape.numElements(); ++i) {
    data[i] = i;
  }
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);
  return HostTensor::createPersistent(std::move(data), std::move(shape),
                                      std::move(mapping), std::move(name));
}

#define EXPECT_HOSTTENSOR_EQ(host1, host2)             \
  {                                                    \
    EXPECT_EQ(host1.shape(), host2.shape());           \
    for (size_t i = 0; i < host1.numElements(); ++i) { \
      EXPECT_EQ(host1.getAtFlatIndex<float>(i),        \
                host2.getAtFlatIndex<float>(i));       \
    }                                                  \
  }

TEST_F(TensorTest, RemoteCopy) {
  Runtime runtime(1, true);
  bool callbackCalled = false;
  // create a 8x2xfloat tensor mapped to 4 tiles
  HostTensor host1 = createHostTensor<float>(
      DistributedShape::createLinearlyDistributed({8, 2}, 4), Type::FLOAT32);

  RemoteTensor remote1 = host1.copyToRemote();
  Tensor tensor = remote1.copyToTile();
  tensor.copyToHost([&](const HostTensor &host2) {
    callbackCalled = true;
    EXPECT_EQ(host1.shape(), host2.shape());
    EXPECT_HOSTTENSOR_EQ(host1, host2);
  });

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, Print) {
  Runtime runtime(1, true);
  std::stringstream ss1;
  std::stringstream ss2;
  std::stringstream ss3;
  HostTensor host1 = createHostTensor<float>(
      DistributedShape::createLinearlyDistributed({8, 2}, 4), Type::FLOAT32);
  RemoteTensor remote1 = host1.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  HostTensor host2 = createHostTensor<float>(
      DistributedShape::createLinearlyDistributed({3, 3}, 2), Type::FLOAT32);
  RemoteTensor remote2 = host2.copyToRemote();
  Tensor tensor2 = remote2.copyToTile();

  tensor1.print("tensor", PrintTensorFormat(8), ss1);
  tensor1.print(
      "tensor",
      PrintTensorFormat(5, 0, PrintTensorFormat::FloatFormat::Fixed, 2), ss2);
  tensor2.print(
      "tensor",
      PrintTensorFormat(2, 0, PrintTensorFormat::FloatFormat::Fixed, 1), ss3);

  std::string expected1 = R"(tensor<8x2xfloat> tensor = [
  [0, 1]
  [2, 3]
  [4, 5]
  [6, 7]
  [8, 9]
  [10, 11]
  [12, 13]
  [14, 15]
]
)";

  std::string expected2 = R"(tensor<8x2xfloat> tensor = [
  [0, 1]
  [2, 3]
  ...
  [12, 13]
  [14, 15]
]
)";

  std::string expected3 = R"(tensor<3x3xfloat> tensor = [
  [0, ..., 2]
  ...
  [6, ..., 8]
]
)";

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  EXPECT_EQ(ss1.str(), expected1);
  EXPECT_EQ(ss2.str(), expected2);
  EXPECT_EQ(ss3.str(), expected3);
}

TEST_F(TensorTest, ReduceAcrossTilesSingleIPU) {
  Runtime runtime(1, true);
  bool callbackCalled = false;
  // create a 8x2xfloat tensor mapped to 4 tiles
  HostTensor host1 = createHostTensor<float>(
      DistributedShape::createLinearlyDistributed({8, 2}, 4), Type::FLOAT32);

  RemoteTensor remote1 = host1.copyToRemote();
  Tensor tensor = remote1.copyToTile();
  Tensor reduced = tensor.reduce(0, ReduceOperation::ADD);

  reduced.copyToHost([&](const HostTensor &host2) {
    callbackCalled = true;
    EXPECT_EQ(host2.numElements(), 2);
    EXPECT_EQ(host2.get<float>({0, 0}), 56);
    EXPECT_EQ(host2.get<float>({0, 1}), 64);
  });

  // check the shape of the reduced tensor
  EXPECT_EQ(reduced.shape(), DistributedShape::onSingleTile({1, 2}, 0));

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, ReduceAcrossTilesMultipleIPUs) {
  Runtime runtime(2, true);
  bool callbackCalled = false;
  // create a 8x2xfloat tensor mapped to 4 tiles across 2 IPUs
  HostTensor host1 = createHostTensor<float>(
      DistributedShape::onTiles({8, 2},
                                {{0, 2}, {1000, 2}, {1472, 2}, {2472, 2}}),
      Type::FLOAT32);

  RemoteTensor remote1 = host1.copyToRemote();
  Tensor tensor = remote1.copyToTile();
  Tensor reduced = tensor.reduce(0, ReduceOperation::ADD);

  reduced.copyToHost([&](const HostTensor &host2) {
    callbackCalled = true;
    EXPECT_EQ(host2.numElements(), 2);
    EXPECT_EQ(host2.get<float>({0, 0}), 56);
    EXPECT_EQ(host2.get<float>({0, 1}), 64);
  });

  // check the shape of the reduced tensor
  EXPECT_EQ(reduced.shape(), DistributedShape::onSingleTile({1, 2}, 0));

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, ReduceSameTiles) {
  Runtime runtime(1, true);
  bool callbackCalled = false;
  // create a 8x8xfloat tensor mapped to 4 tiles
  HostTensor host1 = createHostTensor<float>(
      DistributedShape::createLinearlyDistributed({8, 8}, 4), Type::FLOAT32);

  RemoteTensor remote1 = host1.copyToRemote();
  Tensor tensor = remote1.copyToTile();
  Tensor reduced = tensor.reduce(1, ReduceOperation::ADD);
  tensor.print("tensor");
  reduced.print("reduced");

  reduced.copyToHost([&](const HostTensor &host2) {
    callbackCalled = true;
    EXPECT_EQ(host2.numElements(), 8);
    EXPECT_EQ(host2.get<float>({0, 0}), 28);
    EXPECT_EQ(host2.get<float>({1, 0}), 92);
    EXPECT_EQ(host2.get<float>({2, 0}), 156);
    EXPECT_EQ(host2.get<float>({3, 0}), 220);
    EXPECT_EQ(host2.get<float>({4, 0}), 284);
    EXPECT_EQ(host2.get<float>({5, 0}), 348);
    EXPECT_EQ(host2.get<float>({6, 0}), 412);
    EXPECT_EQ(host2.get<float>({7, 0}), 476);
  });

  // check the shape of the reduced tensor
  EXPECT_EQ(reduced.shape(),
            DistributedShape::createLinearlyDistributed({8, 1}, 4));

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, DotProductSameShape) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create 2x3 vector fields (2 vectors with 3 components each)
  std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data2 = {2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 1.0f};

  DistributedShape shape = DistributedShape::onSingleTile({2, 3}, 0);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  HostTensor host1 =
      HostTensor::createPersistent(std::move(data1), shape, mapping, "vec1");
  HostTensor host2 =
      HostTensor::createPersistent(std::move(data2), shape, mapping, "vec2");

  RemoteTensor remote1 = host1.copyToRemote();
  RemoteTensor remote2 = host2.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  Tensor tensor2 = remote2.copyToTile();

  // Compute dot product
  Expression dotResult = ops::DotProduct(tensor1, tensor2);
  Tensor result = dotResult.materialize();

  result.copyToHost([&](const HostTensor &host3) {
    callbackCalled = true;
    EXPECT_EQ(host3.shape(), DistributedShape::onSingleTile({2, 1}, 0));
    // First vector: [1,2,3] · [2,3,4] = 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    EXPECT_FLOAT_EQ(host3.get<float>({0, 0}), 20.0f);
    // Second vector: [4,5,6] · [1,2,1] = 4*1 + 5*2 + 6*1 = 4 + 10 + 6 = 20
    EXPECT_FLOAT_EQ(host3.get<float>({1, 0}), 20.0f);
  });

  poplar::Engine engine1 = runtime.compileGraph();
  runtime.loadAndRunEngine(engine1);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, CrossProductSameShape) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create 2x3 vector fields (2 3D vectors)
  std::vector<float> data1 = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::vector<float> data2 = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  DistributedShape shape = DistributedShape::onSingleTile({2, 3}, 0);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  HostTensor host1 =
      HostTensor::createPersistent(std::move(data1), shape, mapping, "vec1");
  HostTensor host2 =
      HostTensor::createPersistent(std::move(data2), shape, mapping, "vec2");

  RemoteTensor remote1 = host1.copyToRemote();
  RemoteTensor remote2 = host2.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  Tensor tensor2 = remote2.copyToTile();

  // Compute cross product
  Expression crossResult = ops::CrossProduct(tensor1, tensor2);
  Tensor result = crossResult.materialize();

  result.copyToHost([&](const HostTensor &host3) {
    callbackCalled = true;
    EXPECT_EQ(host3.shape(), DistributedShape::onSingleTile({2, 3}, 0));
    // First cross product: [1,0,0] × [0,1,0] = [0,0,1]
    EXPECT_FLOAT_EQ(host3.get<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({0, 2}), 1.0f);
    // Second cross product: [0,1,0] × [0,0,1] = [1,0,0]
    EXPECT_FLOAT_EQ(host3.get<float>({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 1}), 0.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 2}), 0.0f);
  });

  poplar::Engine engine2 = runtime.compileGraph();
  runtime.loadAndRunEngine(engine2);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, ComplexExpressionWithMultipleOperations) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create vector fields for complex expression testing
  std::vector<float> vecA = {1.0f, 2.0f, 3.0f,
                             2.0f, 3.0f, 4.0f};  // 2 x 3 vectors
  std::vector<float> vecB = {4.0f, 5.0f, 6.0f,
                             1.0f, 2.0f, 1.0f};  // 2 x 3 vectors
  std::vector<float> vecC = {0.5f, 1.5f, 2.5f,
                             3.0f, 1.0f, 2.0f};  // 2 x 3 vectors

  DistributedShape vecShape = DistributedShape::onSingleTile({2, 3}, 0);
  DistributedShape scalarShape = DistributedShape::onSingleTile({1, 1}, 0);
  TileMapping vecMapping = TileMapping::linearMappingWithShape(vecShape);

  HostTensor hostA = HostTensor::createPersistent(std::move(vecA), vecShape,
                                                  vecMapping, "vecA");
  HostTensor hostB = HostTensor::createPersistent(std::move(vecB), vecShape,
                                                  vecMapping, "vecB");
  HostTensor hostC = HostTensor::createPersistent(std::move(vecC), vecShape,
                                                  vecMapping, "vecC");

  RemoteTensor remoteA = hostA.copyToRemote();
  RemoteTensor remoteB = hostB.copyToRemote();
  RemoteTensor remoteC = hostC.copyToRemote();

  Tensor tensorA = remoteA.copyToTile();
  Tensor tensorB = remoteB.copyToTile();
  Tensor tensorC = remoteC.copyToTile();

  // Complex expression: dot(cross(A, B) + C * 2.0f, A - B)
  // This combines: cross product, addition, scalar multiplication, subtraction,
  // and dot product
  Expression crossAB = ops::CrossProduct(tensorA, tensorB);
  Expression scaledC = tensorC * 2.0f;
  Expression sumCrossScaled = crossAB + scaledC;
  Expression diffAB = tensorA - tensorB;
  Expression finalResult = ops::DotProduct(sumCrossScaled, diffAB);

  Tensor result = finalResult.materialize();

  result.copyToHost([&](const HostTensor &hostResult) {
    callbackCalled = true;
    EXPECT_EQ(hostResult.shape(), DistributedShape::onSingleTile({2, 1}, 0));

    // Verify the computation manually for first vector:
    // A[0] = [1, 2, 3], B[0] = [4, 5, 6], C[0] = [0.5, 1.5, 2.5]
    // cross(A[0], B[0]) = [-3, 6, -3]
    // scaledC[0] = [1, 3, 5]
    // sum = [-2, 9, 2]
    // diff = [-3, -3, -3]
    // dot = (-2)*(-3) + 9*(-3) + 2*(-3) = 6 - 27 - 6 = -27
    EXPECT_FLOAT_EQ(hostResult.get<float>({0, 0}), -27.0f);

    // Verify the computation manually for second vector:
    // A[1] = [2, 3, 4], B[1] = [1, 2, 1], C[1] = [3, 1, 2]
    // cross(A[1], B[1]) = [3*1 - 4*2, 4*1 - 2*1, 2*2 - 3*1] = [-5, 2, 1]
    // scaledC[1] = [6, 2, 4]
    // sum = [1, 4, 5]
    // diff = [1, 1, 3]
    // dot = 1*1 + 4*1 + 5*3 = 1 + 4 + 15 = 20
    EXPECT_FLOAT_EQ(hostResult.get<float>({1, 0}), 20.0f);
  });

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, VectorProductReduction) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create a 4x3 tensor representing 4 vectors with 3 components each
  std::vector<float> vectorData = {
      1.0f, 2.0f, 3.0f,  // vector 1
      2.0f, 3.0f, 4.0f,  // vector 2
      3.0f, 4.0f, 5.0f,  // vector 3
      4.0f, 5.0f, 6.0f   // vector 4
  };

  DistributedShape shape = DistributedShape::onSingleTile({4, 3}, 0);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  HostTensor hostVectors = HostTensor::createPersistent(
      std::move(vectorData), shape, mapping, "vectors");

  RemoteTensor remoteVectors = hostVectors.copyToRemote();
  Tensor tensorVectors = remoteVectors.copyToTile();

  // Compute element-wise product of vectors and then reduce along vector
  // dimension
  Expression vectorProduct =
      tensorVectors * tensorVectors;  // Square each component
  Tensor productResult = vectorProduct.materialize();

  // Reduce along the component dimension (dimension 1) to get magnitude squared
  // for each vector
  Tensor magnitudeSquared = productResult.reduce(1, ReduceOperation::ADD);

  magnitudeSquared.copyToHost([&](const HostTensor &hostResult) {
    callbackCalled = true;
    EXPECT_EQ(hostResult.shape(), DistributedShape::onSingleTile({4, 1}, 0));

    // Verify magnitude squared for each vector:
    // Vector 1: 1² + 2² + 3² = 1 + 4 + 9 = 14
    EXPECT_FLOAT_EQ(hostResult.get<float>({0, 0}), 14.0f);
    // Vector 2: 2² + 3² + 4² = 4 + 9 + 16 = 29
    EXPECT_FLOAT_EQ(hostResult.get<float>({1, 0}), 29.0f);
    // Vector 3: 3² + 4² + 5² = 9 + 16 + 25 = 50
    EXPECT_FLOAT_EQ(hostResult.get<float>({2, 0}), 50.0f);
    // Vector 4: 4² + 5² + 6² = 16 + 25 + 36 = 77
    EXPECT_FLOAT_EQ(hostResult.get<float>({3, 0}), 77.0f);
  });

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, AdditionSameShape) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create 2x3 tensors for addition
  std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data2 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  DistributedShape shape = DistributedShape::onSingleTile({2, 3}, 0);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  HostTensor host1 =
      HostTensor::createPersistent(std::move(data1), shape, mapping, "tensor1");
  HostTensor host2 =
      HostTensor::createPersistent(std::move(data2), shape, mapping, "tensor2");

  RemoteTensor remote1 = host1.copyToRemote();
  RemoteTensor remote2 = host2.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  Tensor tensor2 = remote2.copyToTile();

  // Compute addition
  Expression addResult = tensor1 + tensor2;
  Tensor result = addResult.materialize();

  result.copyToHost([&](const HostTensor &host3) {
    callbackCalled = true;
    EXPECT_EQ(host3.shape(), shape);
    // Expected results: [11, 22, 33, 44, 55, 66]
    EXPECT_FLOAT_EQ(host3.get<float>({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({0, 2}), 33.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 0}), 44.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 1}), 55.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 2}), 66.0f);
  });

  poplar::Engine engine3 = runtime.compileGraph();
  runtime.loadAndRunEngine(engine3);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, AdditionWithBroadcasting) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create 1x3 tensor (for broadcasting) and 2x3 tensor
  std::vector<float> data1 = {100.0f, 200.0f, 300.0f};
  std::vector<float> data2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  DistributedShape shape1 = DistributedShape::onSingleTile({1, 3}, 0);
  DistributedShape shape2 = DistributedShape::onSingleTile({2, 3}, 0);
  TileMapping mapping1 = TileMapping::linearMappingWithShape(shape1);
  TileMapping mapping2 = TileMapping::linearMappingWithShape(shape2);

  HostTensor host1 = HostTensor::createPersistent(std::move(data1), shape1,
                                                  mapping1, "tensor1");
  HostTensor host2 = HostTensor::createPersistent(std::move(data2), shape2,
                                                  mapping2, "tensor2");

  RemoteTensor remote1 = host1.copyToRemote();
  RemoteTensor remote2 = host2.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  Tensor tensor2 = remote2.copyToTile();

  // Compute addition (should broadcast first tensor)
  Expression addResult = tensor1 + tensor2;
  Tensor result = addResult.materialize();

  result.copyToHost([&](const HostTensor &host3) {
    callbackCalled = true;
    EXPECT_EQ(host3.shape(), shape2);
    // Expected results: [101, 202, 303, 104, 205, 306]
    EXPECT_FLOAT_EQ(host3.get<float>({0, 0}), 101.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({0, 1}), 202.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({0, 2}), 303.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 0}), 104.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 1}), 205.0f);
    EXPECT_FLOAT_EQ(host3.get<float>({1, 2}), 306.0f);
  });

  poplar::Engine engine4 = runtime.compileGraph();
  runtime.loadAndRunEngine(engine4);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, AdditionAndSubtraction) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create 2x3 tensors for complex expression
  std::vector<float> data1 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  std::vector<float> data2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data3 = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  DistributedShape shape = DistributedShape::onSingleTile({2, 3}, 0);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  HostTensor host1 =
      HostTensor::createPersistent(std::move(data1), shape, mapping, "tensor1");
  HostTensor host2 =
      HostTensor::createPersistent(std::move(data2), shape, mapping, "tensor2");
  HostTensor host3 =
      HostTensor::createPersistent(std::move(data3), shape, mapping, "tensor3");

  RemoteTensor remote1 = host1.copyToRemote();
  RemoteTensor remote2 = host2.copyToRemote();
  RemoteTensor remote3 = host3.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  Tensor tensor2 = remote2.copyToTile();
  Tensor tensor3 = remote3.copyToTile();

  // Compute complex expression: (tensor1 + tensor2) - tensor3
  Expression complexResult = (tensor1 + tensor2) - tensor3;
  Tensor result = complexResult.materialize();

  result.copyToHost([&](const HostTensor &host4) {
    callbackCalled = true;
    EXPECT_EQ(host4.shape(), shape);
    // Expected results: (10+1-2, 20+2-4, 30+3-6, 40+4-8, 50+5-10, 60+6-12) =
    // [9, 18, 27, 36, 45, 54]
    EXPECT_FLOAT_EQ(host4.get<float>({0, 0}), 9.0f);
    EXPECT_FLOAT_EQ(host4.get<float>({0, 1}), 18.0f);
    EXPECT_FLOAT_EQ(host4.get<float>({0, 2}), 27.0f);
    EXPECT_FLOAT_EQ(host4.get<float>({1, 0}), 36.0f);
    EXPECT_FLOAT_EQ(host4.get<float>({1, 1}), 45.0f);
    EXPECT_FLOAT_EQ(host4.get<float>({1, 2}), 54.0f);
  });

  poplar::Engine engine5 = runtime.compileGraph();
  runtime.loadAndRunEngine(engine5);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(TensorTest, CrossAndDotProductSameShape) {
  Runtime runtime(1, true);
  bool callbackCalled = false;

  // Create 2x3 vector fields (2 3D vectors)
  std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> data2 = {11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

  DistributedShape shape = DistributedShape::onSingleTile({2, 3}, 0);
  TileMapping mapping = TileMapping::linearMappingWithShape(shape);

  HostTensor host1 =
      HostTensor::createPersistent(std::move(data1), shape, mapping, "vec1");
  HostTensor host2 =
      HostTensor::createPersistent(std::move(data2), shape, mapping, "vec2");

  RemoteTensor remote1 = host1.copyToRemote();
  RemoteTensor remote2 = host2.copyToRemote();
  Tensor tensor1 = remote1.copyToTile();
  Tensor tensor2 = remote2.copyToTile();

  // Compute dot product of cross product and second vector
  Expression resultExpr =
      ops::DotProduct(ops::CrossProduct(tensor1, tensor2), tensor2);
  Tensor result = resultExpr.materialize();

  result.copyToHost([&](const HostTensor &host3) {
    callbackCalled = true;
    EXPECT_EQ(host3.shape(), DistributedShape::onSingleTile({2, 1}, 0));
    // All cross products will yield 0 when dotted with the second vector
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 1; ++j) {
        EXPECT_FLOAT_EQ(host3.get<float>({i, j}), 0.0f);
      }
    }
  });

  poplar::Engine engine2 = runtime.compileGraph();
  runtime.loadAndRunEngine(engine2);
  EXPECT_TRUE(callbackCalled);
}
