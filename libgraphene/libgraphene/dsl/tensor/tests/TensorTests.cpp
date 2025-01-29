#include <gtest/gtest.h>

#include <sstream>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
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
  Runtime runtime(1);
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
  Runtime runtime(1);
  std::stringstream ss;
  HostTensor host1 = createHostTensor<float>(
      DistributedShape::createLinearlyDistributed({8, 2}, 4), Type::FLOAT32);
  RemoteTensor remote1 = host1.copyToRemote();
  Tensor tensor = remote1.copyToTile();
  tensor.print("tensor", {}, ss);

  std::string expected = R"(tensor<8x2xfloat> tensor = [
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

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  EXPECT_EQ(ss.str(), expected);
}

TEST_F(TensorTest, ReduceAcrossTilesSingleIPU) {
  Runtime runtime(1);
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
  Runtime runtime(2);
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
  Runtime runtime(1);
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