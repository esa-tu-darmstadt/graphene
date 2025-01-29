#include <spdlog/spdlog.h>

#include <cstddef>
#include <memory>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Runtime.hpp"
#include "libtwofloat/twofloat.hpp"

using namespace graphene;

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);
  //   Runtime::enableProfiling();

  Runtime runtime(1);

  spdlog::info("Building data flow graph");

  // Tensor a({doubleword::from(1.1234567891), doubleword::from(1.1234567891),
  //           doubleword::from(1.1234567891)},
  //          {3});
  // Tensor b = a + a;
  // // Tensor c = a + b;

  // a.print("a");
  // b.print("b");

  Tensor a = Tensor::withInitialValue({1.0f, 2.0f, 3.0f},
                                      DistributedShape::onSingleTile({3, 1}));
  Tensor b = Tensor::withInitialValue({1.0f, 2.0f, 3.0f},
                                      DistributedShape::onSingleTile({1, 3}));

  Expression ab = a * b;
  Tensor c = ab;

  c.print("c");

  c.reduce(0, ReduceOperation::ADD).print("reduce(c, 0)");
  c.reduce(1, ReduceOperation::ADD).print("reduce(c, 1)");

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  spdlog::info("Done!");
  return 0;
}