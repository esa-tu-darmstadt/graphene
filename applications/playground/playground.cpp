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