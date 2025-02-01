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

#include <print.h>

#include <StackSizeDefs.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

extern int __supervisor_stack_max_growth_plus_one;
extern int __worker_stack_max_growth_plus_one;

namespace sim {
namespace codelet {
class StackReservationWorker : public MultiVertex {
 public:
  bool compute(unsigned workerID) { return true; }
};

class StackReservationSupervisor : public SupervisorVertex {
 public:
  __attribute__((target("supervisor"))) bool compute() { return true; }
};

class TestVertex {
 public:
  float* __restrict__ in;
  float* __restrict__ out;
  size_t size;
  bool compute();
};

bool TestVertex::compute() {
  for (unsigned i = 0; i < size; i += 2) {
    out[i] = in[i];
    out[i + 1] = in[i];
  }
  return true;
}

}  // namespace codelet
}  // namespace sim

// clang-format off
DEF_STACK_USAGE(512, __runCodelet_sim__codelet__StackReservationWorker);
DEF_STACK_USAGE(512, __runCodelet_sim__codelet__StackReservationSupervisor);