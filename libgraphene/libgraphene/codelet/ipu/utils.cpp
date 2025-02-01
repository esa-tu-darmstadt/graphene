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