#include <print.h>

#include <StackSizeDefs.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

extern int __supervisor_stack_max_growth_plus_one;
extern int __worker_stack_max_growth_plus_one;

namespace sim {
namespace codelet {
class StackReservation : public Vertex {
 public:
  bool compute() { return true; }
};

}  // namespace codelet
}  // namespace sim

// clang-format off
DEF_STACK_USAGE(200, __runCodelet_sim__codelet__StackReservation);