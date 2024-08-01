#include "libgraphene/util/Tracepoint.hpp"

#include <pvti/pvti.hpp>

namespace graphene {
pvti::TraceChannel traceLibFvm("libfvm");
}