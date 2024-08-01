#include "libgraphene/matrix/Norm.hpp"

#include <stdexcept>

namespace graphene {
VectorNorm parseVectorNorm(std::string const& norm) {
  if (norm == "L1") {
    return VectorNorm::L1;
  } else if (norm == "LINF") {
    return VectorNorm::LINF;
  } else if (norm == "L2") {
    return VectorNorm::L2;
  } else {
    throw std::runtime_error("Invalid norm: " + norm);
  }
}
std::string normToString(VectorNorm norm) {
  switch (norm) {
    case VectorNorm::L1:
      return "L1";
    case VectorNorm::L2:
      return "L2";
    case VectorNorm::LINF:
      return "LINF";
    case VectorNorm::None:
      return "None";
  }
  throw std::runtime_error("Invalid norm");
}
}  // namespace graphene