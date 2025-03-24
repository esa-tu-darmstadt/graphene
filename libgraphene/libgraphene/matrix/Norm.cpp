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