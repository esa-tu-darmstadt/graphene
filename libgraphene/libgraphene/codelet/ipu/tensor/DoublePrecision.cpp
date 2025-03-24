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

#include <cstddef>
#include <ipu_memory_intrinsics>
#include <poplar/InOutTypes.hpp>
#include <poplar/Vertex.hpp>

#include "libtwofloat/twofloat.hpp"
#include "poplar/InOutTypes.hpp"

using namespace poplar;
using namespace twofloat;

namespace graphene::ops {
class CastDoublePrecisionToFloatVertex : public MultiVertex {
 public:
  Input<Vector<long long>> in;
  Output<Vector<float, VectorLayout::SPAN, 8>> out;

  bool compute(unsigned workerId) {
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      double value = *reinterpret_cast<const double*>(&in[i]);
      out[i] = (float)value;
    }
    return true;
  }
};

class CastFloatToDoublePrecisionVertex : public MultiVertex {
 public:
  Input<Vector<float, VectorLayout::SPAN, 8>> in;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      *reinterpret_cast<double*>(&out[i]) = (double)in[i];
    }
    return true;
  }
};

template <typename lhs_poplar_t, typename rhs_poplar_t, typename out_poplar_t>
struct AddDoublePrecision {};

template <>
struct AddDoublePrecision<long long, float, long long> : public MultiVertex {
  Input<Vector<long long>> lhs;
  Input<Vector<float>> rhs;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    bool broadcastLeft = lhs.size() == 1;
    bool broadcastRight = rhs.size() == 1;
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      double lhs_val =
          *reinterpret_cast<const double*>(&lhs[broadcastLeft ? 0 : i]);
      float rhs_val = rhs[broadcastRight ? 0 : i];

      double out_val = lhs_val + (double)rhs_val;
      out[i] = *reinterpret_cast<long long*>(&out_val);
    }
    return true;
  }
};

template <>
struct AddDoublePrecision<float, long long, long long> : public MultiVertex {
  Input<Vector<float>> lhs;
  Input<Vector<long long>> rhs;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    bool broadcastLeft = lhs.size() == 1;
    bool broadcastRight = rhs.size() == 1;
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      float lhs_val = lhs[broadcastLeft ? 0 : i];
      double rhs_val =
          *reinterpret_cast<const double*>(&rhs[broadcastRight ? 0 : i]);

      double out_val = (double)lhs_val + rhs_val;
      out[i] = *reinterpret_cast<long long*>(&out_val);
    }
    return true;
  }
};

template <>
struct AddDoublePrecision<long long, long long, long long>
    : public MultiVertex {
  Input<Vector<long long>> lhs;
  Input<Vector<long long>> rhs;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    bool broadcastLeft = lhs.size() == 1;
    bool broadcastRight = rhs.size() == 1;
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      double lhs_val =
          *reinterpret_cast<const double*>(&lhs[broadcastLeft ? 0 : i]);
      double rhs_val =
          *reinterpret_cast<const double*>(&rhs[broadcastRight ? 0 : i]);

      double out_val = lhs_val + rhs_val;

      out[i] = *reinterpret_cast<long long*>(&out_val);
    }
    return true;
  }
};

}  // namespace graphene::ops