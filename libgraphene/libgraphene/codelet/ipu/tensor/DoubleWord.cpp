

#include <cstddef>
#include <ipu_memory_intrinsics>
#include <poplar/InOutTypes.hpp>
#include <poplar/Vertex.hpp>

#include "libtwofloat/twofloat.hpp"
#include "poplar/InOutTypes.hpp"
// #include "libtwofloat/archs/ipu/ipu.hpp"
#include <libtwofloat/arithmetics/double-word-arithmetic.hpp>

using namespace poplar;
using namespace twofloat;

namespace graphene::ops {
class UnrollDoubleWordVertex : public MultiVertex {
 public:
  Input<Vector<long long>> in;
  Output<Vector<float>> out;

  bool compute(unsigned workerId) {
    size_t inChunkSize = in.size() / numWorkers();
    size_t inStart = workerId * inChunkSize;
    size_t inEnd = (workerId + 1) * inChunkSize;

    if (workerId == numWorkers() - 1) {
      inEnd = in.size();
    }
    // Remember that inT is twice the size of outT
    memcpy(&out[inStart * 2], &in[inStart],
           (inEnd - inStart) * sizeof(long long));
    return true;
  }
};

class CastDoubleWordToFloatVertex : public MultiVertex {
 public:
  Input<Vector<long long>> in;
  Output<Vector<float, VectorLayout::SPAN, 8>> out;

  bool compute(unsigned workerId) {
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      two<float> value = *reinterpret_cast<const two<float>*>(&in[i]);
      out[i] = value.eval();
    }
    return true;
  }
};

class CastFloatToDoubleWordVertex : public MultiVertex {
 public:
  Input<Vector<float, VectorLayout::SPAN, 8>> in;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      two<float> value(in[i]);
      *reinterpret_cast<two<float>*>(&out[i]) = value;
    }
    return true;
  }
};

template <typename lhs_poplar_t, typename rhs_poplar_t, typename out_poplar_t>
struct AddDoubleWord {};

template <>
struct AddDoubleWord<long long, float, long long> : public MultiVertex {
  Input<Vector<long long>> lhs;
  Input<Vector<float>> rhs;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    bool broadcastLeft = lhs.size() == 1;
    bool broadcastRight = rhs.size() == 1;
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      two<float> lhs_val =
          *reinterpret_cast<const two<float>*>(&lhs[broadcastLeft ? 0 : i]);
      float rhs_val = rhs[broadcastRight ? 0 : i];

      two<float> out_val = twofloat::doubleword::add(lhs_val, rhs_val);
      out[i] = *reinterpret_cast<long long*>(&out_val);
    }
    return true;
  }
};

template <>
struct AddDoubleWord<float, long long, long long> : public MultiVertex {
  Input<Vector<float>> lhs;
  Input<Vector<long long>> rhs;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    bool broadcastLeft = lhs.size() == 1;
    bool broadcastRight = rhs.size() == 1;
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      float lhs_val = lhs[broadcastLeft ? 0 : i];
      two<float> rhs_val =
          *reinterpret_cast<const two<float>*>(&rhs[broadcastRight ? 0 : i]);

      two<float> out_val = twofloat::doubleword::add(lhs_val, rhs_val);
      out[i] = *reinterpret_cast<long long*>(&out_val);
    }
    return true;
  }
};

template <>
struct AddDoubleWord<long long, long long, long long> : public MultiVertex {
  Input<Vector<long long>> lhs;
  Input<Vector<long long>> rhs;
  Output<Vector<long long>> out;

  bool compute(unsigned workerId) {
    bool broadcastLeft = lhs.size() == 1;
    bool broadcastRight = rhs.size() == 1;
    for (size_t i = workerId; i < out.size(); i += numWorkers()) {
      two<float> lhs_val =
          *reinterpret_cast<const two<float>*>(&lhs[broadcastLeft ? 0 : i]);
      two<float> rhs_val =
          *reinterpret_cast<const two<float>*>(&rhs[broadcastRight ? 0 : i]);

      two<float> out_val =
          twofloat::doubleword::add<twofloat::doubleword::Mode::Accurate>(
              lhs_val, rhs_val);

      out[i] = *reinterpret_cast<long long*>(&out_val);
    }
    return true;
  }
};

}  // namespace graphene::ops