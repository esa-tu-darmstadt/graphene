#include <cstddef>
#include <libtwofloat/arithmetics/double-word-arithmetic.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/InOutTypes.hpp"

using namespace poplar;
using namespace twofloat;

namespace graphene::matrix::crs {
template <typename rowptr_t, typename colind_t>
class ResidualMixedPrecisionDoublewordToFloat : public poplar::MultiVertex {
 public:
  poplar::Input<poplar::Vector<float>> diagCoeffs;
  poplar::Input<poplar::Vector<float>> offDiagCoeffs;

  poplar::Input<poplar::Vector<rowptr_t>> rowPtr;
  poplar::Input<poplar::Vector<colind_t>> colInd;

  poplar::Input<poplar::Vector<long long>> x;
  poplar::Input<poplar::Vector<float>> b;
  poplar::Output<poplar::Vector<float>> result;

  bool compute(unsigned workerId) {
    const size_t nCells = diagCoeffs.size();
    for (size_t row = workerId; row < nCells; row += numWorkers()) {
      two<float> rowValue = *reinterpret_cast<const two<float> *>(&x[row]);
      two<float> res = doubleword::sub(
          b[row], doubleword::mul<doubleword::Mode::Accurate, false>(
                      diagCoeffs[row], rowValue));

      size_t start = rowPtr[row];
      size_t end = rowPtr[row + 1];

      // Assume that the number of non-zero elements per row is restricted to
      // 1000. This helps the compiler use more efficient loop instructions.
      __builtin_assume(end - start < 1000);

      for (size_t i = start; i < end; i++) {
        colind_t col = colInd[i];
        two<float> colValue = *reinterpret_cast<const two<float> *>(&x[col]);
        auto flux = doubleword::mul<doubleword::Mode::Accurate, false>(
            offDiagCoeffs[i], colValue);
        res = doubleword::sub<doubleword::Mode::Accurate>(res, flux);
      }

      result[row] = res.eval();
    }
    return true;
  }
};

template <typename rowptr_t, typename colind_t>
class ResidualMixedPrecisionDoubleToFloat : public poplar::MultiVertex {
 public:
  poplar::Input<poplar::Vector<float>> diagCoeffs;
  poplar::Input<poplar::Vector<float>> offDiagCoeffs;

  poplar::Input<poplar::Vector<rowptr_t>> rowPtr;
  poplar::Input<poplar::Vector<colind_t>> colInd;

  poplar::Input<poplar::Vector<long long>> x;
  poplar::Input<poplar::Vector<float>> b;
  poplar::Output<poplar::Vector<float>> result;

  bool compute(unsigned workerId) {
    const size_t nCells = diagCoeffs.size();
    for (size_t row = workerId; row < nCells; row += numWorkers()) {
      double rowValue = *reinterpret_cast<const double *>(&x[row]);
      double res = (double)b[row] - (double)diagCoeffs[row] * rowValue;

      size_t start = rowPtr[row];
      size_t end = rowPtr[row + 1];

      // Assume that the number of non-zero elements per row is restricted to
      // 1000. This helps the compiler use more efficient loop instructions.
      __builtin_assume(end - start < 1000);

      for (size_t i = start; i < end; i++) {
        colind_t col = colInd[i];
        double colValue = *reinterpret_cast<const double *>(&x[col]);
        double flux = (double)offDiagCoeffs[i] * colValue;
        res -= flux;
      }

      result[row] = (float)res;
    }
    return true;
  }
};

// Instantiate the template for float and all combinations of uint8_t, uint16_t,
// and uint32_t for rowptr_t and colind_t.
#define INSTANTIATE_1()          \
  INSTANTIATE_2(unsigned char);  \
  INSTANTIATE_2(unsigned short); \
  INSTANTIATE_2(unsigned int);

#define INSTANTIATE_2(rowptr_t)            \
  INSTANTIATE_3(rowptr_t, unsigned char);  \
  INSTANTIATE_3(rowptr_t, unsigned short); \
  INSTANTIATE_3(rowptr_t, unsigned int);

#define INSTANTIATE_3(rowptr_t, colind_t)                                     \
  template class ResidualMixedPrecisionDoublewordToFloat<rowptr_t, colind_t>; \
  template class ResidualMixedPrecisionDoubleToFloat<rowptr_t, colind_t>;

INSTANTIATE_1()

#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3

}  // namespace graphene::matrix::crs