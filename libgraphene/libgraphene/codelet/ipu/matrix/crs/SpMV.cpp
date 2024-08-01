#include <cstddef>
#include <poplar/Vertex.hpp>

#include "poplar/InOutTypes.hpp"

using namespace poplar;

namespace graphene::matrix::crs {
template <typename value_t, typename rowptr_t, typename colind_t>
class MatrixVectorMultiply : public poplar::MultiVertex {
 public:
  poplar::Input<poplar::Vector<value_t>> diagCoeffs;
  poplar::Input<poplar::Vector<value_t>> offDiagCoeffs;

  poplar::Input<poplar::Vector<rowptr_t>> rowPtr;
  poplar::Input<poplar::Vector<colind_t>> colInd;

  poplar::Input<poplar::Vector<value_t>> x;
  poplar::Output<poplar::Vector<value_t>> result;

  /// \brief Compute the matrix-vector product \f$result = A\psi\f$.
  bool compute(unsigned workerId) {
    const size_t nCells = diagCoeffs.size();
    for (size_t cell = workerId; cell < nCells; cell += numWorkers()) {
      float res = diagCoeffs[cell] * x[cell];

      size_t start = rowPtr[cell];
      size_t end = rowPtr[cell + 1];

      // Assume that the number of non-zero elements per row is restricted to
      // 1000. This helps the compiler use more efficient loop instructions.
      __builtin_assume(end - start < 1000);

      for (size_t i = start; i < end; i++) {
        colind_t col = colInd[i];
        res += offDiagCoeffs[i] * x[col];
      }

      result[cell] = res;
    }
    return true;
  }
};

// Instantiate the template for float and all combinations of uint8_t, uint16_t,
// and uint32_t for rowptr_t and colind_t.
template class MatrixVectorMultiply<float, unsigned char, unsigned char>;
template class MatrixVectorMultiply<float, unsigned char, unsigned short>;
template class MatrixVectorMultiply<float, unsigned char, unsigned int>;
template class MatrixVectorMultiply<float, unsigned short, unsigned char>;
template class MatrixVectorMultiply<float, unsigned short, unsigned short>;
template class MatrixVectorMultiply<float, unsigned short, unsigned int>;
template class MatrixVectorMultiply<float, unsigned int, unsigned char>;
template class MatrixVectorMultiply<float, unsigned int, unsigned short>;
template class MatrixVectorMultiply<float, unsigned int, unsigned int>;

}  // namespace graphene::matrix::crs