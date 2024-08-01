#include <cstddef>
#include <ipu_memory_intrinsics>
#include <poplar/InOutTypes.hpp>
#include <poplar/Vertex.hpp>

#include "ipu-thread-sync/ipu-thread-sync.hpp"

using namespace poplar;

namespace graphene::matrix::solver::gaussseidel {
template <typename value_t, typename rowptr_t, typename colind_t>
class GausSeidelSolveSmoothCRS : public MultiVertex {
 public:
  Input<Vector<value_t>> diagCoeffs;
  Input<Vector<value_t>> offDiagCoeffs;

  Input<Vector<rowptr_t>> rowPtr;
  Input<Vector<colind_t>> colInd;

  InOut<Vector<value_t>> x;
  Input<Vector<value_t>> b;

  bool compute(unsigned workerId) {
    const size_t nCells = diagCoeffs.size();
    for (size_t row = workerId; row < nCells; row += numWorkers()) {
      float xi = b[row];

      size_t start = rowPtr[row];
      size_t end = rowPtr[row + 1];

      // Assume that the number of non-zero elements per row is restricted to
      // 1000. This helps the compiler use more efficient loop instructions.
      __builtin_assume(end - start < 1000);

      for (size_t i = start; i < end; i++) {
        colind_t col = colInd[i];
        xi -= offDiagCoeffs[i] * x[col];
      }

      x[row] = xi / diagCoeffs[row];
    }
    return true;
  }
};

// Instantiate the vertex for all combinations of supported data types
#define INSTANTIATE_1(value_t)            \
  INSTANTIATE_2(value_t, unsigned char);  \
  INSTANTIATE_2(value_t, unsigned short); \
  INSTANTIATE_2(value_t, unsigned int);

#define INSTANTIATE_2(value_t, rowptr_t)            \
  INSTANTIATE_3(value_t, rowptr_t, unsigned char);  \
  INSTANTIATE_3(value_t, rowptr_t, unsigned short); \
  INSTANTIATE_3(value_t, rowptr_t, unsigned int);

#define INSTANTIATE_3(value_t, rowptr_t, colind_t) \
  template class GausSeidelSolveSmoothCRS<value_t, rowptr_t, colind_t>;

INSTANTIATE_1(float);
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3

template <typename value_t, typename rowptr_t, typename colind_t,
          typename colorsortaddr_t, typename colorsortstartaddr_t>
class GausSeidelSolveSmoothCRSMulticolor : public SupervisorVertex {
  using ThisType =
      GausSeidelSolveSmoothCRSMulticolor<value_t, rowptr_t, colind_t,
                                         colorsortaddr_t, colorsortstartaddr_t>;

 private:
  volatile uint8_t currentColor_;
  size_t nRowsWithoutHalo;

 public:
  Input<Vector<value_t>> diagCoeffs;
  Input<Vector<value_t>> offDiagCoeffs;

  Input<Vector<rowptr_t>> rowPtr;
  Input<Vector<colind_t>> colInd;

  InOut<Vector<value_t>> x;
  Input<Vector<value_t>> b;

  Input<Vector<colorsortaddr_t>> colorSortAddr;
  Input<Vector<colorsortstartaddr_t>> colorSortStartAddr;

  SUPERVISOR_FUNC bool compute() {
    // Compute each color in parallel on all workers
    // Syncing all workers before and not after starting the workers is
    // beneficial because the supervisor can prepare the next color while some
    // of the workers are still working on the previous color.
    nRowsWithoutHalo = diagCoeffs.size();
    for (size_t colorI = 0; colorI < colorSortStartAddr.size() - 1; colorI++) {
      currentColor_ = colorI;
      ipu::syncAndStartOnAllWorkers<ThisType, &ThisType::computeColor>(this);
    }

    // Make sure all workers are done before returning
    ipu::syncAllWorkers();

    return true;
  }

  WORKER_FUNC bool computeColor(unsigned threadId) {
    // Copy the current color from the volatile variable to a local variable so
    // the compiler can optimize the code better.
    uint8_t currentColor = currentColor_;

    size_t colorSortStart = colorSortStartAddr[currentColor];
    size_t colorSortEnd = colorSortStartAddr[currentColor + 1];

    // Iterate over all rows of the matrix
    for (size_t colorSortI = colorSortStart + threadId;
         colorSortI < colorSortEnd;
         colorSortI += poplar::MultiVertex::numWorkers()) {
      size_t row = colorSortAddr[colorSortI];
      float xi = b[row];

      size_t start = rowPtr[row];
      size_t end = rowPtr[row + 1];

      // Assume that the number of non-zero elements per row is restricted to
      // 1000. This helps the compiler use more efficient loop instructions.
      __builtin_assume(end - start < 1000);

      for (size_t i = start; i < end; i++) {
        colind_t col = colInd[i];
        xi -= offDiagCoeffs[i] * x[col];
      }

      x[row] = xi / diagCoeffs[row];
    }

    return true;
  }
};

// Instantiate the vertex for all combinations of supported data types
#define INSTANTIATE_1(value_t)            \
  INSTANTIATE_2(value_t, unsigned char);  \
  INSTANTIATE_2(value_t, unsigned short); \
  INSTANTIATE_2(value_t, unsigned int);

#define INSTANTIATE_2(value_t, rowptr_t)            \
  INSTANTIATE_3(value_t, rowptr_t, unsigned char);  \
  INSTANTIATE_3(value_t, rowptr_t, unsigned short); \
  INSTANTIATE_3(value_t, rowptr_t, unsigned int);

#define INSTANTIATE_3(value_t, rowptr_t, colind_t)            \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned char);  \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned short); \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned int);

#define INSTANTIATE_4(value_t, rowptr_t, colind_t, colorsortaddr_t)            \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned char);  \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned short); \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned int);

#define INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, \
                      colorsortstartaddr_t)                         \
  template class GausSeidelSolveSmoothCRSMulticolor<                \
      value_t, rowptr_t, colind_t, colorsortaddr_t, colorsortstartaddr_t>;

INSTANTIATE_1(float);
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3
#undef INSTANTIATE_4
#undef INSTANTIATE_5

}  // namespace graphene::matrix::solver::gaussseidel