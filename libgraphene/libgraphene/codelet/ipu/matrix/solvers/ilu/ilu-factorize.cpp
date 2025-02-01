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

#include <print.h>

#include <StackSizeDefs.hpp>
#include <poplar/Vertex.hpp>

#include "ipu-thread-sync/ipu-thread-sync.hpp"
#include "poplar/InOutTypes.hpp"

using namespace poplar;

namespace graphene::matrix::solver::ilu {

// ------------------------------------------------------------
// DILU factorization, sequential
// ------------------------------------------------------------
template <typename value_t, typename rowptr_t, typename colind_t>
class ILUFactorizeCRSDiagonal : public poplar::Vertex {
 public:
  InOut<Vector<value_t, VectorLayout::SPAN, 8>> diagCoeffs;
  Input<Vector<value_t, VectorLayout::SPAN, 8>> offDiagCoeffs;

  Input<Vector<rowptr_t, VectorLayout::SPAN, 8>> rowPtr;
  Input<Vector<colind_t, VectorLayout::SPAN, 8>> colInd;

  value_t getOffDiagValue(size_t i, size_t j) {
    size_t start = rowPtr[i];
    size_t end = rowPtr[i + 1];
    __builtin_assume(end - start < 1000);
    for (rowptr_t a = start; a < end; a++) {
      colind_t jCurrent = colInd[a];
      if (jCurrent == j) {
        return offDiagCoeffs[a];
      }
    }
    return 0;
  }

  // Calculates the DIC diagonal
  bool compute() {
    const auto nRowsWithoutHalo = diagCoeffs.size();

    // Iterate over all rows of the matrix
    for (size_t i = 0; i < nRowsWithoutHalo; i++) {
      value_t &a_ii = diagCoeffs[i];
      value_t temp = a_ii;

      // Iterate over lower triangular part of the matrix
      // => for every i > j
      size_t start = rowPtr[i];
      size_t end = rowPtr[i + 1];
      __builtin_assume(end - start < 1000);
      for (size_t a = start; a < end; a++) {
        size_t j = colInd[a];

        // Stop if we reach the diagonal, we only need the lower triangular part
        if (j > i) break;

        // Ignore halo columns
        if (j >= nRowsWithoutHalo) continue;

        value_t a_ij = offDiagCoeffs[a];
        value_t a_ji = getOffDiagValue(j, i);
        value_t a_jj = diagCoeffs[j];
        temp -= a_ij * a_ji / a_jj;
      }
      a_ii = temp;
    }

    return true;
  }
};

#define INSTANTIATE() INSTANTIATE_1(float);
#define INSTANTIATE_1(value_t)            \
  INSTANTIATE_2(value_t, unsigned int);   \
  INSTANTIATE_2(value_t, unsigned short); \
  INSTANTIATE_2(value_t, unsigned char);
#define INSTANTIATE_2(value_t, rowptr_t)            \
  INSTANTIATE_3(value_t, rowptr_t, unsigned int);   \
  INSTANTIATE_3(value_t, rowptr_t, unsigned short); \
  INSTANTIATE_3(value_t, rowptr_t, unsigned char);
#define INSTANTIATE_3(value_t, rowptr_t, colind_t) \
  template class ILUFactorizeCRSDiagonal<value_t, rowptr_t, colind_t>;

INSTANTIATE();
#undef INSTANTIATE
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3

// ------------------------------------------------------------
// DILU factorization, multicolor
// ------------------------------------------------------------
template <typename value_t, typename rowptr_t, typename colind_t,
          typename colorsortaddr_t, typename colorsortstartaddr_t>
class ILUFactorizeCRSDiagonalMulticolor : public SupervisorVertex {
  using ThisType =
      ILUFactorizeCRSDiagonalMulticolor<value_t, rowptr_t, colind_t,
                                        colorsortaddr_t, colorsortstartaddr_t>;

 private:
  volatile uint8_t currentColor_;
  size_t nRowsWithoutHalo;

 public:
  InOut<Vector<value_t, VectorLayout::SPAN, 8>> diagCoeffs;
  Input<Vector<value_t, VectorLayout::SPAN, 8>> offDiagCoeffs;

  Input<Vector<rowptr_t, VectorLayout::SPAN, 8>> rowPtr;
  Input<Vector<colind_t, VectorLayout::SPAN, 8>> colInd;

  Input<Vector<colorsortaddr_t, VectorLayout::SPAN, 8>> colorSortAddr;
  Input<Vector<colorsortstartaddr_t, VectorLayout::SPAN, 8>> colorSortStartAddr;

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

  WORKER_FUNC value_t getOffDiagValue(size_t i, size_t j) {
    size_t start = rowPtr[i];
    size_t end = rowPtr[i + 1];
    __builtin_assume(end - start < 1000);
    for (rowptr_t a = start; a < end; a++) {
      colind_t jCurrent = colInd[a];
      if (jCurrent == j) {
        return offDiagCoeffs[a];
      }
    }
    return 0;
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
      auto i = colorSortAddr[colorSortI];
      value_t &a_ii = diagCoeffs[i];
      value_t temp = a_ii;

      // Iterate over lower triangular part of the matrix
      // => for every i > j
      size_t start = rowPtr[i];
      size_t end = rowPtr[i + 1];
      __builtin_assume(end - start < 1000);
      for (size_t a = start; a < end; a++) {
        size_t j = colInd[a];

        // Stop if we reach the diagonal, we only need the lower triangular part
        if (j > i) break;

        // Ignore halo columns
        if (j >= nRowsWithoutHalo) continue;

        value_t a_ij = offDiagCoeffs[a];
        value_t a_ji = getOffDiagValue(j, i);
        value_t a_jj = diagCoeffs[j];
        temp -= a_ij * a_ji / a_jj;
      }
      a_ii = temp;
    }

    return true;
  }
};

// Instantiate the template for all possible combinations of the template
// parameters
#define INSTANTIATE() INSTANTIATE_1(float);
#define INSTANTIATE_1(value_t)            \
  INSTANTIATE_2(value_t, unsigned int);   \
  INSTANTIATE_2(value_t, unsigned short); \
  INSTANTIATE_2(value_t, unsigned char);
#define INSTANTIATE_2(value_t, rowptr_t)            \
  INSTANTIATE_3(value_t, rowptr_t, unsigned int);   \
  INSTANTIATE_3(value_t, rowptr_t, unsigned short); \
  INSTANTIATE_3(value_t, rowptr_t, unsigned char);
#define INSTANTIATE_3(value_t, rowptr_t, colind_t)            \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned int);   \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned short); \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned char);
#define INSTANTIATE_4(value_t, rowptr_t, colind_t, colorsortaddr_t)            \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned int);   \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned short); \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned char);
#define INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, \
                      colorsortstartaddr_t)                         \
  template class ILUFactorizeCRSDiagonalMulticolor<                 \
      value_t, rowptr_t, colind_t, colorsortaddr_t, colorsortstartaddr_t>;

INSTANTIATE();

#undef INSTANTIATE
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3
#undef INSTANTIATE_4
#undef INSTANTIATE_5

// ------------------------------------------------------------
// ILU(0) factorization, sequential
// ------------------------------------------------------------
template <typename value_t, typename rowptr_t, typename colind_t>
class ILUFactorizeCRS : public poplar::Vertex {
 public:
  InOut<Vector<value_t, VectorLayout::SPAN, 8>> diagCoeffs;
  InOut<Vector<value_t, VectorLayout::SPAN, 8>> offDiagCoeffs;

  Input<Vector<rowptr_t, VectorLayout::SPAN, 8>> rowPtr;
  Input<Vector<colind_t, VectorLayout::SPAN, 8>> colInd;

  // Helper function to get an off-diagonal value
  value_t getOffDiagValue(size_t i, size_t j) {
    size_t start = rowPtr[i];
    size_t end = rowPtr[i + 1];
    __builtin_assume(end - start < 1000);
    for (size_t a = start; a < end; a++) {
      auto jCurrent = colInd[a];
      if (jCurrent == j) {
        return offDiagCoeffs[a];
      }
    }
    return 0;
  }

  bool compute() {
    const auto nRowsWithoutHalo = diagCoeffs.size();

    // This is algorithm 10.4 in "Iterative Methods for Sparse Linear Systems"
    // The coefficients must be initialized with the original matrix A

    // Main loop of ILU(0) factorization
    for (size_t i = 1; i < nRowsWithoutHalo; i++) {
      // Update the i-th row

      // for k = 1, ..., i - 1
      // => iterate over the lower triangular part of the current row
      size_t iStart = rowPtr[i];
      size_t iEnd = rowPtr[i + 1];
      __builtin_assume(iEnd - iStart < 1000);
      for (size_t ik = iStart; ik < iEnd; ik++) {
        size_t k = colInd[ik];
        if (k >= i) break;

        // We don't need to check for halo coefficients in the lower triangular
        // part (due to k < i)

        value_t a_kk = diagCoeffs[k];
        value_t &a_ik = offDiagCoeffs[ik];
        a_ik = a_ik / a_kk;

        // The algorithm requires us to iterate over
        // j = k + 1, ..., n
        // Due to our matrix layout, we iterate over:
        // 1. j = k + 1, ..., n with j != i (off-diagonal)
        // 2. j = i (diagonal)

        // For j = k + 1, ..., n, j != i (off-diagonal)
        for (size_t ij = ik + 1; ij < iEnd; ij++) {
          auto j = colInd[ij];
          // Stop at halo coefficients
          if (j >= nRowsWithoutHalo) break;
          value_t a_kj = getOffDiagValue(k, j);
          value_t &a_ij = offDiagCoeffs[ij];
          a_ij -= a_ik * a_kj;
        }

        // For j = i (diagonal)
        value_t a_ki = getOffDiagValue(k, i);
        value_t &a_ii = diagCoeffs[i];
        a_ii -= a_ik * a_ki;
      }
    }

    return true;
  }
};

#define INSTANTIATE() INSTANTIATE_1(float);
#define INSTANTIATE_1(value_t)            \
  INSTANTIATE_2(value_t, unsigned int);   \
  INSTANTIATE_2(value_t, unsigned short); \
  INSTANTIATE_2(value_t, unsigned char);
#define INSTANTIATE_2(value_t, rowptr_t)            \
  INSTANTIATE_3(value_t, rowptr_t, unsigned int);   \
  INSTANTIATE_3(value_t, rowptr_t, unsigned short); \
  INSTANTIATE_3(value_t, rowptr_t, unsigned char);
#define INSTANTIATE_3(value_t, rowptr_t, colind_t) \
  template class ILUFactorizeCRS<value_t, rowptr_t, colind_t>;

INSTANTIATE();
#undef INSTANTIATE
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3

// ------------------------------------------------------------
// ILU(0) factorization, multicolor
// ------------------------------------------------------------
template <typename value_t, typename rowptr_t, typename colind_t,
          typename colorsortaddr_t, typename colorsortstartaddr_t>
class ILUFactorizeCRSMulticolor : public SupervisorVertex {
  using ThisType =
      ILUFactorizeCRSMulticolor<value_t, rowptr_t, colind_t, colorsortaddr_t,
                                colorsortstartaddr_t>;

  volatile uint8_t currentColor_;  // Current color being processed
  size_t nRowsWithoutHalo;  // Number of rows in the matrix (excluding halo)

  InOut<Vector<value_t>> diagCoeffs;
  InOut<Vector<value_t>> offDiagCoeffs;

  Input<Vector<rowptr_t>> rowPtr;
  Input<Vector<colind_t>> colInd;

  Input<Vector<colorsortaddr_t>> colorSortAddr;
  Input<Vector<colorsortstartaddr_t>> colorSortStartAddr;

 public:
  // Supervisor function to manage the computation across all colors
  SUPERVISOR_FUNC bool compute() {
    nRowsWithoutHalo = diagCoeffs.size();
    // Process each color sequentially
    for (size_t colorI = 0; colorI < colorSortStartAddr.size() - 1; colorI++) {
      currentColor_ = colorI;
      // Start the computation for the current color on all workers
      ipu::syncAndStartOnAllWorkers<ThisType, &ThisType::computeColor>(this);
    }

    // Ensure all workers have completed before returning
    ipu::syncAllWorkers();

    return true;
  }

  // Helper function to get an off-diagonal value
  value_t getOffDiagValue(size_t i, size_t j) {
    size_t start = rowPtr[i];
    size_t end = rowPtr[i + 1];
    __builtin_assume(end - start < 1000);
    for (size_t a = start; a < end; a++) {
      auto jCurrent = colInd[a];
      if (jCurrent == j) {
        return offDiagCoeffs[a];
      }
    }
    return 0;
  }

  // Worker function to compute ILU(0) for a specific color
  WORKER_FUNC bool computeColor(unsigned threadId) {
    uint8_t currentColor = currentColor_;

    // Get the range of rows for the current color
    size_t colorSortStart = colorSortStartAddr[currentColor];
    size_t colorSortEnd = colorSortStartAddr[currentColor + 1];

    // Distribute rows of the current color to workers
    for (size_t colorSortI = colorSortStart + threadId;
         colorSortI < colorSortEnd;
         colorSortI += poplar::MultiVertex::numWorkers()) {
      // Get the actual row index
      auto i = colorSortAddr[colorSortI];

      // Update the i-th row

      // for k = 1, ..., i - 1
      // => iterate over the lower triangular part of the current row
      size_t iStart = rowPtr[i];
      size_t iEnd = rowPtr[i + 1];
      __builtin_assume(iEnd - iStart < 1000);
      for (size_t ik = iStart; ik < iEnd; ik++) {
        size_t k = colInd[ik];
        if (k >= i) break;

        // We don't need to check for halo coefficients in the lower
        // triangular part (due to k < i)

        value_t a_kk = diagCoeffs[k];
        value_t &a_ik = offDiagCoeffs[ik];
        a_ik = a_ik / a_kk;

        // The algorithm requires us to iterate over
        // j = k + 1, ..., n
        // Due to our matrix layout, we iterate over:
        // 1. j = k + 1, ..., n with j != i (off-diagonal)
        // 2. j = i (diagonal)

        // For j = k + 1, ..., n, j != i (off-diagonal)
        for (size_t ij = ik + 1; ij < iEnd; ij++) {
          auto j = colInd[ij];
          // Stop at halo coefficients
          if (j >= nRowsWithoutHalo) break;
          value_t a_kj = getOffDiagValue(k, j);
          value_t &a_ij = offDiagCoeffs[ij];
          a_ij -= a_ik * a_kj;
        }

        // For j = i (diagonal)
        value_t a_ki = getOffDiagValue(k, i);
        value_t &a_ii = diagCoeffs[i];
        a_ii -= a_ik * a_ki;
      }
    }

    return true;
  }
};

// Instantiate the template for all possible combinations of the template
// parameters
#define INSTANTIATE() INSTANTIATE_1(float);
#define INSTANTIATE_1(value_t)            \
  INSTANTIATE_2(value_t, unsigned int);   \
  INSTANTIATE_2(value_t, unsigned short); \
  INSTANTIATE_2(value_t, unsigned char);
#define INSTANTIATE_2(value_t, rowptr_t)            \
  INSTANTIATE_3(value_t, rowptr_t, unsigned int);   \
  INSTANTIATE_3(value_t, rowptr_t, unsigned short); \
  INSTANTIATE_3(value_t, rowptr_t, unsigned char);
#define INSTANTIATE_3(value_t, rowptr_t, colind_t)            \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned int);   \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned short); \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, unsigned char);
#define INSTANTIATE_4(value_t, rowptr_t, colind_t, colorsortaddr_t)            \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned int);   \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned short); \
  INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, unsigned char);
#define INSTANTIATE_5(value_t, rowptr_t, colind_t, colorsortaddr_t, \
                      colorsortstartaddr_t)                         \
  template class ILUFactorizeCRSMulticolor<                         \
      value_t, rowptr_t, colind_t, colorsortaddr_t, colorsortstartaddr_t>;

INSTANTIATE();

#undef INSTANTIATE
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3
#undef INSTANTIATE_4
#undef INSTANTIATE_5
}  // namespace graphene::matrix::solver::ilu