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
template <typename value_t, typename rowptr_t, typename colind_t,
          bool diagonalBased>
class ILUSolveCRS : public poplar::Vertex {
 public:
  InOut<Vector<value_t>> x;
  Input<Vector<value_t>> b;
  Input<Vector<value_t>> inverseDiagCoeffs;
  Input<Vector<value_t>> offDiagCoeffs;

  Input<Vector<rowptr_t>> rowPtr;
  Input<Vector<colind_t>> colInd;

  bool compute() {
    // printf("Lower triangular solve\n");
    const auto nRowsWithoutHalo = inverseDiagCoeffs.size();
    // lower triangular solve
    for (size_t i = 0; i < nRowsWithoutHalo; i++) {
      float temp = b[i];

      // Iterate over lower triangular part of the matrix
      // => for every i > j
      size_t start = rowPtr[i];
      size_t end = rowPtr[i + 1];
      __builtin_assume(end - start < 1000);
      for (size_t a = start; a < end; a++) {
        size_t j = colInd[a];

        // Stop if we reach the diagonal, we only need the lower triangular part
        if (i < j) break;

        if (j >= nRowsWithoutHalo) break;

        float l_ij = offDiagCoeffs[a];
        temp -= l_ij * x[j];
      }

      if constexpr (diagonalBased) {
        x[i] = temp * inverseDiagCoeffs[i];
      } else {
        // The diagonal of L is 1, so no need to multiply
        x[i] = temp;
      }
    }

    // upper triangular solve
    for (int i = (int)nRowsWithoutHalo - 1; i >= 0; i--) {
      float temp = 0;

      // Iterate over upper triangular part of the matrix
      // => for every i < j
      int start = rowPtr[i];
      int end = rowPtr[i + 1];

      __builtin_assume(end - start < 1000);
      for (int a = (int)end - 1; a >= start; --a) {
        size_t j = colInd[a];

        // Stop if we reach the diagonal, we only need the upper triangular part
        if (j < i) break;

        // Ignore halo coefficients
        if (j >= nRowsWithoutHalo) continue;

        float a_ij = offDiagCoeffs[a];
        temp -= a_ij * x[j];
      }
      if constexpr (diagonalBased) {
        x[i] = x[i] + temp * inverseDiagCoeffs[i];
      } else {
        x[i] = (x[i] + temp) * inverseDiagCoeffs[i];
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
#define INSTANTIATE_3(value_t, rowptr_t, colind_t)  \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, true); \
  INSTANTIATE_4(value_t, rowptr_t, colind_t, false);
#define INSTANTIATE_4(value_t, rowptr_t, colind_t, diagonalBased) \
  template class ILUSolveCRS<value_t, rowptr_t, colind_t, diagonalBased>;

INSTANTIATE();
#undef INSTANTIATE
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3
#undef INSTANTIATE_4

template <typename value_t, typename rowptr_t, typename colind_t,
          typename colorsortaddr_t, typename colorsortstartaddr_t,
          bool diagonalBased>
class ILUSolveCRSMulticolor : public poplar::SupervisorVertex {
  using ThisType =
      ILUSolveCRSMulticolor<value_t, rowptr_t, colind_t, colorsortaddr_t,
                            colorsortstartaddr_t, diagonalBased>;
  volatile size_t currentColor_;
  size_t nRowsWithoutHalo;

 public:
  InOut<Vector<value_t>> x;
  Input<Vector<value_t>> b;
  Input<Vector<value_t>> inverseDiagCoeffs;
  Input<Vector<value_t>> offDiagCoeffs;

  Input<Vector<rowptr_t>> rowPtr;
  Input<Vector<colind_t>> colInd;

  Input<Vector<colorsortaddr_t, VectorLayout::SPAN, 8>> colorSortAddr;
  Input<Vector<colorsortstartaddr_t, VectorLayout::SPAN, 8>> colorSortStartAddr;

  SUPERVISOR_FUNC bool compute() {
    // Syncing all workers before and not after starting the workers is
    // beneficial because the supervisor can prepare the next color while some
    // of the workers are still working on the previous color.
    nRowsWithoutHalo = inverseDiagCoeffs.size();
    size_t numColors = colorSortStartAddr.size() - 1;
    // Solve lower triangular for each color in parallel
    for (size_t colorI = 0; colorI < numColors; colorI++) {
      currentColor_ = colorI;
      ipu::syncAndStartOnAllWorkers<ThisType, &ThisType::solveLowerForColor>(
          this);
    }

    // Solve upper triangular for each color in parallel
    for (int colorI = numColors - 1; colorI >= 0; colorI--) {
      currentColor_ = colorI;
      ipu::syncAndStartOnAllWorkers<ThisType, &ThisType::solveUpperForColor>(
          this);
    }

    // Make sure all workers are done before returning
    ipu::syncAllWorkers();
    return true;
  }

  WORKER_FUNC bool solveLowerForColor(unsigned threadId) {
    size_t currentColor = currentColor_;
    size_t colorSortStart = colorSortStartAddr[currentColor];
    size_t colorSortEnd = colorSortStartAddr[currentColor + 1];

    for (size_t colorSortI = colorSortStart + threadId;
         colorSortI < colorSortEnd; colorSortI += MultiVertex::numWorkers()) {
      size_t i = colorSortAddr[colorSortI];
      float temp = b[i];

      // Iterate over lower triangular part of the matrix
      // => for every i > j
      size_t start = rowPtr[i];
      size_t end = rowPtr[i + 1];
      __builtin_assume(end - start < 1000);
      for (size_t a = start; a < end; a++) {
        size_t j = colInd[a];

        // Stop if we reach the diagonal, we only need the lower triangular part
        if (i < j) break;

        if (j >= nRowsWithoutHalo) break;

        float l_ij = offDiagCoeffs[a];
        temp -= l_ij * x[j];
      }
      if constexpr (diagonalBased) {
        x[i] = temp * inverseDiagCoeffs[i];
      } else {
        // The diagonal of L is 1, so no need to multiply
        x[i] = temp;
      }
    }

    return true;
  }

  WORKER_FUNC bool solveUpperForColor(unsigned threadId) {
    size_t currentColor = currentColor_;
    size_t colorSortStart = colorSortStartAddr[currentColor];
    size_t colorSortEnd = colorSortStartAddr[currentColor + 1];

    // upper triangular solve
    for (size_t colorSortI = colorSortStart + threadId;
         colorSortI < colorSortEnd; colorSortI += MultiVertex::numWorkers()) {
      size_t i = colorSortAddr[colorSortI];
      float temp = 0;

      // Iterate over upper triangular part of the matrix
      // => for every i < j
      int start = rowPtr[i];
      int end = rowPtr[i + 1];

      __builtin_assume(end - start < 1000);
      for (int a = (int)end - 1; a >= start; --a) {
        size_t j = colInd[a];

        // Stop if we reach the diagonal, we only need the upper triangular part
        if (j < i) break;

        // Ignore halo coefficients
        if (j >= nRowsWithoutHalo) continue;

        float a_ij = offDiagCoeffs[a];
        temp -= a_ij * x[j];
      }
      if constexpr (diagonalBased) {
        x[i] = x[i] + temp * inverseDiagCoeffs[i];
      } else {
        x[i] = (x[i] + temp) * inverseDiagCoeffs[i];
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
  INSTANTIATE_6(value_t, rowptr_t, colind_t, colorsortaddr_t,       \
                colorsortstartaddr_t, true);                        \
  INSTANTIATE_6(value_t, rowptr_t, colind_t, colorsortaddr_t,       \
                colorsortstartaddr_t, false);
#define INSTANTIATE_6(value_t, rowptr_t, colind_t, colorsortaddr_t,           \
                      colorsortstartaddr_t, diagonalBased)                    \
  template class ILUSolveCRSMulticolor<value_t, rowptr_t, colind_t,           \
                                       colorsortaddr_t, colorsortstartaddr_t, \
                                       diagonalBased>;

INSTANTIATE();
#undef INSTANTIATE
#undef INSTANTIATE_1
#undef INSTANTIATE_2
#undef INSTANTIATE_3
#undef INSTANTIATE_4
#undef INSTANTIATE_5
#undef INSTANTIATE_6

}  // namespace graphene::matrix::solver::ilu
