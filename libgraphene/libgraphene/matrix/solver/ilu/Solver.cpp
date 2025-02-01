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

#include "libgraphene/matrix/solver/ilu/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/ControlFlow.hpp"
#include "libgraphene/dsl/code/Function.hpp"
#include "libgraphene/dsl/code/Operators.hpp"
#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/code/Vertex.hpp"
#include "libgraphene/dsl/tensor/ControlFlow.hpp"
#include "libgraphene/dsl/tensor/Execute.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/details/crs/CRSMatrix.hpp"
#include "libgraphene/matrix/solver/SolverStats.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene::matrix::solver::ilu {

Solver::Solver(const Matrix& matrix, std::shared_ptr<Configuration> config)
    : solver::Solver(matrix),
      config_(config),
      solveMulticolor_(this->shouldUseMulticolor(config_->solveMulticolor)),
      factorizeMulticolor_(
          this->shouldUseMulticolor(config_->factorizeMulticolor)) {
  factorize();
}

void Solver::factorizeCRS_ILU_old() {
  const crs::CRSMatrix& A = this->matrix().template getImpl<crs::CRSMatrix>();
  GRAPHENE_TRACEPOINT();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  factorization_.factorizedInverseDiag =
      std::make_unique<Tensor>(A.diagonalCoefficients);
  if (!config_->diagonalBased)
    factorization_.factorizedOffDiag =
        std::make_unique<Tensor>(A.offDiagonalCoefficients);

  if (factorizeMulticolor_) {
    if (!A.addressing->coloring.has_value()) {
      throw std::runtime_error(
          "Matrix has no coloring. Cannot use multicolor factorization");
    }
    spdlog::trace("Using multicolor factorization");
  } else {
    spdlog::trace("Using sequential factorization");
  }
  spdlog::trace("Factorizing in {}-mode", this->name());

  poplar::ComputeSet cs = graph.addComputeSet(di);

  for (size_t tile = 0; tile < A.numTiles(); ++tile) {
    poplar::Tensor rowPtrTile = A.addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = A.addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile =
        factorization_.factorizedInverseDiag->tensorOnTile(tile);
    poplar::Tensor offDiagCoeffsTile;
    if (config_->diagonalBased)
      offDiagCoeffsTile = A.offDiagonalCoefficients.tensorOnTile(tile);
    else
      offDiagCoeffsTile = factorization_.factorizedOffDiag->tensorOnTile(tile);
    poplar::Tensor colorSortAddrTile, colorSortStartPtrTile;
    if (factorizeMulticolor_) {
      colorSortAddrTile =
          A.addressing->coloring->colorSortAddr.tensorOnTile(tile);
      colorSortStartPtrTile =
          A.addressing->coloring->colorSortStartPtr.tensorOnTile(tile);
    }

    std::string codeletName = "graphene::matrix::solver::ilu::ILUFactorizeCRS";
    if (config_->diagonalBased) {
      codeletName += "Diagonal";
    }
    if (factorizeMulticolor_) {
      codeletName += "Multicolor";
      codeletName = poputil::templateVertex(
          codeletName, Type::FLOAT32->poplarType(), rowPtrTile.elementType(),
          colIndTile.elementType(), colorSortAddrTile.elementType(),
          colorSortStartPtrTile.elementType());
    } else {
      codeletName = poputil::templateVertex(
          codeletName, Type::FLOAT32->poplarType(), rowPtrTile.elementType(),
          colIndTile.elementType());
    }

    auto vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["diagCoeffs"], diagCoeffsTile);
    graph.connect(vertex["offDiagCoeffs"], offDiagCoeffsTile);
    if (factorizeMulticolor_) {
      graph.connect(vertex["colorSortAddr"], colorSortAddrTile);
      graph.connect(vertex["colorSortStartAddr"], colorSortStartPtrTile);
    }

    graph.setPerfEstimate(vertex, 100);
  }

  program.add(poplar::program::Execute(cs, di));

  // Invert the diagonal
  *factorization_.factorizedInverseDiag =
      1 / *factorization_.factorizedInverseDiag;
}

void Solver::factorizeCRS() {
  const crs::CRSMatrix& A = this->matrix().template getImpl<crs::CRSMatrix>();
  GRAPHENE_TRACEPOINT();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  // Initialize the factorized matrix with the original matrix A (copy)
  factorization_.factorizedInverseDiag =
      std::make_unique<Tensor>(A.diagonalCoefficients);
  if (!config_->diagonalBased)
    factorization_.factorizedOffDiag =
        std::make_unique<Tensor>(A.offDiagonalCoefficients);

  // Returns the off-diagonal value at position (i, j)
  using namespace codedsl;
  auto getOffDiagValue = [&](Value i, Value j, Value offDiagCoeffs,
                             Value rowPtr, Value colInd) -> Value {
    Variable offDiagValue(offDiagCoeffs[0].type(), 0);
    Variable start = rowPtr[i];
    Variable end = rowPtr[i + 1];
    AssumeHardwareLoop(end - start);
    For(start, end, 1, [&](Value a) {
      Value jCurrent = colInd[a];
      If(jCurrent == j, [&] { offDiagValue = offDiagCoeffs[a]; });
    });
    return offDiagValue;
  };

  // Factorizes the ith row of the matrix
  // This is algorithm 10.4 in "Iterative Methods for Sparse Linear
  // Systems".The coefficients must be initialized with the original
  // matrix A
  auto factorizeRow = [&](Value i, Value rowPtr, Value colInd, Value diagCoeffs,
                          Value offDiagCoeffs, Value nRowsWithoutHalo) {
    // for k = 1, ..., i - 1
    // => iterate over the lower triangular part of the current row
    Value iStart = rowPtr[i];
    Value iEnd = rowPtr[i + 1];
    // AssumeHardwareLoop(iEnd - iStart);
    For(iStart, iEnd, 1, [&](Value ik) {
      Value k = colInd[ik];

      // We don't need to check for halo coefficients in the lower
      // triangular part (due to k < i)
      If(k >= i, [&] { Break(); });
      // FIXME: k >= i for ILU, k > i for DILU??!!!!!! <-------
      Value a_kk = diagCoeffs[k];
      Value a_ik = offDiagCoeffs[ik];

      // The algorithm requires us to iterate over
      // j = k + 1, ..., n
      // Due to our matrix layout, we iterate over:
      // 1. j = k + 1, ..., n with j != i (off-diagonal)
      // 2. j = i (diagonal)

      // The DILU algorithm ignores off-diagonal elements during factorization:
      if (!config_->diagonalBased) {
        // we can divide a_ik in-place, as we can modify the off-diagonal
        // coefficients (in contrast to DILU)
        a_ik = a_ik / a_kk;

        // For j = k + 1, ..., n, j != i (off-diagonal)
        For(ik + 1, iEnd, 1, [&](Value ij) {
          Value j = colInd[ij];
          // Stop at halo coefficients
          If(j >= nRowsWithoutHalo, [] { Break(); });

          Value a_kj = getOffDiagValue(k, j, offDiagCoeffs, rowPtr, colInd);
          Value a_ij = offDiagCoeffs[ij];
          a_ij -= a_ik * a_kj;
        });
      }

      // For j = i (diagonal)
      Value a_ki = getOffDiagValue(k, i, offDiagCoeffs, rowPtr, colInd);
      Value a_ii = diagCoeffs[i];
      if (config_->diagonalBased)
        // because we can't modify the diagonal coefficients in case of DILU, we
        // need to divide through a_kk here
        a_ii -= a_ik * a_ki / a_kk;
      else
        a_ii -= a_ik * a_ki;
    });
  };

  if (factorizeMulticolor_) {
    spdlog::trace("Using multicolor {} factorization", name());
    ExecuteOnSupervisor(
        [&](Value diagCoeffs, Value offDiagCoeffs, Value rowPtr, Value colInd,
            Value colorSortAddr, Value colorSortStartPtr, Value currentColor,
            Value nRowsWithoutHalo) {
          using namespace codedsl;

          codedsl::Function factorizeColor(
              "factorizeColor", Type::BOOL, {Type::UINT32}, ThreadKind::Worker,
              [&](Parameter workerID) -> void {
                // Copy because currentColor is volatile
                Variable currentColor_ = currentColor;
                Variable nRowsWithoutHalo_ = nRowsWithoutHalo;
                Value colorSortStart = colorSortStartPtr[currentColor_];
                Value colorSortEnd = colorSortStartPtr[currentColor_ + 1];

                // Factorize the rows of the current color
                For(colorSortStart + workerID, colorSortEnd,
                    getNumWorkerThreadsPerTile(), [&](Value colorSortI) {
                      Value i = colorSortAddr[colorSortI];
                      factorizeRow(i, rowPtr, colInd, diagCoeffs, offDiagCoeffs,
                                   nRowsWithoutHalo_);
                    });
                Return(true);
              });

          nRowsWithoutHalo = diagCoeffs.size();

          Value numColors = colorSortStartPtr.size() - 1;
          // Solve lower triangular for each color in parallel
          For(0, numColors, 1, [&](Value colorI) {
            currentColor = colorI;
            syncAndStartOnAllWorkers(factorizeColor);
          });

          // Make sure all workers are done before returning
          syncAllWorkers();
        },
        InOut(*factorization_.factorizedInverseDiag),
        config_->diagonalBased ? In(A.offDiagonalCoefficients)
                               : InOut(*factorization_.factorizedOffDiag),
        In(A.addressing->rowPtr), In(A.addressing->colInd),
        In(A.addressing->coloring->colorSortAddr),
        In(A.addressing->coloring->colorSortStartPtr),
        /* current color */
        Member(Type::UINT32, CTypeQualifiers::getVolatile()),
        /* nRowsWithoutHalo */
        Member(Type::UINT32, CTypeQualifiers::getVolatile()));
  } else {
    spdlog::trace("Using sequential {} factorization", name());
    Execute(
        [&](Value diagCoeffs, Value offDiagCoeffs, Value rowPtr, Value colInd) {
          using namespace codedsl;

          Value nRowsWithoutHalo = diagCoeffs.size();

          // Main loop of ILU(0) factorization
          For(1, nRowsWithoutHalo, 1, [&](Value i) {
            // Update the i-th row
            factorizeRow(i, rowPtr, colInd, diagCoeffs, offDiagCoeffs,
                         nRowsWithoutHalo);
          });
        },
        InOut(*factorization_.factorizedInverseDiag),
        config_->diagonalBased ? In(A.offDiagonalCoefficients)
                               : InOut(*factorization_.factorizedOffDiag),
        In(A.addressing->rowPtr), In(A.addressing->colInd));
  }
  // Invert the diagonal
  *factorization_.factorizedInverseDiag =
      1 / *factorization_.factorizedInverseDiag;
}

void Solver::solveCRS_old(Tensor& x, Tensor& b) {
  const auto& A = this->matrix().template getImpl<crs::CRSMatrix>();
  GRAPHENE_TRACEPOINT();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  if (solveMulticolor_) {
    if (!A.addressing->coloring.has_value()) {
      throw std::runtime_error(
          "Matrix has no coloring. Cannot use multicolor substitution");
    }
    spdlog::trace("Using multicolor substitution");
  } else {
    spdlog::trace("Using sequential substitution");
  }
  spdlog::trace("Solving in {}-mode", this->name());

  poplar::ComputeSet cs = graph.addComputeSet(di);

  for (size_t tile = 0; tile < A.numTiles(); ++tile) {
    poplar::Tensor xTile = x.tensorOnTile(tile);
    poplar::Tensor bTile = b.tensorOnTile(tile);
    poplar::Tensor rowPtrTile = A.addressing->rowPtr.tensorOnTile(tile);
    poplar::Tensor colIndTile = A.addressing->colInd.tensorOnTile(tile);
    poplar::Tensor diagCoeffsTile =
        factorization_.factorizedInverseDiag->tensorOnTile(tile);

    poplar::Tensor offDiagCoeffsTile;
    if (config_->diagonalBased)
      offDiagCoeffsTile = A.offDiagonalCoefficients.tensorOnTile(tile);
    else
      offDiagCoeffsTile = factorization_.factorizedOffDiag->tensorOnTile(tile);
    poplar::Tensor colorSortAddrTile, colorSortStartPtrTile;
    if (solveMulticolor_) {
      colorSortAddrTile =
          A.addressing->coloring->colorSortAddr.tensorOnTile(tile);
      colorSortStartPtrTile =
          A.addressing->coloring->colorSortStartPtr.tensorOnTile(tile);
    }

    std::string codeletName = "graphene::matrix::solver::ilu::ILUSolveCRS";
    if (solveMulticolor_) {
      codeletName += "Multicolor";
      codeletName = poputil::templateVertex(
          codeletName, Type::FLOAT32->poplarType(), rowPtrTile.elementType(),
          colIndTile.elementType(), colorSortAddrTile.elementType(),
          colorSortStartPtrTile.elementType(), config_->diagonalBased);
    } else {
      codeletName = poputil::templateVertex(
          codeletName, Type::FLOAT32->poplarType(), rowPtrTile.elementType(),
          colIndTile.elementType(), config_->diagonalBased);
    }

    auto vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["x"], xTile);
    graph.connect(vertex["b"], bTile);
    graph.connect(vertex["rowPtr"], rowPtrTile);
    graph.connect(vertex["colInd"], colIndTile);
    graph.connect(vertex["inverseDiagCoeffs"], diagCoeffsTile);
    graph.connect(vertex["offDiagCoeffs"], offDiagCoeffsTile);
    if (solveMulticolor_) {
      graph.connect(vertex["colorSortAddr"], colorSortAddrTile);
      graph.connect(vertex["colorSortStartAddr"], colorSortStartPtrTile);
    }

    graph.setPerfEstimate(vertex, 100);
  }

  program.add(poplar::program::Execute(cs, di));
}

void Solver::solveCRS(Tensor& x, Tensor& b) {
  GRAPHENE_TRACEPOINT();
  const auto& A = this->matrix().template getImpl<crs::CRSMatrix>();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  if (solveMulticolor_ && !A.addressing->coloring.has_value()) {
    throw std::runtime_error(
        "Matrix has no coloring. Cannot use multicolor substitution");
  }

  using namespace codedsl;
  auto lowerTriangularSolveForRow =
      [&](Value i, Value rowPtr, Value offDiagCoeffs, Value colInd, Value x,
          Value b, Value inverseDiagCoeffs, Value nRowsWithoutHalo) {
        using namespace codedsl;
        Variable temp(x[0].type(), b[i]);

        // Iterate over lower triangular part of the matrix
        // => for every i > j
        Value start = rowPtr[i];
        Value end = rowPtr[i + 1];
        // AssumeHardwareLoop(end-start);
        For(start, end, 1, [&](Value a) {
          Value j = colInd[a];

          // Stop if we reach the diagonal, we only need the lower triangular
          // part
          If(i < j, [] { Break(); });

          // Ignore halo coefficients
          If(j >= nRowsWithoutHalo, [] { Break(); });

          Value l_ij = offDiagCoeffs[a];
          temp -= l_ij * x[j];
        });

        if (config_->diagonalBased) {
          x[i] = temp * inverseDiagCoeffs[i];
        } else {
          // The diagonal of L is 1, so no need to multiply
          x[i] = temp;
        }
      };

  auto upperTriangularSolveForRow =
      [&](Value i, Value rowPtr, Value offDiagCoeffs, Value colInd, Value x,
          Value inverseDiagCoeffs, Value nRowsWithoutHalo) {
        using namespace codedsl;
        Variable temp(x[0].type(), 0);

        // Iterate over upper triangular part of the matrix
        // => for every i < j
        Value start = rowPtr[i];
        Value end = rowPtr[i + 1];
        // AssumeHardwareLoop(end-start);
        ForReverse(end - 1, start, 1, [&](Value a) {
          Value j = colInd[a];

          // Stop if we reach the diagonal, we only need the upper triangular
          // part
          If(j < i, [] { Break(); });

          // Ignore halo coefficients
          If(j >= nRowsWithoutHalo, [] { Continue(); });

          Value a_ij = offDiagCoeffs[a];
          temp -= a_ij * x[j];
        });
        if (config_->diagonalBased) {
          x[i] = x[i] + temp * inverseDiagCoeffs[i];
        } else {
          x[i] = (x[i] + temp) * inverseDiagCoeffs[i];
        }
      };

  if (solveMulticolor_) {
    using namespace codedsl;
    // Multicolor parallel substitution
    spdlog::trace("Using multicolor {} substitution", name());
    ExecuteOnSupervisor(
        [&](Value x, Value b, Value inverseDiagCoeffs, Value offDiagCoeffs,
            Value rowPtr, Value colInd, Value colorSortAddr,
            Value colorSortStartPtr, Value currentColor,
            Value nRowsWithoutHalo) {
          codedsl::Function lowerTriangularSolveForColor(
              "solveLowerForColor", Type::BOOL, {Type::UINT32},
              ThreadKind::Worker, [&](Parameter workerID) -> void {
                // Copy because currentColor is volatile
                Variable currentColor_ = currentColor;
                Variable nRowsWithoutHalo_ = nRowsWithoutHalo;
                Value colorSortStart = colorSortStartPtr[currentColor_];
                Value colorSortEnd = colorSortStartPtr[currentColor_ + 1];

                // Lower triangular solve for all rows in the current color
                For(colorSortStart + workerID, colorSortEnd,
                    getNumWorkerThreadsPerTile(), [&](Value colorSortI) {
                      Value i = colorSortAddr[colorSortI];
                      lowerTriangularSolveForRow(
                          i, rowPtr, offDiagCoeffs, colInd, x, b,
                          inverseDiagCoeffs, nRowsWithoutHalo_);
                    });
                Return(true);
              });

          codedsl::Function upperTriangularSolveForColor(
              "solveUpperForColor", Type::BOOL, {Type::UINT32},
              ThreadKind::Worker, [&](Parameter workerID) -> void {
                // Copy because currentColor and nRowsWithoutHalo are volatile
                Variable currentColor_ = currentColor;
                Variable nRowsWithoutHalo_ = nRowsWithoutHalo;
                Value colorSortStart = colorSortStartPtr[currentColor_];
                Value colorSortEnd = colorSortStartPtr[currentColor_ + 1];

                // Upper triangular solve for all rows in the current color
                For(colorSortStart + workerID, colorSortEnd,
                    getNumWorkerThreadsPerTile(), [&](Value colorSortI) {
                      Value i = colorSortAddr[colorSortI];
                      upperTriangularSolveForRow(i, rowPtr, offDiagCoeffs,
                                                 colInd, x, inverseDiagCoeffs,
                                                 nRowsWithoutHalo_);
                    });
                Return(true);
              });

          // Syncing all workers before and not after starting the workers is
          // beneficial because the supervisor can prepare the next color while
          // some of the workers are still working on the previous color.
          nRowsWithoutHalo = inverseDiagCoeffs.size();
          Value numColors = colorSortStartPtr.size() - 1;
          // Solve lower triangular for each color in parallel
          For(0, numColors, 1, [&](Value colorI) {
            currentColor = colorI;
            syncAndStartOnAllWorkers(lowerTriangularSolveForColor);
          });

          // Solve upper triangular for each color in parallel
          ForReverse(numColors - 1, 0, 1, [&](Value colorI) {
            currentColor = colorI;
            syncAndStartOnAllWorkers(upperTriangularSolveForColor);
          });

          // Make sure all workers are done before returning
          syncAllWorkers();
        },
        InOut(x), In(b), In(*factorization_.factorizedInverseDiag),
        config_->diagonalBased ? In(A.offDiagonalCoefficients)
                               : In(*factorization_.factorizedOffDiag),
        In(A.addressing->rowPtr), In(A.addressing->colInd),
        In(A.addressing->coloring->colorSortAddr),
        In(A.addressing->coloring->colorSortStartPtr),
        /* current color */
        Member(Type::UINT32, CTypeQualifiers::getVolatile()),
        /* nRowsWithoutHalo */
        Member(Type::UINT32, CTypeQualifiers::getVolatile()));
  } else {
    // Sequential substitution
    spdlog::trace("Using sequential {} substitution", name());
    Execute(
        [&](Value x, Value b, Value inverseDiagCoeffs, Value offDiagCoeffs,
            Value rowPtr, Value colInd) {
          using namespace codedsl;

          Value nRowsWithoutHalo = inverseDiagCoeffs.size();

          // lower triangular solve
          For(0, nRowsWithoutHalo, 1, [&](Value i) {
            lowerTriangularSolveForRow(i, rowPtr, offDiagCoeffs, colInd, x, b,
                                       inverseDiagCoeffs, nRowsWithoutHalo);
          });

          // upper triangular solve
          ForReverse(nRowsWithoutHalo - 1, 0, 1, [&](Value i) {
            upperTriangularSolveForRow(i, rowPtr, offDiagCoeffs, colInd, x,
                                       inverseDiagCoeffs, nRowsWithoutHalo);
          });
        },
        InOut(x), In(b), In(*factorization_.factorizedInverseDiag),
        config_->diagonalBased ? In(A.offDiagonalCoefficients)
                               : In(*factorization_.factorizedOffDiag),
        In(A.addressing->rowPtr), In(A.addressing->colInd));
  }
}

void Solver::factorize() {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      factorizeCRS();
      break;
    default:
      throw std::runtime_error("Unsupported matrix format");
  }
}

SolverStats Solver::solve(Tensor& x, Tensor& b) {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      solveCRS(x, b);
      break;
    default:
      throw std::runtime_error("Unsupported matrix format");
  }

  SolverStats stats(this->name(), VectorNorm::None, this->matrix().numTiles());
  stats.iterations = 1;
  return stats;
}
}  // namespace graphene::matrix::solver::ilu