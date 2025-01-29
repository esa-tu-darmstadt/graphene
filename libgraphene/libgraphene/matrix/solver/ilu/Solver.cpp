#include "libgraphene/matrix/solver/ilu/Solver.hpp"

#include <spdlog/spdlog.h>

#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/code/Operators.hpp"
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

void Solver::factorizeCRS_ILU() {
  const crs::CRSMatrix& A = this->matrix().template getImpl<crs::CRSMatrix>();
  GRAPHENE_TRACEPOINT();
  DebugInfo di("ILUSolver");
  auto& graph = Context::graph();
  auto& program = Context::program();

  // Initialize the factorized matrix with the original matrix A (copy)
  factorization_.factorizedInverseDiag =
      std::make_unique<Tensor>(A.diagonalCoefficients);
  factorization_.factorizedOffDiag =
      std::make_unique<Tensor>(A.offDiagonalCoefficients);

  if (factorizeMulticolor_)
    throw std::runtime_error("Multicolor factorization not yet implemented");

  // ILU(0)
  Execute(
      [](codedsl::Value diagCoeffs, codedsl::Value offDiagCoeffs,
         codedsl::Value rowPtr, codedsl::Value colInd) {
        using namespace codedsl;

        Value nRowsWithoutHalo = diagCoeffs.size();

        // We could also use a codedsl Function instead. This lambda will get
        // inlined
        auto getOffDiagValue = [&](Value i, Value j) -> Value {
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

        // This is algorithm 10.4 in "Iterative Methods for Sparse Linear
        // Systems".The coefficients must be initialized with the original
        // matrix A

        // Main loop of ILU(0) factorization
        For(1, nRowsWithoutHalo, 1, [&](Value i) {
          // Update the i-th row

          // for k = 1, ..., i - 1
          // => iterate over the lower triangular part of the current row
          Value iStart = rowPtr[i];
          Value iEnd = rowPtr[i + 1];
          // AssumeHardwareLoop(iEnd - iStart);
          For(iStart, iEnd, 1, [&](Value ik) {
            Value k = colInd[ik];

            // We don't need to check for halo coefficients in the lower
            // triangular part (due to k < i)
            If(k >= i, [] { Break(); });

            Value a_kk = diagCoeffs[k];
            Value a_ik = offDiagCoeffs[ik];
            a_ik = a_ik / a_kk;

            // The algorithm requires us to iterate over
            // j = k + 1, ..., n
            // Due to our matrix layout, we iterate over:
            // 1. j = k + 1, ..., n with j != i (off-diagonal)
            // 2. j = i (diagonal)

            // For j = k + 1, ..., n, j != i (off-diagonal)
            For(ik + 1, iEnd, 1, [&](Value ij) {
              Value j = colInd[ij];
              // Stop at halo coefficients
              If(j >= nRowsWithoutHalo, [] { Break(); });

              Value a_kj = getOffDiagValue(k, j);
              Value a_ij = offDiagCoeffs[ij];
              a_ij -= a_ik * a_kj;
            });

            // For j = i (diagonal)
            Value a_ki = getOffDiagValue(k, i);
            Value a_ii = diagCoeffs[i];
            a_ii -= a_ik * a_ki;
          });
        });
      },
      InOut(*factorization_.factorizedInverseDiag),
      config_->diagonalBased ? In(A.offDiagonalCoefficients)
                             : InOut(*factorization_.factorizedOffDiag),
      In(A.addressing->rowPtr), In(A.addressing->colInd));

  // Invert the diagonal
  *factorization_.factorizedInverseDiag =
      1 / *factorization_.factorizedInverseDiag;
}

void Solver::solveCRS(Tensor& x, Tensor& b) {
  GRAPHENE_TRACEPOINT();
  const auto& A = this->matrix().template getImpl<crs::CRSMatrix>();
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

  Execute(
      [&](codedsl::Value x, codedsl::Value b, codedsl::Value inverseDiagCoeffs,
          codedsl::Value offDiagCoeffs, codedsl::Value rowPtr,
          codedsl::Value colInd) {
        using namespace codedsl;

        Value nRowsWithoutHalo = inverseDiagCoeffs.size();

        // lower triangular solve
        For(0, nRowsWithoutHalo, 1, [&](Value i) {
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
        });

        // upper triangular solve
        ForReverse(nRowsWithoutHalo - 1, 0, 1, [&](Value i) {
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
            If(j >= nRowsWithoutHalo, [] { Break(); });

            Value a_ij = offDiagCoeffs[a];
            temp -= a_ij * x[j];
          });
          if (config_->diagonalBased) {
            x[i] = x[i] + temp * inverseDiagCoeffs[i];
          } else {
            x[i] = (x[i] + temp) * inverseDiagCoeffs[i];
          }
        });
        return true;
      },
      InOut(x), In(b), In(*factorization_.factorizedInverseDiag),
      config_->diagonalBased ? In(A.offDiagonalCoefficients)
                             : In(*factorization_.factorizedOffDiag),
      In(A.addressing->rowPtr), In(A.addressing->colInd));
}

void Solver::factorize() {
  switch (this->matrix().getFormat()) {
    case MatrixFormat::CRS:
      if (config_->diagonalBased)
        // factorizeCRS_DILU();
        throw std::runtime_error("Diagonal-based ILU not yet implemented");
      else
        factorizeCRS_ILU();
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