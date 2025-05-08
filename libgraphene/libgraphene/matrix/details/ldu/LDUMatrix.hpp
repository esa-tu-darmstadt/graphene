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

#pragma once

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/ldu/LDUAddressing.hpp"
#include "libgraphene/matrix/host/DistributedTileLayout.hpp"

namespace graphene::matrix::ldu {

/**
 * @brief A class representing a matrix in LDU format.
 */
struct LDUMatrix : public MatrixBase {
  std::shared_ptr<LDUAddressing> addressing;
  Tensor diagonalCoefficients;
  Tensor lowerCoefficients;
  std::optional<Tensor> upperCoefficients;

  /**
   * @brief Construct a new LDU matrix.
   *
   * @param hostMatrix The host matrix this LDU matrix is based on.
   * @param addressing The addressing scheme for the LDU matrix.
   * @param diagonalCoefficients The diagonal coefficients of the matrix.
   * @param lowerCoefficients The coefficients of the lower triangular part.
   * @param upperCoefficients The coefficients of the upper triangular part, if
   * they are different from the lower coefficients (asymmetric matrix).
   */
  LDUMatrix(const host::DistributedTileLayout &tileLayout,
            std::shared_ptr<LDUAddressing> addressing,
            Tensor diagonalCoefficients, Tensor lowerCoefficients,
            std::optional<Tensor> upperCoefficients = std::nullopt)
      : MatrixBase(tileLayout),
        addressing(std::move(addressing)),
        diagonalCoefficients(std::move(diagonalCoefficients)),
        lowerCoefficients(std::move(lowerCoefficients)),
        upperCoefficients(std::move(upperCoefficients)) {}

  Tensor spmv(Tensor &x, TypeRef destType = nullptr,
              TypeRef intermediateType = nullptr,
              bool withHalo = false) const override;

  Tensor residual(Tensor &x, const Tensor &b, TypeRef destType = nullptr,
                  TypeRef intermediateType = nullptr,
                  bool withHalo = false) const override;

  MatrixFormat getFormat() const final { return MatrixFormat::LDU; }
};

}  // namespace graphene::matrix::ldu