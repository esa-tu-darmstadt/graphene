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
#include "libgraphene/matrix/details/MatrixBase.hpp"
#include "libgraphene/matrix/details/ldu/LDUAddressing.hpp"
#include "libgraphene/matrix/host/DistributedTileLayout.hpp"

namespace graphene::matrix::ldu {

/**
 * @brief A class representing a matrix in LDU (lower-diagonal-upper) format.
 */
struct LDUMatrix : public MatrixBase {
  std::shared_ptr<LDUAddressing>
      addressing;                  ///< Addressing scheme for the LDU matrix.
  Tensor offDiagonalCoefficients;  ///< Off-diagonal coefficients of the matrix.
  Tensor diagonalCoefficients;     ///< Diagonal coefficients of the matrix.

  /**
   * @brief Construct a new LDU matrix.
   *
   * @param hostMatrix The host matrix this LDU matrix is based on.
   * @param addressing The addressing scheme for the LDU matrix.
   * @param offDiagonalCoefficients The off-diagonal coefficients of the matrix.
   * @param diagonalCoefficients The diagonal coefficients of the matrix.
   */
  LDUMatrix(const host::DistributedTileLayout &tileLayout,
            std::shared_ptr<LDUAddressing> addressing,
            Tensor offDiagonalCoefficients, Tensor diagonalCoefficients)
      : MatrixBase(tileLayout),
        addressing(std::move(addressing)),
        offDiagonalCoefficients(std::move(offDiagonalCoefficients)),
        diagonalCoefficients(std::move(diagonalCoefficients)) {}

  Tensor spmv(Tensor &x, TypeRef destType = nullptr,
              TypeRef intermediateType = nullptr,
              bool withHalo = false) const override;

  Tensor residual(Tensor &x, const Tensor &b, TypeRef destType = nullptr,
                  TypeRef intermediateType = nullptr,
                  bool withHalo = false) const override;

  MatrixFormat getFormat() const final { return MatrixFormat::LDU; }
};

}  // namespace graphene::matrix::ldu