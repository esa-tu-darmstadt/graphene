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

#include <optional>

#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/dsl/tensor/Traits.hpp"
#include "libgraphene/matrix/Addressing.hpp"
#include "libgraphene/matrix/Coloring.hpp"

namespace graphene::matrix::ldu {

/// \brief Represents the addressing of a LDU matrix. The LDU format is used in
/// OpenFOAM and is handy to represent meshes and matrices in a finite volume
/// methods.
/// \details The combination of lowerAddr, upperAddr, and ownerStartAddr allows
/// to address the faces/coefficients that are owned by a cell/row. This only
/// coveres half of the addressing. The other half - i.e. the faces/coefficients
/// that are neighboring a cell/row) - is addressed by the \ref
/// neighbourStartPtr and
/// \ref neighbourColInd arrays. These arrays form a CRS-like sparse matrix that
/// allow to look up the neighboring cells/rows of a cell/row.
/// See the OpenFOAM documentation for more details:
/// https://openfoamwiki.net/index.php/OpenFOAM_guide/Matrices_in_OpenFOAM
struct LDUAddressing : matrix::Addressing {
  /// For each face, the index of the cell that owns the face
  Tensor lowerAddr;

  /// For each face, the index of the cell that is neighboring the face
  /// (i.e. the cell that is not the owner of the face)
  Tensor upperAddr;

  /// For each cell, the index of the first face that is owned by the cell
  Tensor ownerStartAddr;

  /// For each cell, the index of its first element in the \ref neighbourColInd
  /// array. Called \ref losortStartAddr in OpenFOAM.
  Tensor neighbourStartPtr;

  /// The indices of the faces neighboring a cell. Indexed by the
  /// \ref neighbourStartPtr array. Called \ref losortAddr in OpenFOAM.
  Tensor neighbourColInd;

  /// For each boundary face, the index of the cell that the face is connected
  /// to. Each boundary face is connected to a single cell.
  Tensor patchAddr;

  /** Optional coloring of the matrix */
  std::optional<Coloring> coloring;

  LDUAddressing(Tensor lowerAddr, Tensor upperAddr, Tensor ownerStartAddr,
                Tensor neighbourStartPtr, Tensor neighbourColInd,
                Tensor patchAddr,
                std::optional<Coloring> coloring = std::nullopt)
      : lowerAddr(std::move(lowerAddr)),
        upperAddr(std::move(upperAddr)),
        ownerStartAddr(std::move(ownerStartAddr)),
        neighbourStartPtr(std::move(neighbourStartPtr)),
        neighbourColInd(std::move(neighbourColInd)),
        patchAddr(std::move(patchAddr)),
        coloring(std::move(coloring)) {
    assert(this->lowerAddr.numElements() == this->upperAddr.numElements());
    assert(this->ownerStartAddr.numElements() ==
           this->neighbourStartPtr.numElements());
  }

  // Addressings should usually not be copied. Deleting the copy
  // constructor and assignment operator to prevent us from accidentally
  // copying them.
  LDUAddressing(const LDUAddressing&) = delete;
  LDUAddressing(LDUAddressing&&) = default;
  LDUAddressing& operator=(const LDUAddressing&) = delete;
  LDUAddressing& operator=(LDUAddressing&&) = default;
};

}  // namespace graphene::matrix::ldu