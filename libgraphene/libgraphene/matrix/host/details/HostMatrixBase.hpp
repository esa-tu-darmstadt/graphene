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
#include "libgraphene/matrix/MatrixFormat.hpp"
#include "libgraphene/matrix/host/DistributedTileLayout.hpp"
#include "libgraphene/matrix/host/details/MatrixMarket.hpp"

namespace graphene::matrix {
class Matrix;

namespace host {

class HostMatrixBase : public Runtime::HostResource,
                       public std::enable_shared_from_this<HostMatrixBase>,
                       public DistributedTileLayout {
 public:
  HostMatrixBase(size_t numTiles, std::string name)
      : DistributedTileLayout(), name_(std::move(name)), numTiles_(numTiles) {}

  HostMatrixBase(const HostMatrixBase &other) = delete;
  HostMatrixBase(HostMatrixBase &&other) = delete;

  virtual ~HostMatrixBase() = default;

  virtual matrix::Matrix copyToTile() = 0;
  virtual MatrixFormat getFormat() const = 0;

  const std::string &name() const { return name_; }
  size_t numTiles() const override { return numTiles_; }

 private:
  std::string name_;
  size_t numTiles_;
};

}  // namespace host
}  // namespace graphene::matrix