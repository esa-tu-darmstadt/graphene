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

#include <cassert>
#include <cstddef>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
namespace graphene::matrix::host {
struct TileLayout {
  struct SeperatorRegion {
    std::set<size_t> dstProcs;
    size_t proc;

    // the global row indices of the seperator cells in this region
    std::vector<size_t> cells;

    SeperatorRegion() = delete;
    SeperatorRegion(std::set<size_t> dstProcs, size_t proc,
                    std::vector<size_t> cells)
        : dstProcs(dstProcs), proc(proc), cells(cells) {
      assert(!dstProcs.empty());
    }
  };

  struct HaloRegion {
    SeperatorRegion &srcRegion;

    // the global row indices of the halo cells in this region
    std::vector<size_t> cells;
    HaloRegion() = delete;
    HaloRegion(SeperatorRegion &srcRegion, std::vector<size_t> cells)
        : srcRegion(srcRegion), cells(cells) {
      assert(!srcRegion.dstProcs.empty());
    }

    size_t srcProc() const { return srcRegion.proc; }
  };

  size_t tileId;

  // The global row indices of the interior rows of the tile
  std::vector<size_t> interiorRows;

  std::vector<std::unique_ptr<SeperatorRegion>> seperatorRegions;
  std::vector<std::unique_ptr<HaloRegion>> haloRegions;

  // The local to global row mapping
  std::vector<size_t> localToGlobalRow;

  // The global to local row mapping
  std::unordered_map<size_t, size_t> globalToLocalRow;

  TileLayout(size_t tileId) : tileId(tileId) {}

  /** Returns the seperator region to the given processors */
  SeperatorRegion &getSeperatorRegionTo(std::set<size_t> dstProcs);

  /** Returns the halo region from the given seperator region */
  HaloRegion &getHaloRegionFrom(SeperatorRegion &srcRegion);

  /**  Calculates the local to global row mapping */
  void calculateRowMapping();

  /** Returns the number of interior rows */
  size_t numInteriorRows() const { return interiorRows.size(); }

  /** Returns the number of seperator rows */
  size_t numSeperatorRows() const {
    size_t num = 0;
    for (auto &region : seperatorRegions) {
      num += region->cells.size();
    }
    return num;
  }

  /** Returns the number of halo rows */
  size_t numHaloRows() const {
    size_t num = 0;
    for (auto &region : haloRegions) {
      num += region->cells.size();
    }
    return num;
  }

  /** Returns true if the given local row is an interior row */
  bool isInterior(size_t localRow) const {
    return localRow < interiorRows.size();
  }

  /**  Returns true if the given local row is a seperator row */
  bool isSeperator(size_t localRow) const {
    return localRow >= numInteriorRows() &&
           localRow < numInteriorRows() + numSeperatorRows();
  }

  /** Returns true if the given local row is a halo row */
  bool isHalo(size_t localRow) const {
    return localRow >= numInteriorRows() + numSeperatorRows() &&
           localRow < numInteriorRows() + numSeperatorRows() + numHaloRows();
  }
};

}  // namespace graphene::matrix::host