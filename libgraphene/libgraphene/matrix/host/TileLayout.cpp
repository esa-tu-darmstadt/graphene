#include "libgraphene/matrix/host/TileLayout.hpp"

#include <cstddef>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Traits.hpp"

namespace graphene::matrix::host {
TileLayout::SeperatorRegion &TileLayout::getSeperatorRegionTo(
    std::set<size_t> dstProcs) {
  assert(!dstProcs.empty());
  // Try to find an existing region
  for (auto &region : seperatorRegions) {
    if (region->dstProcs == dstProcs) return *region;
  }

  // Create a new region
  seperatorRegions.push_back(
      make_unique<SeperatorRegion>(dstProcs, tileId, std::vector<size_t>()));
  return *seperatorRegions.back();
}

TileLayout::HaloRegion &TileLayout::getHaloRegionFrom(
    SeperatorRegion &srcRegion) {
  // Try to find an existing region from the processor that owns the
  // seperator
  for (auto &region : haloRegions) {
    if (&region->srcRegion == &srcRegion) return *region;
  }

  // Create a new region
  haloRegions.push_back(
      make_unique<HaloRegion>(srcRegion, std::vector<size_t>()));
  return *haloRegions.back();
}

void TileLayout::calculateRowMapping() {
  // First, determine the number of rows (interior, seperator,
  // and halo) to reserve enough space
  size_t numRows = interiorRows.size();
  for (auto &seperatorRegion : seperatorRegions) {
    numRows += seperatorRegion->cells.size();
  }
  for (auto &haloRegion : haloRegions) {
    numRows += haloRegion->cells.size();
  }

  // Reserve space for the local to global row mapping
  localToGlobalRow.reserve(numRows);

  // Add the interior rows first
  for (size_t i = 0; i < interiorRows.size(); i++) {
    localToGlobalRow.push_back(interiorRows[i]);
  }
  // Add the seperator rows
  for (auto &seperatorRegion : seperatorRegions) {
    for (size_t i = 0; i < seperatorRegion->cells.size(); i++) {
      localToGlobalRow.push_back(seperatorRegion->cells[i]);
    }
  }
  // Add the halo rows
  for (auto &haloRegion : haloRegions) {
    for (size_t i = 0; i < haloRegion->cells.size(); i++) {
      localToGlobalRow.push_back(haloRegion->cells[i]);
    }
  }

  // Create the global to local row mapping
  for (size_t i = 0; i < localToGlobalRow.size(); i++) {
    globalToLocalRow[localToGlobalRow[i]] = i;
  }
}
}  // namespace graphene::matrix::host
