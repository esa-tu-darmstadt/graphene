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

#include "libgraphene/util/VertexCache.hpp"

#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

using namespace graphene;
using namespace nlohmann;

std::unordered_map<size_t, VertexCache::Entry> VertexCache::cache_;
std::filesystem::path VertexCache::cacheDir_;

namespace {
std::filesystem::path getCacheDictionaryFilename(
    std::filesystem::path cacheDir) {
  return cacheDir / "vertices.json";
}

/// Returns the 16-character hex string representation of a hash
std::string hashToString(size_t hash) {
  std::ostringstream oss;
  oss << std::setw(16) << std::setfill('0') << std::hex << hash;
  return oss.str();
}
}  // namespace

bool VertexCache::initCache(std::filesystem::path cacheDir) {
  cacheDir_ = cacheDir;
  if (!restoreCache()) {
    // If the cache could not be restored, create an empty one
    return saveCache();
  }

  // Remove invalid entries
  for (auto it = cache_.begin(); it != cache_.end();) {
    auto& [hash, entry] = *it;
    if (!std::filesystem::exists(entry.srcPath) ||
        !std::filesystem::exists(entry.elfPath)) {
      spdlog::trace("Removing invalid vertex {} from cache", entry.vertexName);
      it = cache_.erase(it);
    } else {
      ++it;
    }
  }
  spdlog::trace("Restored {} vertices from cache", cache_.size());

  return true;
}

bool VertexCache::saveCache() {
  json cache;
  cache["vertices"] = json::object();
  cache["version"] = 1;
  for (auto& [hash, entry] : cache_) {
    cache["vertices"][std::to_string(hash)] = {{"vertexName", entry.vertexName},
                                               {"srcPath", entry.srcPath},
                                               {"elfPath", entry.elfPath}};
  }
  std::ofstream cacheFile(getCacheDictionaryFilename(cacheDir_),
                          std::ios::trunc);
  if (cacheFile.fail()) return false;

  cacheFile << cache.dump(2);
  spdlog::trace("Saved vertex cache to {}", cacheDir_.string());
  return true;
}

bool VertexCache::restoreCache() {
  auto cacheFileName = getCacheDictionaryFilename(cacheDir_);
  if (std::filesystem::exists(cacheFileName)) {
    std::ifstream cacheFile(cacheFileName);
    json cache;
    cacheFile >> cache;
    for (auto& [hash, entry] : cache["vertices"].items()) {
      cache_[std::stoull(hash)] = {entry["vertexName"], entry["srcPath"],
                                   entry["elfPath"]};
    }
    spdlog::trace("Restored vertex cache from {}", cacheDir_.string());
    return true;
  }
  spdlog::trace("No vertex cache found in {}", cacheDir_.string());
  return false;
}

bool VertexCache::isCached(size_t hash) {
  return cache_.find(hash) != cache_.end();
}

std::optional<VertexCache::Entry> VertexCache::lookup(size_t hash) {
  if (auto it = cache_.find(hash); it != cache_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::string VertexCache::getUniqueVertexName(size_t hash) {
  return "vertex" + hashToString(hash);
}

std::string VertexCache::getVertexNamePlaceholder() {
  // must have the same length as the real vertex name
  return "vertexPLACEHOLDER12345";
}

void VertexCache::insert(size_t hash, std::string vertexName) {
  Entry entry{vertexName, getVertexSourcePath(hash), getVertexSourcePath(hash)};
  cache_[hash] = entry;

  saveCache();
}

std::filesystem::path VertexCache::getVertexSourcePath(size_t hash) {
  return cacheDir_ / (getUniqueVertexName(hash) + ".cpp");
}

std::filesystem::path VertexCache::getVertexObjectPath(size_t hash) {
  return cacheDir_ / (getUniqueVertexName(hash) + ".elf");
}