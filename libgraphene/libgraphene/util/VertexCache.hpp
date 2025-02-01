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

#include <filesystem>
#include <optional>
#include <unordered_map>
namespace graphene {
class Runtime;

class VertexCache {
  friend class Runtime;

  VertexCache() = delete;

  static bool initCache(std::filesystem::path cacheDir);

  static bool saveCache();
  static bool restoreCache();

 public:
  struct Entry {
    std::string vertexName;
    std::filesystem::path srcPath;
    std::filesystem::path elfPath;
  };

  /// Checks if a vertex is already cached
  static bool isCached(size_t hash);

  /// Looks up a vertex in the cache
  static std::optional<Entry> lookup(size_t hash);

  /// Returns a unique name for a vertex that is not used yet
  static std::string getUniqueVertexName(size_t hash);

  static std::string getVertexNamePlaceholder();

  /// Return the path that the source file of a vertex with a given name would
  /// have
  static std::filesystem::path getVertexSourcePath(size_t hash);

  static std::filesystem::path getVertexObjectPath(size_t hash);

  /// Inserts a vertex into the cache
  static void insert(size_t hash, std::string vertexName);

 private:
  static std::unordered_map<size_t, Entry> cache_;
  static std::filesystem::path cacheDir_;
};
}  // namespace graphene