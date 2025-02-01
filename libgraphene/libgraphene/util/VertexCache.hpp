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