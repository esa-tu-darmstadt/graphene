#pragma once

#include <array>
#include <cstdint>
#include <poplar/ArrayRef.hpp>
#include <poplar/Type.hpp>
#include <string_view>

#include "libtwofloat/twofloat.hpp"

namespace graphene {

inline size_t hash_combine(std::size_t seed) { return seed; }
template <typename T, typename... Rest>
inline size_t hash_combine(std::size_t seed, const T &v, Rest... rest) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return hash_combine(seed, rest...);
}
template <typename T, typename... Rest>
inline size_t hash(T v, Rest... rest) {
  size_t seed = 0;
  return hash_combine(seed, v, rest...);
}

}  // namespace graphene