#pragma once

#include <array>
#include <cstdint>
#include <poplar/ArrayRef.hpp>
#include <poplar/Type.hpp>
#include <string_view>

#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/Type.hpp"
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

namespace std {
template <typename T>
struct hash;

template <>
struct hash<::twofloat::two<float>> {
  size_t operator()(const ::twofloat::two<float> &value) const {
    return std::hash<double>{}(value.eval<double>());
  }
};

// template <>
// struct hash<graphene::TypeRef> {
//   size_t operator()(graphene::TypeRef value) const {
//     return std::hash<string>{}(value->str());
//   }
// };

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(const std::vector<T> &value) const {
    size_t h = 0;
    for (const auto &v : value) {
      h = graphene::hash_combine(h, v);
    }
    return h;
  }
};

template <>
struct hash<graphene::TensorShape> {
  size_t operator()(const graphene::TensorShape &value) const {
    size_t h = 0;
    for (const auto &v : value) {
      h = graphene::hash_combine(h, v);
    }
    return h;
  }
};

template <>
struct hash<graphene::FirstDimDistribution> {
  size_t operator()(const graphene::FirstDimDistribution &value) const {
    size_t h = 0;
    for (auto [tile, value] : value) {
      h = graphene::hash_combine(h, tile, value);
    }
    return h;
  }
};

template <>
struct hash<graphene::DistributedShape> {
  size_t operator()(const graphene::DistributedShape &value) const {
    return graphene::hash(value.globalShape(), value.firstDimDistribution());
  }
};
}  // namespace std