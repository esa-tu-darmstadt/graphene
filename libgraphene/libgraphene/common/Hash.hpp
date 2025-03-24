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