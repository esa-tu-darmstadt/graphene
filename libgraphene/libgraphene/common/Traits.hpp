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

#include <cstdint>
#include <poplar/ArrayRef.hpp>
#include <poplar/Type.hpp>
#include <variant>

#include "libgraphene/common/Concepts.hpp"
#include "libtwofloat/algorithms.hpp"
#include "libtwofloat/twofloat.hpp"

namespace graphene {

template <DataType Type>
struct Traits {
 public:
  static_assert(sizeof(Type) == 0, "Traits not specialized for this type");
};
template <>
struct Traits<float> {
  static float inline zero() { return 0.0f; }
  static float inline one() { return 1.0f; }
  static float inline large() { return 1e-20f; }
  static float inline small() { return 1e-20f; }
  static float inline vsmall() { return 1e-37f; }
  static constexpr poplar::Type &PoplarType = poplar::FLOAT;
  using PoplarHostType = float;
};
template <>
struct Traits<double> {
  static double inline zero() { return 0.0; };
  static double inline one() { return 1.0; };
  static constexpr poplar::Type &PoplarType = poplar::LONGLONG;
  using PoplarHostType = signed long long;
};
template <>
struct Traits<doubleword> {
  static twofloat::two<float> inline zero() { return {0.0f, 0.0f}; };
  static twofloat::two<float> inline one() { return {1.0f, 0.0f}; };
  static constexpr poplar::Type &PoplarType = poplar::LONGLONG;
  using PoplarHostType = signed long long;
};
template <>
struct Traits<uint32_t> {
  static uint32_t inline zero() { return 0; }
  static uint32_t inline one() { return 1; }
  static constexpr poplar::Type &PoplarType = poplar::UNSIGNED_INT;
  using PoplarHostType = unsigned int;
};
template <>
struct Traits<int32_t> {
  static int32_t inline zero() { return 0; }
  static int32_t inline one() { return 1; }
  static constexpr poplar::Type &PoplarType = poplar::INT;
  using PoplarHostType = signed int;
};
template <>
struct Traits<int16_t> {
  static int16_t inline zero() { return 0; }
  static int16_t inline one() { return 1; }
  static constexpr poplar::Type &PoplarType = poplar::SHORT;
  using PoplarHostType = signed short;
};
template <>
struct Traits<uint16_t> {
  static uint16_t inline zero() { return 0; }
  static uint16_t inline one() { return 1; }
  static constexpr poplar::Type &PoplarType = poplar::UNSIGNED_SHORT;
  using PoplarHostType = unsigned short;
};
template <>
struct Traits<uint8_t> {
  static uint8_t inline zero() { return 0; }
  static uint8_t inline one() { return 1; }
  static constexpr poplar::Type &PoplarType = poplar::UNSIGNED_CHAR;
  using PoplarHostType = unsigned char;
};
template <>
struct Traits<int8_t> {
  static int8_t inline zero() { return 0; }
  static int8_t inline one() { return 1; }
  static constexpr poplar::Type &PoplarType = poplar::CHAR;
  using PoplarHostType = signed char;
};

template <>
struct Traits<bool> {
  static bool inline zero() { return false; }
  static bool inline one() { return true; }
  static constexpr poplar::Type &PoplarType = poplar::BOOL;
  // Bools are 8 bits long
  using PoplarHostType = uint8_t;
};

template <PoplarNativeType Type>
auto inline toPoplarHostType(Type val)
  requires PoplarNativeType<Type>
{
  return (Type)val;
}

auto inline toPoplarHostType(doubleword val) {
  return *reinterpret_cast<long long *>(&val);
}

auto inline toPoplarHostType(double val) {
  return *reinterpret_cast<long long *>(&val);
}

using PoplarDataTypeVariant =
    std::variant<bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                 uint64_t, int64_t, float>;

}  // namespace graphene