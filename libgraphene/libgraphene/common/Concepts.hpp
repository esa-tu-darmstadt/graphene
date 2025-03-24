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

#include <poplar/Type.hpp>
#include <type_traits>

#include "libtwofloat/twofloat.hpp"
namespace graphene {
using doubleword = twofloat::two<float>;

template <typename T>
concept PoplarNativeType = requires {
  poplar::equivalent_device_type<T>::value;
};

template <typename T>
concept TwoFloatType = std::is_same_v<T, doubleword>;

template <typename T>
concept DoublePrecisionType = std::is_same_v<T, double>;

template <typename T>
concept DataType =
    PoplarNativeType<T> || TwoFloatType<T> || DoublePrecisionType<T>;

// Supports byte, short and int
template <typename T>
concept MatrixIndexType = std::is_same_v<T, uint8_t> ||
    std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>;

template <typename T>
concept FloatDataType =
    std::is_same_v<T, float> || std::is_same_v<T, double> || TwoFloatType<T>;

}  // namespace graphene