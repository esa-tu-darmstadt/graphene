#pragma once

#include <poplar/Type.hpp>

#include "libtwofloat/twofloat.hpp"
namespace graphene {
template <typename T>
concept PoplarNativeType =
    requires { poplar::equivalent_device_type<T>::value; };

template <typename T>
concept TwoFloatType = std::is_same_v<T, double>;

template <typename T>
concept DataType = PoplarNativeType<T> || TwoFloatType<T>;

// Supports byte, short and int
template <typename T>
concept MatrixIndexType =
    std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, uint32_t>;

}  // namespace graphene