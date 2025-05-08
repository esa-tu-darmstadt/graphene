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

#include "libgraphene/common/Type.hpp"

namespace graphene::codedsl {

/**
 * @brief Represents a fixed-size array type in the Graphene DSL.
 */
class ArrayType : public Type {
 public:
  /**
   * @brief Constructs a fixed-size array type.
   * @param elementType The type of elements in the array.
   * @param size The size of the array.
   */
  constexpr ArrayType(TypeRef elementType, size_t size);

  TypeRef elementType() const override;
  bool isSubscriptable() const override;

  /**
   * @brief Gets or creates an ArrayType instance.
   * @param elementType The type of elements in the array.
   * @param size The size of the array.
   */
  static const ArrayType* get(TypeRef elementType, size_t size);

  std::string str() const override;

 private:
  TypeRef elementType_;
  size_t size_;
};

}  // namespace graphene::codedsl