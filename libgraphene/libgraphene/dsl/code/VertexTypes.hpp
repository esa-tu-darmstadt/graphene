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
 * @brief Represents the poplar::Vector type from the Poplar runtime.
 */
class VertexVectorType : public Type {
 public:
  /**
   * @brief Constructs a VertexVectorType.
   * @param elementType The type of elements in the vector.
   */
  constexpr VertexVectorType(TypeRef elementType);

  TypeRef elementType() const override;
  bool isSubscriptable() const override;

  /**
   * @brief Gets or creates a VertexVectorType instance.
   * @param elementType The type of elements in the vector.
   * @return A pointer to the VertexVectorType instance.
   */
  static const VertexVectorType* get(TypeRef elementType);

  std::string str() const override;
  bool hasFunction(std::string func) const override;
  TypeRef functionReturnType(std::string func) const override;

 private:
  TypeRef elementType_;
};

/**
 * @brief Represents the poplar::Input, poplar::Output, and poplar::InOut types
 * from the Poplar runtime.
 */
class VertexInOutType : public Type {
 public:
  enum Direction : size_t { Input = 0, Output, InOut };

  /**
   * @brief Constructs a VertexInOutType.
   * @param direction The direction of the data flow.
   * @param elementType The type of elements.
   */
  constexpr VertexInOutType(Direction direction, TypeRef elementType);

  Direction kind() const;
  TypeRef elementType() const override;

  /**
   * @brief Gets native poplar type, which is the innermost type.
   *  I.e., poplar::Input<poplar::Vector<T>> -> T and poplar::Input<T> -> T.
   */
  TypeRef nativePoplarType() const;

  /**
   * @brief Gets or creates a VertexInOutType instance.
   * @param direction The direction of the data flow.
   * @param elementType The type of elements.
   * @return A pointer to the VertexInOutType instance.
   */
  static VertexInOutType* get(Direction direction, TypeRef elementType);

  bool isSubscriptable() const override;
  std::string str() const override;
  bool hasFunction(std::string func) const override;
  TypeRef functionReturnType(std::string func) const override;

 private:
  TypeRef elementType_;
  Direction direction_;
};

}  // namespace graphene::codedsl