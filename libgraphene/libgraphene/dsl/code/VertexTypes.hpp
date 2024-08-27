#pragma once
#include <spdlog/fmt/bundled/core.h>

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
   * @brief Gets or creates a VertexInOutType instance.
   * @param direction The direction of the data flow.
   * @param elementType The type of elements.
   * @return A pointer to the VertexInOutType instance.
   */
  static VertexInOutType* get(Direction direction, TypeRef elementType);

  bool isSubscriptable() const override;
  std::string str() const override;

 private:
  TypeRef elementType_;
  Direction direction_;
};

}  // namespace graphene::codedsl