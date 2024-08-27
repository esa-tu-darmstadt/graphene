#include "VertexTypes.hpp"

#include <unordered_map>

namespace graphene::codedsl {

constexpr VertexVectorType::VertexVectorType(TypeRef elementType)
    : Type(8), elementType_(elementType) {}

TypeRef VertexVectorType::elementType() const { return elementType_; }

bool VertexVectorType::isSubscriptable() const { return true; }

const VertexVectorType* VertexVectorType::get(TypeRef elementType) {
  static std::unordered_map<TypeRef, std::unique_ptr<VertexVectorType>>
      instances;

  auto it = instances.find(elementType);
  if (it == instances.end()) {
    auto instance = std::make_unique<VertexVectorType>(elementType);
    return (instances[elementType] = std::move(instance)).get();
  }
  return it->second.get();
}

std::string VertexVectorType::str() const {
  return fmt::format("::poplar::Vector<{}>", elementType_->str());
}

constexpr VertexInOutType::VertexInOutType(Direction direction,
                                           TypeRef elementType)
    : Type(8), elementType_(elementType), direction_(direction) {}

VertexInOutType::Direction VertexInOutType::kind() const { return direction_; }

TypeRef VertexInOutType::elementType() const {
  if (auto vectorType = dynamic_cast<const VertexVectorType*>(elementType_)) {
    return vectorType->elementType();
  }
  return elementType_;
}

VertexInOutType* VertexInOutType::get(Direction direction,
                                      TypeRef elementType) {
  static std::unordered_map<TypeRef, std::unique_ptr<VertexInOutType>>
      instances[3];
  size_t k = static_cast<size_t>(direction);

  auto it = instances[k].find(elementType);
  if (it == instances[k].end()) {
    auto instance = std::make_unique<VertexInOutType>(direction, elementType);
    return (instances[k][elementType] = std::move(instance)).get();
  }
  return it->second.get();
}

bool VertexInOutType::isSubscriptable() const {
  return dynamic_cast<const VertexVectorType*>(elementType_) != nullptr;
}

std::string VertexInOutType::str() const {
  std::string_view className;
  switch (direction_) {
    case Direction::Input:
      className = "Input";
      break;
    case Direction::Output:
      className = "Output";
      break;
    case Direction::InOut:
      className = "InOut";
      break;
  }
  return fmt::format("::poplar::{}<{}>", className, elementType_->str());
}

}  // namespace graphene::codedsl