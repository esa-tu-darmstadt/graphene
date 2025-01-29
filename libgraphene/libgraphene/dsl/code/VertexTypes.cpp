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
  // FIXME: Make configurable
  return fmt::format("::poplar::Vector<{}, poplar::VectorLayout::SPAN, 8>",
                     elementType_->str());
}

bool VertexVectorType::hasFunction(std::string func) const {
  return func == "size";
}

TypeRef VertexVectorType::functionReturnType(std::string func) const {
  if (func == "size") {
    return Type::INT32;
  }
  throw std::runtime_error("Function not found.");
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

bool VertexInOutType::hasFunction(std::string func) const {
  if (auto vectorType = dynamic_cast<const VertexVectorType*>(elementType_)) {
    return vectorType->hasFunction(func);
  }
  return false;
}

TypeRef VertexInOutType::functionReturnType(std::string func) const {
  if (auto vectorType = dynamic_cast<const VertexVectorType*>(elementType_)) {
    return vectorType->functionReturnType(func);
  }
  throw std::runtime_error("Function not found.");
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

TypeRef VertexInOutType::nativePoplarType() const {
  if (auto vectorType = dynamic_cast<const VertexVectorType*>(elementType_)) {
    return vectorType->elementType();
  }
  return elementType_;
}

}  // namespace graphene::codedsl