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

#include "libgraphene/dsl/code/types/ArrayType.hpp"

#include <fmt/format.h>

#include <boost/container_hash/extensions.hpp>
#include <unordered_map>

namespace graphene::codedsl {

constexpr ArrayType::ArrayType(TypeRef elementType, size_t size)
    : Type(elementType->size() * size),
      elementType_(elementType),
      size_(size) {}

TypeRef ArrayType::elementType() const { return elementType_; }

bool ArrayType::isSubscriptable() const { return true; }

const ArrayType* ArrayType::get(TypeRef elementType, size_t size) {
  static std::unordered_map<std::pair<TypeRef, size_t>,
                            std::unique_ptr<ArrayType>,
                            boost::hash<std::pair<TypeRef, size_t>>>
      instances;

  auto key = std::make_pair(elementType, size);
  auto it = instances.find(key);
  if (it == instances.end()) {
    auto instance = std::make_unique<ArrayType>(elementType, size);
    return (instances[key] = std::move(instance)).get();
  }
  return it->second.get();
}

std::string ArrayType::str() const {
  return fmt::format("{}[{}]", elementType_->str(), size_);
}

}  // namespace graphene::codedsl