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

#include "InputExprDeduplication.hpp"

#include "libgraphene/dsl/tensor/details/Expressions.hpp"

namespace graphene::optimization {

void InputExprDeduplication::optimize(detail::ExpressionBase& expr) {
  std::vector<std::pair<poplar::Tensor, std::shared_ptr<detail::InputExpr>>>
      inputCache;
  size_t deduplicationCount = 0;
  optimizeRecursive(expr, inputCache, deduplicationCount);
  spdlog::debug("InputExprDeduplication: Deduplicated {} InputExpr instances",
                deduplicationCount);
}

std::string InputExprDeduplication::getName() const {
  return "InputExprDeduplication";
}

void InputExprDeduplication::optimizeRecursive(
    detail::ExpressionBase& expr,
    std::vector<std::pair<poplar::Tensor, std::shared_ptr<detail::InputExpr>>>&
        inputCache,
    size_t& deduplicationCount) {
  for (size_t i = 0; i < expr.numChildren(); ++i) {
    std::shared_ptr<detail::ExpressionBase> child = expr.child(i);
    if (auto inputExprChild =
            std::dynamic_pointer_cast<detail::InputExpr>(child)) {
      poplar::Tensor inputTensor = inputExprChild->tensor();

      auto it = std::find_if(inputCache.begin(), inputCache.end(),
                             [&inputTensor](const auto& pair) {
                               return pair.first == inputTensor;
                             });
      if (it == inputCache.end()) {
        // Cache the InputExpr if it is not already cached
        inputCache.emplace_back(inputTensor, inputExprChild);
      } else {
        // Replace the current InputExpr with the cached one
        expr.replaceChild(i, it->second);
        deduplicationCount++;
      }
    } else {
      optimizeRecursive(*child, inputCache, deduplicationCount);
    }
  }
}
}  // namespace graphene::optimization