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

#include <memory>
#include <vector>

#include "libgraphene/dsl/tensor/details/Expressions.hpp"

namespace graphene::optimization {

class ExpressionOptimization {
 public:
  virtual ~ExpressionOptimization() = default;

  virtual void optimize(detail::ExpressionBase& expr) = 0;

  virtual std::string getName() const = 0;
};

class OptimizationPipeline {
  std::vector<std::unique_ptr<ExpressionOptimization>> optimizations_;

 public:
  OptimizationPipeline() = default;
  ~OptimizationPipeline() = default;

  OptimizationPipeline(const OptimizationPipeline&) = delete;
  OptimizationPipeline& operator=(const OptimizationPipeline&) = delete;

  OptimizationPipeline(OptimizationPipeline&&) = default;
  OptimizationPipeline& operator=(OptimizationPipeline&&) = default;

  void addOptimization(std::unique_ptr<ExpressionOptimization> optimization);

  void optimize(detail::ExpressionBase& expr);

  size_t size() const;
  void clear();
};

}  // namespace graphene::optimization