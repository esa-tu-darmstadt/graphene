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

#include <optional>
#include <poplar/Interval.hpp>
#include <poplar/Tensor.hpp>
#include <vector>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Traits.hpp"
#include "libgraphene/common/Type.hpp"

namespace graphene {
enum class ReduceOperation {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND,  ///< Only supports boolean operands.
  LOGICAL_OR,   ///< Only supports boolean operands.
  SQUARE_ADD,   ///< Squares each element before applying ADD reduction.
};

namespace detail {
class ExpressionBase;
}
class Tensor;

/**
 * @brief Represents an expression that can be evaluated to a tensor. This class
 * has reference semantics.
 */
class Expression {
  std::unique_ptr<detail::ExpressionBase> expr_;

  // The tile mapping and shape of the tensor. Cached for performance reasons.
  mutable std::optional<TileMapping> tileMapping_;
  mutable std::optional<DistributedShape> shape_;

 public:
  /** Constructors */
  Expression() = delete;
  virtual ~Expression();

  /** Copy and move constructors */
  Expression(const Expression &expr);
  Expression(Expression &&expr) noexcept;

  /** Copy and move assignments */
  Expression &operator=(const Expression &expr);
  Expression &operator=(Expression &&expr) noexcept;

  /**
   * @brief Construct an expression from an inner expression.
   */
  explicit Expression(std::unique_ptr<detail::ExpressionBase> expr);

  /**
   * @brief Construct an expression from a tensor.
   */
  Expression(const Tensor &tensor);

  /**
   * @brief Construct an expression from a constant value
   * The type of the expression is inferred from the value.
   *
   * @param value The constant value.
   */
  template <DataType Type>
  Expression(Type value);

  /**
   * @brief Construct an expression from a poplar tensor. The type of the
   * expression is inferred from the tensor, or explicitly provided if the
   * poplar type is ambiguous (i.e., LONGLONG can be int64_t, double, or
   * two<float>).
   *
   * @param tensor The poplar tensor.
   */
  Expression(poplar::Tensor tensor, TypeRef type);

  /**
   * @brief Get the shape of the tensor. The shape is cached for performance
   * reasons.
   *
   * @return TileMapping The tile mapping.
   */
  const DistributedShape &shape() const;

  /**
   * @brief Get the tile mapping of the tensor. The tile mapping is cached for
   * performance reasons.
   *
   * @return TileMapping The tile mapping.
   */
  const TileMapping &tileMapping() const;

  /**
   * @brief Get the rank of the expression.
   */
  size_t rank() const;

  /**
   * @brief Get the number of elements in the expression.
   */
  size_t numElements() const;

  /**
   * @brief Get the data type of the expression.
   */
  TypeRef type() const;

  /**
   * @brief Get the expression as a string.
   */
  std::string asString() const;

  /**
   * @brief Cast the expression to a different data type.
   */
  Expression cast(TypeRef destType) const;

  /**
   * @brief Broadcast the expression to a new shape.
   */
  Expression broadcast(DistributedShape shape) const;

  /**
   * @brief Reduce the value along the given dimensions.
   */
  Expression reduce(size_t dim = 0,
                    ReduceOperation op = ReduceOperation::ADD) const;

  /**
   * @brief Reduce the value along the given dimensions on each tile. The shape
   * of the resulting tensor has the size of the reduced dimension set to the
   * number of tiles.
   */
  Expression reducePerTile(size_t dim, ReduceOperation op) const;

  /**
   * @brief Reduce the value along the given dimensions on each worker thread.
   * The shape of the resulting tensor has the size of the reduced dimension set
   * to the number of tiles times the number of worker threads per tile (6).
   */
  Expression reducePerWorker(size_t dim, ReduceOperation op) const;

  /**
   * @brief Reduce the tensor across the first dimension in groups of tiles.
   * @details Each group is of size \p groupSize in the tile dimension.
   * If \p groupSize is numTilesPerIPU, this effectively performs a per-IPU
   * reduction. If \p groupSize is numTiles, then it's a global reduction.
   *
   * @param groupSize The number of tiles per group.
   * @param op The reduction operation to apply.
   * @return Tensor The reduced tensor.
   */
  Expression reduceGrouped(size_t groupSize, ReduceOperation op) const;

  /**
   * @brief Permute the dimensions of the expression. Currently, only supports
   * permutation in which the first dimension is not moved.
   */
  Expression permute(std::vector<size_t> permutation) const;

  /**
   * @brief Materialize the expression to a new \ref Value.
   * @return Tensor The materialized value.
   */
  Tensor materialize() const;

  /**
   * @brief Materialize the expression if it is not already materialized.
   */
  Tensor materializeIfNecessary() const;

  /**
   * @brief Get the root of the underlying expression tree.
   */
  detail::ExpressionBase &base() const { return *expr_; }

  /**
   * @brief Prints the expression to the output stream. May require
   * materializing the expression.
   */
  void print(std::string name = "") const;
};

/**
 * @brief Materializes an expression to a new \ref Tensor.
 *
 * @tparam Type The data type of the expression.
 * @param expr The expression to materialize.
 * @return Tensor The materialized value.
 */
Tensor materializeExpression(const Expression &expr);

/**
 * @brief Materializes an expression into an existing \ref Tensor.
 *
 * @tparam Type The data type of the expression.
 * @param expr The expression to materialize.
 * @return Tensor The materialized value.
 */
void materializeExpression(const Expression &expr, Tensor &dest);

}  // namespace graphene