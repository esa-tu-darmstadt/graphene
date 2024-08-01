#pragma once

#include <poplar/Graph.hpp>
#include <popops/Expr.hpp>
#include <vector>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Traits.hpp"

namespace graphene {

template <DataType Type>
class Value;

/**
 * @brief Represents an expression consisting of operations, constants, and
 * placeholders.
 *
 * This class has reference semantics, which means when passed-by-value, the
 * expression is copied but the underlying tensors are not.
 *
 * @tparam Type The data type of the expression.
 */
template <DataType Type>
class Expression {
  popops::expr::Any expr_;
  std::vector<poplar::Tensor> placeholders_;

 public:
  using DataType = Type;

  /** Constructors */
  Expression() = delete;

  /**
   * @brief Construct an expression from a popops expression and placeholders.
   *
   * @param expr The popops expression.
   * @param placeholders The placeholders used in the expression.
   */
  Expression(popops::expr::Any expr, std::vector<poplar::Tensor> placeholders)
    requires PoplarNativeType<Type>;

  /**
   * @brief Construct an expression from a constant value.
   *
   * @param value The constant value.
   */
  Expression(Type value)
    requires PoplarNativeType<Type>;

  /**
   * @brief Construct an expression from a poplar tensor.
   *
   * @param tensor The poplar tensor.
   */
  Expression(poplar::Tensor tensor);

  /**
   * @brief Get the underlying popops expression.
   */
  const popops::expr::Any &expr() const { return expr_; }

  /**
   * @brief Get the placeholders used in the expression.
   */
  const std::vector<poplar::Tensor> &placeholders() const {
    return placeholders_;
  }

  /**
   * @brief Get the shape of the expression.
   */
  std::vector<size_t> shape() const;

  /**
   * @brief Get the rank of the expression.
   */
  size_t rank() const;

  /**
   * @brief Get the number of elements in the expression.
   */
  size_t numElements() const;

  /**
   * @brief Cast the expression to a different data type.
   *
   * Only available to and from native poplar data types.
   * @tparam DestType The destination data type.
   * @return Expression<DestType> The casted expression.
   */
  template <typename DestType>
  Expression<DestType> cast() const
    requires PoplarNativeType<Type> && PoplarNativeType<DestType>
  {
    return Expression<DestType>(
        popops::expr::Cast(expr_, Traits<DestType>::PoplarType), placeholders_);
  }
};

/**
 * @brief Materializes an expression to a new \ref Value.
 *
 * @tparam Type The data type of the expression.
 * @param expr The expression to materialize.
 * @return Value<Type> The materialized value.
 */
template <PoplarNativeType Type>
Value<Type> materializeExpression(const Expression<Type> &expr);

/**
 * @brief Materializes an expression into an existing \ref Value.
 *
 * @tparam Type The data type of the expression.
 * @param expr The expression to materialize.
 * @return Value<Type> The materialized value.
 */
template <PoplarNativeType Type>
void materializeExpression(const Expression<Type> &expr, Value<Type> &dest);

}  // namespace graphene