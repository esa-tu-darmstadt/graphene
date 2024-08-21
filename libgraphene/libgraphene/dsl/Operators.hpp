#pragma once

#include <optional>
#include <poputil/VertexTemplates.hpp>

#include "libgraphene/dsl/Expression.hpp"
#include "libgraphene/dsl/Traits.hpp"
#include "libgraphene/dsl/Value.hpp"
#include "libgraphene/dsl/details/OperatorTraits.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"

namespace graphene {

namespace detail {
/**
 * @brief Helper function to wrap a DataType in an Expression if it's not
 * already one.
 *
 * @tparam T The type of the value.
 * @param value The value to wrap.
 * @return Expression<typename T::DataType> The wrapped expression.
 */
template <DataTypeOrExpression T, DataType DataT = unwrap_expression<T>::type>
Expression<DataT> wrapInExpression(const T &value) {
  if constexpr (is_expression_v<T>) {
    return Expression<typename T::DataType>(value);
  } else {
    return Expression<T>(value);
  }
}

template <DataType T>
Value<T> wrapInValue(T value) {
  return Value<T>(value);
}

template <DataType T>
const Value<T> &wrapInValue(const Value<T> &value) {
  return value;
}

/**
 * @brief Shift placeholder indices in an expression.
 *
 * @param nodeConst The expression node to modify.
 * @param offset The offset to apply to placeholder indices.
 */
void shiftPlaceHolderIndices(const popops::expr::Expr *nodeConst,
                             unsigned int offset);

/**
 * @brief Broadcast two shapes according to numpy broadcasting rules if
 * possible.
 *
 * @param shape1 The first shape.
 * @param shape2 The second shape.
 * @return std::optional<std::vector<size_t>> The broadcasted shape if the
 * shapes are compatible, otherwise std::nullopt.
 */
std::optional<std::vector<size_t>> broadcastShapes(std::vector<size_t> shape1,
                                                   std::vector<size_t> shape2);

}  // namespace detail

#define GRAPHENE_DEFINE_EXPR_UNARY_OP(name, op)                                \
  namespace ops {                                                              \
                                                                               \
  template <PoplarNativeType T>                                                \
    requires CompatibleTypeForUnaryOp<popops::expr::UnaryOpType::op, T>        \
  auto name(const Expression<T> &value) {                                      \
    using ReturnType =                                                         \
        typename unary_op_return_type<popops::expr::UnaryOpType::op, T>::type; \
                                                                               \
    return Expression<ReturnType>(popops::expr::name(value.expr()),            \
                                  value.placeholders());                       \
  }                                                                            \
  }  // namespace ops

#define GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(name, op, symbol)           \
  GRAPHENE_DEFINE_EXPR_UNARY_OP(name, op)                                    \
  template <PoplarNativeType T>                                              \
    requires ops::CompatibleTypeForUnaryOp<popops::expr::UnaryOpType::op, T> \
  auto operator symbol(const Expression<T> &value) {                         \
    return ops::name(value);                                                 \
  }

GRAPHENE_DEFINE_EXPR_UNARY_OP(Abs, ABSOLUTE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Asin, ASIN)
GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(BitwiseNot, BITWISE_NOT, ~)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Cbrt, CBRT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Erf, ERF)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Ceil, CEIL)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Cos, COS)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Exp, EXPONENT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Expm1, EXPONENT_MINUS_ONE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Exp2, EXPONENT2)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Floor, FLOOR)
GRAPHENE_DEFINE_EXPR_UNARY_OP(GeluErf, GELU_ERF)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Inv, INVERSE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(IsFinite, IS_FINITE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(IsInf, IS_INF)
GRAPHENE_DEFINE_EXPR_UNARY_OP(IsNaN, IS_NAN)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Log, LOGARITHM)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Log1p, LOGARITHM_ONE_PLUS)
GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Not, LOGICAL_NOT, !)
GRAPHENE_DEFINE_EXPR_UNARY_OP_AND_SYMBOL(Neg, NEGATE, -)
GRAPHENE_DEFINE_EXPR_UNARY_OP(NearbyInt, NEARBY_INT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Signum, SIGNUM)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Sin, SIN)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Tan, TAN)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Tanh, TANH)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Round, ROUND)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Trunc, TRUNC)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Sqrt, SQRT)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Square, SQUARE)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Sigmoid, SIGMOID)
GRAPHENE_DEFINE_EXPR_UNARY_OP(Rsqrt, RSQRT)

#define GRAPHENE_DEFINE_EXPR_BINARY_OP(name, op)                              \
  namespace ops {                                                             \
                                                                              \
  template <PoplarNativeTypeOrExpression T1, PoplarNativeTypeOrExpression T2> \
    requires AtLeastOneExpression<T1, T2> &&                                  \
             CompatibleTypesForBinaryOp<popops::expr::BinaryOpType::op, T1,   \
                                        T2>                                   \
  auto name(const T1 &lhs, const T2 &rhs) {                                   \
    auto lhsExpr = detail::wrapInExpression(lhs);                             \
    auto rhsExpr = detail::wrapInExpression(rhs);                             \
                                                                              \
    if (!detail::broadcastShapes(lhsExpr.shape(), rhsExpr.shape()))           \
      throw std::runtime_error("Shapes are not compatible");                  \
                                                                              \
    using LeftType = typename decltype(lhsExpr)::DataType;                    \
    using RightType = typename decltype(rhsExpr)::DataType;                   \
    using ResultType = binary_op_return_type<popops::expr::BinaryOpType::op,  \
                                             LeftType, RightType>::type;      \
                                                                              \
    std::vector<poplar::Tensor> placeholders = lhsExpr.placeholders();        \
    placeholders.insert(placeholders.end(), rhsExpr.placeholders().begin(),   \
                        rhsExpr.placeholders().end());                        \
                                                                              \
    if (lhsExpr.placeholders().size() > 0 &&                                  \
        rhsExpr.placeholders().size() > 0) {                                  \
      popops::expr::Expr &rhsInnerExpr =                                      \
          const_cast<popops::expr::Any &>(rhsExpr.expr());                    \
      detail::shiftPlaceHolderIndices(&rhsInnerExpr,                          \
                                      lhsExpr.placeholders().size());         \
    }                                                                         \
                                                                              \
    return Expression<ResultType>(                                            \
        popops::expr::name(lhsExpr.expr(), rhsExpr.expr()), placeholders);    \
  }                                                                           \
  }  // namespace ops

#define GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(name, op, symbol)           \
  GRAPHENE_DEFINE_EXPR_BINARY_OP(name, op)                                    \
  template <PoplarNativeTypeOrExpression T1, PoplarNativeTypeOrExpression T2> \
    requires AtLeastOneExpression<T1, T2> &&                                  \
             ops::CompatibleTypesForBinaryOp<popops::expr::BinaryOpType::op,  \
                                             T1, T2>                          \
  auto operator symbol(const T1 &lhs, const T2 &rhs) {                        \
    return ops::name(lhs, rhs);                                               \
  }

GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Add, ADD, +)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Atan2, ATAN2)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseAnd, BITWISE_AND, &)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseOr, BITWISE_OR, |)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(BitwiseXor, BITWISE_XOR, ^)
GRAPHENE_DEFINE_EXPR_BINARY_OP(BitwiseXnor, BITWISE_XNOR)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Divide, DIVIDE, /)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Equal, EQUAL, ==)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Gte, GREATER_THAN_EQUAL, >=)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Gt, GREATER_THAN, >)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Lte, LESS_THAN_EQUAL, <=)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(And, LOGICAL_AND, &&)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Or, LOGICAL_OR, ||)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Lt, LESS_THAN, <)
GRAPHENE_DEFINE_EXPR_BINARY_OP(InvStdDevToVariance, INV_STD_DEV_TO_VARIANCE)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Max, MAXIMUM)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Min, MINIMUM)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Mul, MULTIPLY, *)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(NotEqual, NOT_EQUAL, !=)
GRAPHENE_DEFINE_EXPR_BINARY_OP(Pow, POWER)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Rem, REMAINDER, %)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Shl, SHIFT_LEFT, <<)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Shr, SHIFT_RIGHT, >>)
GRAPHENE_DEFINE_EXPR_BINARY_OP(ShrSE, SHIFT_RIGHT_SIGN_EXTEND)
GRAPHENE_DEFINE_EXPR_BINARY_OP_AND_SYMBOL(Sub, SUBTRACT, -)
GRAPHENE_DEFINE_EXPR_BINARY_OP(VarianceToInvStdDev, VARIANCE_TO_INV_STD_DEV)

namespace ops {

template <DataTypeOrValue T1, DataTypeOrValue T2>
  requires AtLeastOneValue<T1, T2> && AtLeastOneTwoFloatTypeOrValue<T1, T2> &&
           CompatibleTypesForBinaryOp<popops::expr::BinaryOpType::ADD, T1, T2>
auto Add(const T1 &lhs, const T2 &rhs) {
  // When a Value<Type> is passed, the type is a **const reference** to the
  // Value. When a DataType is passed, the type is Value<Type>.
  decltype(detail::wrapInValue(lhs)) lhsValue = detail::wrapInValue(lhs);
  decltype(detail::wrapInValue(rhs)) rhsValue = detail::wrapInValue(rhs);

  std::optional<std::vector<size_t>> outShape =
      detail::broadcastShapes(lhsValue.shape(), rhsValue.shape());
  if (!outShape) throw std::runtime_error("Shapes are not compatible");

  using LeftType = typename unwrap_expression<T1>::type;
  using RightType = typename unwrap_expression<T2>::type;
  using ResultType = binary_op_return_type<popops::expr::BinaryOpType::ADD,
                                           LeftType, RightType>::type;

  auto &graph = Context::graph();

  // FIXME: Get the broadcasted tile mapping
  poplar::Graph::TileToTensorMapping outMapping = lhsValue.tileMapping();
  Value<ResultType> outValue(*outShape, outMapping);

  std::string codeletName = poputil::templateVertex(
      "graphene::ops::AddDoubleWord", lhsValue.tensor().elementType(),
      rhsValue.tensor().elementType(), outValue.tensor().elementType());

  DebugInfo di("DoubleWord");
  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < outMapping.size(); ++tile) {
    if (outMapping[tile].empty()) continue;
    // FIXME: Flatten to vector
    poplar::Tensor lhsTileTensor = lhsValue.tensorOnTile(tile);
    poplar::Tensor rhsTileTensor = rhsValue.tensorOnTile(tile);
    poplar::Tensor outTileTensor = outValue.tensorOnTile(tile);

    poplar::VertexRef vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["lhs"], lhsTileTensor);
    graph.connect(vertex["rhs"], rhsTileTensor);
    graph.connect(vertex["out"], outTileTensor);
    graph.setPerfEstimate(vertex, outTileTensor.numElements() * 15 + 100);
  }

  Context::program().add(poplar::program::Execute(cs, di));
  return outValue;
}

template <DataTypeOrValue T1, DataTypeOrValue T2>
  requires AtLeastOneValue<T1, T2> &&
           AtLeastOneDoublePrecisionTypeOrValue<T1, T2> &&
           CompatibleTypesForBinaryOp<popops::expr::BinaryOpType::ADD, T1, T2>
auto Add(const T1 &lhs, const T2 &rhs) {
  // When a Value<Type> is passed, the type is a **const reference** to the
  // Value. When a DataType is passed, the type is Value<Type>.
  decltype(detail::wrapInValue(lhs)) lhsValue = detail::wrapInValue(lhs);
  decltype(detail::wrapInValue(rhs)) rhsValue = detail::wrapInValue(rhs);

  std::optional<std::vector<size_t>> outShape =
      detail::broadcastShapes(lhsValue.shape(), rhsValue.shape());
  if (!outShape) throw std::runtime_error("Shapes are not compatible");

  using LeftType = typename unwrap_expression<T1>::type;
  using RightType = typename unwrap_expression<T2>::type;
  using ResultType = binary_op_return_type<popops::expr::BinaryOpType::ADD,
                                           LeftType, RightType>::type;

  auto &graph = Context::graph();

  // FIXME: Get the broadcasted tile mapping
  poplar::Graph::TileToTensorMapping outMapping = lhsValue.tileMapping();
  Value<ResultType> outValue(*outShape, outMapping);

  std::string codeletName = poputil::templateVertex(
      "graphene::ops::AddDoublePrecision", lhsValue.tensor().elementType(),
      rhsValue.tensor().elementType(), outValue.tensor().elementType());

  DebugInfo di("DoublePrecision");
  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < outMapping.size(); ++tile) {
    if (outMapping[tile].empty()) continue;
    // FIXME: Flatten to vector
    poplar::Tensor lhsTileTensor = lhsValue.tensorOnTile(tile);
    poplar::Tensor rhsTileTensor = rhsValue.tensorOnTile(tile);
    poplar::Tensor outTileTensor = outValue.tensorOnTile(tile);

    poplar::VertexRef vertex = graph.addVertex(cs, codeletName);
    graph.setTileMapping(vertex, tile);
    graph.connect(vertex["lhs"], lhsTileTensor);
    graph.connect(vertex["rhs"], rhsTileTensor);
    graph.connect(vertex["out"], outTileTensor);
    graph.setPerfEstimate(vertex, outTileTensor.numElements() * 15 + 100);
  }

  Context::program().add(poplar::program::Execute(cs, di));
  return outValue;
}
}  // namespace ops

}  // namespace graphene