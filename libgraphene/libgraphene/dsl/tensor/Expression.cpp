#include "libgraphene/dsl/tensor/Expression.hpp"

#include <spdlog/spdlog.h>

#include <iostream>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <pvti/pvti.hpp>

#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

namespace graphene {

template <DataType Type>
Expression<Type>::Expression(popops::expr::Any expr,
                             std::vector<poplar::Tensor> placeholders)
  requires PoplarNativeType<Type>
    : expr_(std::move(expr)), placeholders_(std::move(placeholders)) {}

template <DataType Type>
Expression<Type>::Expression(Type value)
  requires PoplarNativeType<Type>
    : expr_(popops::expr::Const(value)) {}

template <DataType Type>
Expression<Type>::Expression(poplar::Tensor tensor)
    : expr_(popops::expr::PlaceHolder(1)), placeholders_({tensor}) {
  if (tensor.elementType() != Traits<Type>::PoplarType) {
    throw std::runtime_error("Tensor type does not match the expression type");
  }
}

template <PoplarNativeType Type>
Tensor<Type> materializeExpression(const Expression<Type> &expr) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Expression");

  spdlog::trace("Materializing expression: {}",
                expr.expr().name(expr.placeholders()));

  popops::expr::Expr &e = const_cast<popops::expr::Any &>(expr.expr());
  di.add("expr", e.name(expr.placeholders()));

  poplar::Tensor tensor;
  if (auto *constExpr = e.getAs<popops::expr::Const>()) {
    // popops cannot infere the type of a constant, so we need to handle it
    // manually
    Type value = *reinterpret_cast<Type *>(constExpr->getData());
    return Tensor<Type>(value);
  } else {
    // Let popops do its magic to materialize the expression
    tensor = popops::map(Context::graph(), expr.expr(), expr.placeholders(),
                         Context::program(), di);
  }

  di.addOutput(tensor);
  return Tensor<Type>(tensor);
}

template <PoplarNativeType Type>
void materializeExpression(const Expression<Type> &expr, Tensor<Type> &value) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Expression");

  spdlog::trace("Materializing expression: {}",
                expr.expr().name(expr.placeholders()));

  popops::expr::Expr &e = const_cast<popops::expr::Any &>(expr.expr());
  di.add("expr", e.name(expr.placeholders()));

  if (auto *constExpr = e.getAs<popops::expr::Const>()) {
    // popops cannot infere the type of a constant, so we need to handle it
    // manually
    // TODO: Maybe we can use map() here by passing constTypes
    Type constVal = *reinterpret_cast<Type *>(constExpr->getData());
    auto &graph = Context::graph();

    // Create a tensor with the same rank as the value tensor, but with all
    // dimensions set to 1 and then broadcast it along all dimensions
    std::vector<size_t> shape(value.shape().size(), 1);
    poplar::Tensor constTensor = graph.addConstant<Type>(
        Traits<Type>::PoplarType, shape, {constVal}, di);
    graph.setTileMapping(constTensor, 0);

    for (size_t dim = 0; dim < value.shape().size(); ++dim) {
      size_t N = value.shape()[dim];
      if (N == 1) continue;
      constTensor = constTensor.broadcast(N, dim);
    }

    Context::program().add(
        poplar::program::Copy(constTensor, value.tensor(), false, di));
  } else {
    // popops::outputGeneratedCodelet(Context::graph().getTarget(), expr.expr(),
    //                                expr.placeholders(), {}, std::cout);
    // Let popops do its magic to materialize the expression
    popops::mapWithOutput(Context::graph(), expr.expr(), expr.placeholders(),
                          value.tensor(), Context::program(), di);
  }
}

template <DataType T>
std::vector<size_t> Expression<T>::shape() const {
  if (placeholders_.empty()) {
    return {1};
  }

  unsigned maxRank = 0;
  for (const auto &ph : placeholders_) {
    maxRank = std::max(maxRank, ph.rank());
  }

  std::vector<size_t> shape;
  for (unsigned i = 0; i < maxRank; ++i) {
    size_t dim = 1;
    for (const auto &ph : placeholders_) {
      if (ph.rank() > i) {
        dim = std::max(dim, ph.dim(i));
      }
    }
    shape.push_back(dim);
  }
  return shape;
}

template <DataType T>
size_t Expression<T>::rank() const {
  // Could be optimized
  return shape().size();
}

template <DataType T>
size_t Expression<T>::numElements() const {
  size_t num = 1;
  for (size_t dim : shape()) {
    num *= dim;
  }
  return num;
}

template <DataType T>
TypeRef Expression<T>::type() const {
  return getType<T>();
}

// Explicit instantiation
#define INSTANTIATE_NATIVE_TYPE(T)                                     \
  template class Expression<T>;                                        \
  template Tensor<T> materializeExpression(const Expression<T> &expr); \
  template void materializeExpression(const Expression<T> &expr,       \
                                      Tensor<T> &value);

#define INSTANTIATE_NON_NATIVE_TYPE(T) template class Expression<T>;

INSTANTIATE_NATIVE_TYPE(float)
INSTANTIATE_NON_NATIVE_TYPE(double)
INSTANTIATE_NON_NATIVE_TYPE(doubleword)
INSTANTIATE_NATIVE_TYPE(bool)
INSTANTIATE_NATIVE_TYPE(uint8_t)
INSTANTIATE_NATIVE_TYPE(int8_t)
INSTANTIATE_NATIVE_TYPE(uint16_t)
INSTANTIATE_NATIVE_TYPE(int16_t)
INSTANTIATE_NATIVE_TYPE(uint32_t)
INSTANTIATE_NATIVE_TYPE(int32_t)

#undef INSTANTIATE

}  // namespace graphene