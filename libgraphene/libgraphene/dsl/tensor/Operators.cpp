#include "libgraphene/dsl/tensor/Operators.hpp"

#include <iostream>

namespace graphene::detail {

void shiftPlaceHolderIndices(const popops::expr::Expr *nodeConst,
                             unsigned int offset) {
  auto node = const_cast<popops::expr::Expr *>(nodeConst);
  if (auto ph = dynamic_cast<popops::expr::PlaceHolder *>(node)) {
    (*ph) = popops::expr::PlaceHolder(ph->getIndex() + offset);
  } else if (auto unary = dynamic_cast<popops::expr::UnaryOp *>(node)) {
    shiftPlaceHolderIndices(&unary->getArg(), offset);
  } else if (auto unary = dynamic_cast<popops::expr::Cast *>(node)) {
    shiftPlaceHolderIndices(&unary->getLHS(), offset);
  } else if (auto binary = dynamic_cast<popops::expr::BinaryOp *>(node)) {
    shiftPlaceHolderIndices(&binary->getLHS(), offset);
    shiftPlaceHolderIndices(&binary->getRHS(), offset);
  } else if (auto ternary = dynamic_cast<popops::expr::TernaryOp *>(node)) {
    shiftPlaceHolderIndices(&ternary->getArg0(), offset);
    shiftPlaceHolderIndices(&ternary->getArg1(), offset);
    shiftPlaceHolderIndices(&ternary->getArg2(), offset);
  }
}

std::optional<std::vector<size_t>> broadcastShapes(std::vector<size_t> shape1,
                                                   std::vector<size_t> shape2) {
  // Pad the shorter shape with 1s at the beginning
  std::vector<int> padded_shape1(std::max(shape1.size(), shape2.size()), 1);
  std::vector<int> padded_shape2(std::max(shape1.size(), shape2.size()), 1);

  std::copy(shape1.rbegin(), shape1.rend(), padded_shape1.rbegin());
  std::copy(shape2.rbegin(), shape2.rend(), padded_shape2.rbegin());

  // Calculate the broadcasted shape
  std::vector<size_t> broadcasted_shape(padded_shape1.size());

  for (size_t i = 0; i < padded_shape1.size(); ++i) {
    if (padded_shape1[i] == padded_shape2[i]) {
      broadcasted_shape[i] = padded_shape1[i];
    } else if (padded_shape1[i] == 1) {
      broadcasted_shape[i] = padded_shape2[i];
    } else if (padded_shape2[i] == 1) {
      broadcasted_shape[i] = padded_shape1[i];
    } else {
      // Shapes are not compatible for broadcasting
      return std::nullopt;
    }
  }

  return broadcasted_shape;
}

#undef INSTANTIATE
}  // namespace graphene::detail
