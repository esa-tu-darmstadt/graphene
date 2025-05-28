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

#include "libgraphene/dsl/tensor/details/Expressions.hpp"

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Hash.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/dsl/common/details/Expressions.hpp"
#include "libgraphene/dsl/tensor/Expression.hpp"
#include "libgraphene/util/Context.hpp"

namespace graphene::detail {

//--------------------------------------------------------------------------
// ExpressionBase implementation
//--------------------------------------------------------------------------
ExpressionBase::ExpressionBase(TypeRef type) : type_(type) {}

TypeRef ExpressionBase::type() const { return type_; }

const std::vector<std::shared_ptr<ExpressionBase>>& ExpressionBase::children()
    const {
  return children_;
}

std::shared_ptr<ExpressionBase> ExpressionBase::child(size_t index) const {
  if (index >= children_.size()) {
    throw std::out_of_range("Index out of range");
  }
  return children_[index];
}

size_t ExpressionBase::numChildren() const { return children_.size(); }

void ExpressionBase::addChild(std::shared_ptr<ExpressionBase> child) {
  children_.push_back(std::move(child));
}

void ExpressionBase::replaceChild(size_t index,
                                  std::shared_ptr<ExpressionBase> child) {
  if (index >= children_.size()) {
    children_.resize(index + 1);
  }
  children_[index] = std::move(child);
}

//--------------------------------------------------------------------------
// UnaryExpr implementation
//--------------------------------------------------------------------------
UnaryExpr::UnaryExpr(UnaryOpType op, std::shared_ptr<ExpressionBase> arg)
    : ExpressionBase(inferType(op, arg->type())), op_(op) {
  addChild(std::move(arg));
}

ExpressionBase& UnaryExpr::arg() const { return *child(0); }

UnaryOpType UnaryExpr::op() const { return op_; }

std::string UnaryExpr::getName() const { return std::string(to_string(op_)); }

std::string UnaryExpr::getAsString() const {
  return getName() + "(" + arg().getAsString() + ")";
}

DistributedShape UnaryExpr::shape() const { return arg().shape(); }

TileMapping UnaryExpr::tileMapping() const { return arg().tileMapping(); }

std::unique_ptr<ExpressionBase> UnaryExpr::clone() const {
  return std::make_unique<UnaryExpr>(
      op_, std::shared_ptr<ExpressionBase>(arg().clone()));
}

std::any UnaryExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t UnaryExpr::hash() const {
  return graphene::hash("unary", type(), op_, arg().hash());
}

//--------------------------------------------------------------------------
// BinaryExpr implementation
//--------------------------------------------------------------------------
BinaryExpr::BinaryExpr(BinaryOpType op, std::shared_ptr<ExpressionBase> lhs,
                       std::shared_ptr<ExpressionBase> rhs)
    : ExpressionBase(inferType(op, lhs->type(), rhs->type())), op_(op) {
  addChild(std::move(lhs));
  addChild(std::move(rhs));
  // Ensure that the shapes are compatible for broadcasting
  (void)shape();
}

ExpressionBase& BinaryExpr::lhs() const { return *child(0); }

ExpressionBase& BinaryExpr::rhs() const { return *child(1); }

BinaryOpType BinaryExpr::op() const { return op_; }

std::string BinaryExpr::getName() const { return std::string(to_string(op_)); }

std::string BinaryExpr::getAsString() const {
  return getName() + "(" + lhs().getAsString() + ", " + rhs().getAsString() +
         ")";
}

DistributedShape BinaryExpr::shape() const {
  auto maybeShape = DistributedShape::broadcast(lhs().shape(), rhs().shape());
  if (!maybeShape) {
    throw std::runtime_error("Shapes are not compatible for broadcasting");
  }

  return maybeShape.value();
}

TileMapping BinaryExpr::tileMapping() const {
  return TileMapping::linearMappingWithShape(shape());
}

std::unique_ptr<ExpressionBase> BinaryExpr::clone() const {
  return std::make_unique<BinaryExpr>(
      op_, std::shared_ptr<ExpressionBase>(lhs().clone()),
      std::shared_ptr<ExpressionBase>(rhs().clone()));
}

std::any BinaryExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t BinaryExpr::hash() const {
  return graphene::hash("binary", type(), op_, lhs().hash(), rhs().hash());
}

//--------------------------------------------------------------------------
// DotProductExpr implementation
//--------------------------------------------------------------------------
DotProductExpr::DotProductExpr(std::shared_ptr<ExpressionBase> lhs,
                               std::shared_ptr<ExpressionBase> rhs)
    : ExpressionBase(
          inferType(BinaryOpType::DOT_PRODUCT, lhs->type(), rhs->type())) {
  addChild(std::move(lhs));
  addChild(std::move(rhs));
  // Validate that both operands are vector fields (rank 2, second dim > 1)
  auto lhsShape = this->lhs().shape();
  auto rhsShape = this->rhs().shape();

  if (lhsShape.rank() != 2 || rhsShape.rank() != 2) {
    throw std::runtime_error(
        "Dot product requires rank 2 tensors (vector fields)");
  }

  if (lhsShape.globalShape()[1] <= 1 || rhsShape.globalShape()[1] <= 1) {
    throw std::runtime_error(
        "Dot product requires second dimension > 1 for vector fields");
  }

  if (lhsShape.globalShape()[1] != rhsShape.globalShape()[1]) {
    throw std::runtime_error("Dot product requires vectors of the same length");
  }

  // Check that shapes are compatible for broadcasting
  auto maybeShape = DistributedShape::broadcast(lhsShape, rhsShape);
  if (!maybeShape) {
    throw std::runtime_error(
        "Shapes are not compatible for broadcasting in dot product");
  }
}

ExpressionBase& DotProductExpr::lhs() const { return *child(0); }

ExpressionBase& DotProductExpr::rhs() const { return *child(1); }

std::string DotProductExpr::getName() const { return "dot_product"; }

std::string DotProductExpr::getAsString() const {
  return getName() + "(" + lhs().getAsString() + ", " + rhs().getAsString() +
         ")";
}

DistributedShape DotProductExpr::shape() const {
  // Dot product: broadcast first, then reduce second dimension
  auto lhsShape = lhs().shape();
  auto rhsShape = rhs().shape();

  auto broadcastShape = DistributedShape::broadcast(lhsShape, rhsShape).value();

  // Reduce the vector dimension (second dim becomes 1)
  DistributedShape outputShape = broadcastShape;
  outputShape.globalShape()[1] = 1;

  return outputShape;
}

TileMapping DotProductExpr::tileMapping() const {
  return TileMapping::linearMappingWithShape(shape());
}

std::unique_ptr<ExpressionBase> DotProductExpr::clone() const {
  return std::make_unique<DotProductExpr>(
      std::shared_ptr<ExpressionBase>(lhs().clone()),
      std::shared_ptr<ExpressionBase>(rhs().clone()));
}

std::any DotProductExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t DotProductExpr::hash() const {
  return graphene::hash("dot_product", type(), lhs().hash(), rhs().hash());
}

//--------------------------------------------------------------------------
// CrossProductExpr implementation
//--------------------------------------------------------------------------
CrossProductExpr::CrossProductExpr(std::shared_ptr<ExpressionBase> lhs,
                                   std::shared_ptr<ExpressionBase> rhs)
    : ExpressionBase(
          inferType(BinaryOpType::CROSS_PRODUCT, lhs->type(), rhs->type())) {
  addChild(std::move(lhs));
  addChild(std::move(rhs));
  // Validate that both operands are 3D vector fields
  auto lhsShape = this->lhs().shape();
  auto rhsShape = this->rhs().shape();

  if (lhsShape.rank() != 2 || rhsShape.rank() != 2) {
    throw std::runtime_error(
        "Cross product requires rank 2 tensors (vector fields)");
  }

  if (lhsShape.globalShape()[1] != 3 || rhsShape.globalShape()[1] != 3) {
    throw std::runtime_error(
        "Cross product only works with 3D vectors (second dimension must be "
        "3)");
  }

  // Check that shapes are compatible for broadcasting
  auto maybeShape = DistributedShape::broadcast(lhsShape, rhsShape);
  if (!maybeShape) {
    throw std::runtime_error(
        "Shapes are not compatible for broadcasting in cross product");
  }
}

ExpressionBase& CrossProductExpr::lhs() const { return *child(0); }

ExpressionBase& CrossProductExpr::rhs() const { return *child(1); }

std::string CrossProductExpr::getName() const { return "cross_product"; }

std::string CrossProductExpr::getAsString() const {
  return getName() + "(" + lhs().getAsString() + ", " + rhs().getAsString() +
         ")";
}

DistributedShape CrossProductExpr::shape() const {
  // Cross product: broadcast the shapes, keep vector dimension as 3
  auto lhsShape = lhs().shape();
  auto rhsShape = rhs().shape();

  return DistributedShape::broadcast(lhsShape, rhsShape).value();
}

TileMapping CrossProductExpr::tileMapping() const {
  return TileMapping::linearMappingWithShape(shape());
}

std::unique_ptr<ExpressionBase> CrossProductExpr::clone() const {
  return std::make_unique<CrossProductExpr>(
      std::shared_ptr<ExpressionBase>(lhs().clone()),
      std::shared_ptr<ExpressionBase>(rhs().clone()));
}

std::any CrossProductExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t CrossProductExpr::hash() const {
  return graphene::hash("cross_product", type(), lhs().hash(), rhs().hash());
}

//--------------------------------------------------------------------------
// InputExpr implementation
//--------------------------------------------------------------------------
InputExpr::InputExpr(poplar::Tensor tensor, TypeRef type)
    : ExpressionBase(type ? type : getType(tensor.elementType())),
      tensor_(tensor) {
  assert(tensor.valid() && "tensor must be valid");
}

const poplar::Tensor& InputExpr::tensor() const { return tensor_; }

std::string InputExpr::getName() const { return "input"; }

std::string InputExpr::getAsString() const {
  std::stringstream ss;
  ss << "input<" << shape().globalShape().str() << ", " << type()->str() << ">";
  return ss.str();
}

DistributedShape InputExpr::shape() const {
  return DistributedShape::fromPoplar(tensor_.shape(),
                                      Context::graph().getTileMapping(tensor_));
}
TileMapping InputExpr::tileMapping() const {
  return TileMapping::fromPoplar(Context::graph().getTileMapping(tensor_));
}

std::unique_ptr<ExpressionBase> InputExpr::clone() const {
  return std::make_unique<InputExpr>(tensor_, type());
}

std::any InputExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t InputExpr::hash() const {
  return graphene::hash("input", type(), shape());
}

//--------------------------------------------------------------------------
// CastExpr implementation
//--------------------------------------------------------------------------
CastExpr::CastExpr(std::shared_ptr<ExpressionBase> arg, TypeRef type)
    : ExpressionBase(type) {
  addChild(std::move(arg));
}

ExpressionBase& CastExpr::arg() const { return *child(0); }

std::string CastExpr::getName() const { return "cast"; }

std::string CastExpr::getAsString() const {
  return "cast<" + type()->str() + ">(" + arg().getAsString() + ")";
}

DistributedShape CastExpr::shape() const { return arg().shape(); }
TileMapping CastExpr::tileMapping() const { return arg().tileMapping(); }

std::unique_ptr<ExpressionBase> CastExpr::clone() const {
  return std::make_unique<CastExpr>(
      std::shared_ptr<ExpressionBase>(arg().clone()), type());
}

std::any CastExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t CastExpr::hash() const {
  return graphene::hash("cast", type(), arg().hash());
}

//--------------------------------------------------------------------------
// ConstExpr implementation
//--------------------------------------------------------------------------
ConstExpr::ConstExpr(std::any value, std::string str, TypeRef type,
                     TensorShape shape)
    : ExpressionBase(type), value_(value), str_(str), shape_(shape) {}

std::string ConstExpr::valueAsString() const {
  return "std::array<" + type()->str() + ", " +
         std::to_string(shape_.numElements()) + ">{" + str_ + "}";
}

std::string ConstExpr::getName() const { return "const"; }

std::string ConstExpr::getAsString() const {
  return "const<" + type()->str() + ">(" + str_ + ")";
}

DistributedShape ConstExpr::shape() const {
  return DistributedShape::onSingleTile(shape_);
}

TileMapping ConstExpr::tileMapping() const { return {}; }

std::unique_ptr<ExpressionBase> ConstExpr::clone() const {
  return std::make_unique<ConstExpr>(value_, str_, type(), shape_);
}

std::any ConstExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

namespace {
std::string floatToString(float value) {
  std::stringstream ss;
  ss << std::setprecision(8) << std::scientific << value << "f";
  return ss.str();
}
std::string floatToString(double value) {
  std::stringstream ss;
  ss << std::setprecision(16) << std::scientific << value;
  return ss.str();
}
std::string floatToString(doubleword value) {
  std::stringstream ss;
  ss << std::setprecision(8) << "::twofloat::two<float>{" << value.h << ", "
     << value.l << "}";
  return ss.str();
}

template <typename T>
std::string valueToString(const T& value) {
  if constexpr (std::is_same_v<T, float>) {
    return floatToString(value);
  } else if constexpr (std::is_same_v<T, double>) {
    return floatToString(value);
  } else if constexpr (std::is_same_v<T, doubleword>) {
    return floatToString(value);
  } else {
    return std::to_string(value);
  }
}

template <typename T>
std::string valuesToString(const std::initializer_list<T>& values) {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < values.size(); ++i) {
    ss << valueToString(values.begin()[i]);
    if (i < values.size() - 1) {
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

}  // namespace

template <DataType T>
ConstExpr::ConstExpr(T value)
    : ExpressionBase(getType<T>()),
      value_(value),
      str_(valueToString(value)),
      shape_({1}) {}

template <DataType T>
ConstExpr::ConstExpr(const std::initializer_list<T>& values,
                     std::optional<TensorShape> shape)
    : ExpressionBase(getType<T>()),
      value_(std::vector<T>(values)),
      str_(valuesToString(values)),
      shape_(shape.value_or(TensorShape{values.size()})) {}

size_t ConstExpr::hash() const { return graphene::hash("const", type(), str_); }

// Explicit instantiation
#define INSTANTIATE(T)                                                  \
  template ConstExpr::ConstExpr(T value);                               \
  template ConstExpr::ConstExpr(const std::initializer_list<T>& values, \
                                std::optional<TensorShape> shape);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(doubleword)
INSTANTIATE(bool)
INSTANTIATE(uint8_t)
INSTANTIATE(int8_t)
INSTANTIATE(uint16_t)
INSTANTIATE(int16_t)
INSTANTIATE(uint32_t)
INSTANTIATE(int32_t)
INSTANTIATE(uint64_t)
INSTANTIATE(int64_t)
#undef INSTANTIATE

//--------------------------------------------------------------------------
// PermuteExpr implementation
//--------------------------------------------------------------------------
PermuteExpr::PermuteExpr(std::shared_ptr<ExpressionBase> arg,
                         std::vector<size_t> permutation)
    : ExpressionBase(arg->type()), permutation_(std::move(permutation)) {
  addChild(std::move(arg));
  if (this->arg().shape().rank() != permutation_.size()) {
    throw std::runtime_error(
        "Permutation size must match the rank of the input");
  }
  if (permutation_[0] != 0) {
    throw std::runtime_error("Permutation must keep the first dimension");
  }
}

ExpressionBase& PermuteExpr::arg() const { return *child(0); }

const std::vector<size_t>& PermuteExpr::permutation() const {
  return permutation_;
}

std::string PermuteExpr::getName() const { return "permute"; }

std::string PermuteExpr::getAsString() const {
  std::stringstream ss;
  ss << "permute(" << arg().getAsString() << ", [";
  for (size_t i = 0; i < permutation_.size(); ++i) {
    ss << permutation_[i];
    if (i < permutation_.size() - 1) {
      ss << ", ";
    }
  }
  ss << "])";
  return ss.str();
}

DistributedShape PermuteExpr::shape() const {
  DistributedShape shape = arg().shape();
  DistributedShape permutedShape = shape;
  // The first dimension is always the same
  for (size_t i = 1; i < shape.rank(); ++i) {
    permutedShape.globalShape()[i] = shape.globalShape()[permutation_[i]];
  }
  return permutedShape;
}

TileMapping PermuteExpr::tileMapping() const { return arg().tileMapping(); }

std::unique_ptr<ExpressionBase> PermuteExpr::clone() const {
  return std::make_unique<PermuteExpr>(
      std::shared_ptr<ExpressionBase>(arg().clone()), permutation_);
}

std::any PermuteExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t PermuteExpr::hash() const {
  return graphene::hash("permute", permutation_, arg().hash());
}

// -----------------------------------------------------------------------
// BroadcastExpr implementation
// -----------------------------------------------------------------------
BroadcastExpr::BroadcastExpr(std::shared_ptr<ExpressionBase> arg,
                             DistributedShape shape)
    : ExpressionBase(arg->type()), shape_(shape) {
  addChild(std::move(arg));
  if (this->arg().shape().rank() > shape_.rank()) {
    throw std::runtime_error("Broadcasting to a smaller rank is not allowed");
  }
  if (!DistributedShape::broadcast(this->arg().shape(), shape_)) {
    throw std::runtime_error("Shapes are not compatible for broadcasting");
  }
}

ExpressionBase& BroadcastExpr::arg() const { return *child(0); }

std::string BroadcastExpr::getName() const { return "broadcast"; }

std::string BroadcastExpr::getAsString() const {
  std::stringstream ss;
  ss << "broadcast(" << arg().getAsString() << ", [";
  for (size_t i = 0; i < shape_.rank(); ++i) {
    ss << shape_[i];
    if (i < shape_.rank() - 1) {
      ss << "x";
    }
  }
  ss << "])";
  return ss.str();
}

DistributedShape BroadcastExpr::shape() const { return shape_; }

TileMapping BroadcastExpr::tileMapping() const {
  return TileMapping::linearMappingWithShape(shape());
}

std::unique_ptr<ExpressionBase> BroadcastExpr::clone() const {
  return std::make_unique<BroadcastExpr>(
      std::shared_ptr<ExpressionBase>(arg().clone()), shape_);
}

std::any BroadcastExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t BroadcastExpr::hash() const {
  return graphene::hash("broadcast", shape_, arg().hash());
}

//--------------------------------------------------------------------------
// ExpressionVisitor implementation
//--------------------------------------------------------------------------
std::any ExpressionVisitor::visit(UnaryExpr& expr, std::any arg) {
  expr.arg().accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(BinaryExpr& expr, std::any arg) {
  expr.lhs().accept(*this, arg);
  expr.rhs().accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(DotProductExpr& expr, std::any arg) {
  expr.lhs().accept(*this, arg);
  expr.rhs().accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(CrossProductExpr& expr, std::any arg) {
  expr.lhs().accept(*this, arg);
  expr.rhs().accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(InputExpr& expr, std::any arg) { return {}; }
std::any ExpressionVisitor::visit(CastExpr& expr, std::any arg) {
  expr.arg().accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(ConstExpr& expr, std::any arg) { return {}; }
std::any ExpressionVisitor::visit(PermuteExpr& expr, std::any arg) {
  expr.arg().accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(BroadcastExpr& expr, std::any arg) {
  expr.arg().accept(*this, arg);
  return {};
}
}  // namespace graphene::detail