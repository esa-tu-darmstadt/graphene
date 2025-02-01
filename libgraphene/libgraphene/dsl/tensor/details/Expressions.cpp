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
#include <string>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Hash.hpp"
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

//--------------------------------------------------------------------------
// UnaryExpr implementation
//--------------------------------------------------------------------------
UnaryExpr::UnaryExpr(UnaryOpType op, std::unique_ptr<ExpressionBase> arg)
    : ExpressionBase(inferType(op, arg->type())),
      arg_(std::move(arg)),
      op_(op) {}

ExpressionBase* UnaryExpr::arg() const { return arg_.get(); }

UnaryOpType UnaryExpr::op() const { return op_; }

std::string UnaryExpr::getName() const { return std::string(to_string(op_)); }

std::string UnaryExpr::getAsString() const {
  return getName() + "(" + arg_->getAsString() + ")";
}

DistributedShape UnaryExpr::shape() const { return arg_->shape(); }

TileMapping UnaryExpr::tileMapping() const { return arg_->tileMapping(); }

std::unique_ptr<ExpressionBase> UnaryExpr::clone() const {
  return std::make_unique<UnaryExpr>(op_, arg_->clone());
}

std::any UnaryExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t UnaryExpr::hash() const {
  return graphene::hash("unary", type(), op_, arg_->hash());
}

//--------------------------------------------------------------------------
// BinaryExpr implementation
//--------------------------------------------------------------------------
BinaryExpr::BinaryExpr(BinaryOpType op, std::unique_ptr<ExpressionBase> lhs,
                       std::unique_ptr<ExpressionBase> rhs)
    : ExpressionBase(inferType(op, lhs->type(), rhs->type())),
      lhs_(std::move(lhs)),
      rhs_(std::move(rhs)),
      op_(op) {
  // Ensure that the shapes are compatible for broadcasting
  (void)shape();
}

ExpressionBase* BinaryExpr::lhs() const { return lhs_.get(); }

ExpressionBase* BinaryExpr::rhs() const { return rhs_.get(); }

BinaryOpType BinaryExpr::op() const { return op_; }

std::string BinaryExpr::getName() const { return std::string(to_string(op_)); }

std::string BinaryExpr::getAsString() const {
  return getName() + "(" + lhs_->getAsString() + ", " + rhs_->getAsString() +
         ")";
}

DistributedShape BinaryExpr::shape() const {
  auto maybeShape = DistributedShape::broadcast(lhs_->shape(), rhs_->shape());
  if (!maybeShape) {
    throw std::runtime_error("Shapes are not compatible for broadcasting");
  }

  return maybeShape.value();
}

TileMapping BinaryExpr::tileMapping() const {
  return TileMapping::linearMappingWithShape(shape());
}

std::unique_ptr<ExpressionBase> BinaryExpr::clone() const {
  return std::make_unique<BinaryExpr>(op_, lhs_->clone(), rhs_->clone());
}

std::any BinaryExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t BinaryExpr::hash() const {
  return graphene::hash("binary", type(), op_, lhs_->hash(), rhs_->hash());
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
CastExpr::CastExpr(std::unique_ptr<ExpressionBase> arg, TypeRef type)
    : ExpressionBase(type), arg_(std::move(arg)) {}

ExpressionBase* CastExpr::arg() const { return arg_.get(); }

std::string CastExpr::getName() const { return "cast"; }

std::string CastExpr::getAsString() const {
  return "cast<" + type()->str() + ">(" + arg_->getAsString() + ")";
}

DistributedShape CastExpr::shape() const { return arg_->shape(); }
TileMapping CastExpr::tileMapping() const { return arg_->tileMapping(); }

std::unique_ptr<ExpressionBase> CastExpr::clone() const {
  return std::make_unique<CastExpr>(arg_->clone(), type());
}

std::any CastExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t CastExpr::hash() const {
  return graphene::hash("cast", type(), arg_->hash());
}

//--------------------------------------------------------------------------
// ConstExpr implementation
//--------------------------------------------------------------------------
ConstExpr::ConstExpr(std::any value, std::string str, TypeRef type)
    : ExpressionBase(type), value_(value), str_(str) {}

std::string ConstExpr::valueAsString() const { return str_; }

std::string ConstExpr::getName() const { return "const"; }

std::string ConstExpr::getAsString() const {
  return "const<" + type()->str() + ">(" + str_ + ")";
}

DistributedShape ConstExpr::shape() const { return DistributedShape::scalar(); }

TileMapping ConstExpr::tileMapping() const { return {}; }

std::unique_ptr<ExpressionBase> ConstExpr::clone() const {
  return std::make_unique<ConstExpr>(value_, str_, type());
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

}  // namespace

/// Constructor for floating point types
ConstExpr::ConstExpr(float value)
    : ConstExpr(value, floatToString(value), Type::FLOAT32) {}
ConstExpr::ConstExpr(double value)
    : ConstExpr(value, floatToString(value), Type::FLOAT64) {}
ConstExpr::ConstExpr(doubleword value)
    : ConstExpr(value, floatToString(value), Type::TWOFLOAT32) {}

size_t ConstExpr::hash() const { return graphene::hash("const", type(), str_); }

//--------------------------------------------------------------------------
// PermuteExpr implementation
//--------------------------------------------------------------------------
PermuteExpr::PermuteExpr(std::unique_ptr<ExpressionBase> arg,
                         std::vector<size_t> permutation)
    : ExpressionBase(arg->type()),
      arg_(std::move(arg)),
      permutation_(std::move(permutation)) {
  if (arg_->shape().rank() != permutation_.size()) {
    throw std::runtime_error(
        "Permutation size must match the rank of the input");
  }
  if (permutation_[0] != 0) {
    throw std::runtime_error("Permutation must keep the first dimension");
  }
}

ExpressionBase* PermuteExpr::arg() const { return arg_.get(); }

const std::vector<size_t>& PermuteExpr::permutation() const {
  return permutation_;
}

std::string PermuteExpr::getName() const { return "permute"; }

std::string PermuteExpr::getAsString() const {
  std::stringstream ss;
  ss << "permute(" << arg_->getAsString() << ", [";
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
  DistributedShape shape = arg_->shape();
  DistributedShape permutedShape = shape;
  // The first dimension is always the same
  for (size_t i = 1; i < shape.rank(); ++i) {
    permutedShape.globalShape()[i] = shape.globalShape()[permutation_[i]];
  }
  return permutedShape;
}

TileMapping PermuteExpr::tileMapping() const { return arg_->tileMapping(); }

std::unique_ptr<ExpressionBase> PermuteExpr::clone() const {
  return std::make_unique<PermuteExpr>(arg_->clone(), permutation_);
}

std::any PermuteExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t PermuteExpr::hash() const {
  return graphene::hash("permute", permutation_, arg_->hash());
}

// -----------------------------------------------------------------------
// BroadcastExpr implementation
// -----------------------------------------------------------------------
BroadcastExpr::BroadcastExpr(std::unique_ptr<ExpressionBase> arg,
                             DistributedShape shape)
    : ExpressionBase(arg->type()), arg_(std::move(arg)), shape_(shape) {
  if (arg_->shape().rank() > shape_.rank()) {
    throw std::runtime_error("Broadcasting to a smaller rank is not allowed");
  }
  if (!DistributedShape::broadcast(arg_->shape(), shape_)) {
    throw std::runtime_error("Shapes are not compatible for broadcasting");
  }
}

ExpressionBase* BroadcastExpr::arg() const { return arg_.get(); }

std::string BroadcastExpr::getName() const { return "broadcast"; }

std::string BroadcastExpr::getAsString() const {
  std::stringstream ss;
  ss << "broadcast(" << arg_->getAsString() << ", [";
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
  return std::make_unique<BroadcastExpr>(arg_->clone(), shape_);
}

std::any BroadcastExpr::accept(ExpressionVisitor& visitor, std::any arg) {
  return visitor.visit(*this, arg);
}

size_t BroadcastExpr::hash() const {
  return graphene::hash("broadcast", shape_, arg_->hash());
}

//--------------------------------------------------------------------------
// ExpressionVisitor implementation
//--------------------------------------------------------------------------
std::any ExpressionVisitor::visit(UnaryExpr& expr, std::any arg) {
  expr.arg()->accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(BinaryExpr& expr, std::any arg) {
  expr.lhs()->accept(*this, arg);
  expr.rhs()->accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(InputExpr& expr, std::any arg) { return {}; }
std::any ExpressionVisitor::visit(CastExpr& expr, std::any arg) {
  expr.arg()->accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(ConstExpr& expr, std::any arg) { return {}; }
std::any ExpressionVisitor::visit(PermuteExpr& expr, std::any arg) {
  expr.arg()->accept(*this, arg);
  return {};
}
std::any ExpressionVisitor::visit(BroadcastExpr& expr, std::any arg) {
  expr.arg()->accept(*this, arg);
  return {};
}
}  // namespace graphene::detail