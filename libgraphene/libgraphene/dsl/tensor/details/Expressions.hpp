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

#include <any>
#include <memory>
#include <poplar/Interval.hpp>
#include <poplar/Tensor.hpp>
#include <sstream>
#include <type_traits>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/common/details/Expressions.hpp"

namespace graphene::detail {

class UnaryExpr;
class BinaryExpr;
class DotProductExpr;
class CrossProductExpr;
class InputExpr;
class CastExpr;
class ConstExpr;
class PermuteExpr;
class BroadcastExpr;

struct ExpressionVisitor {
  virtual ~ExpressionVisitor() = default;
  virtual std::any visit(UnaryExpr& expr, std::any arg);
  virtual std::any visit(BinaryExpr& expr, std::any arg);
  virtual std::any visit(DotProductExpr& expr, std::any arg);
  virtual std::any visit(CrossProductExpr& expr, std::any arg);
  virtual std::any visit(InputExpr& expr, std::any arg);
  virtual std::any visit(CastExpr& expr, std::any arg);
  virtual std::any visit(ConstExpr& expr, std::any arg);
  virtual std::any visit(PermuteExpr& expr, std::any arg);
  virtual std::any visit(BroadcastExpr& expr, std::any arg);
};

class ExpressionBase {
  TypeRef type_;
  std::vector<std::shared_ptr<ExpressionBase>> children_;

 public:
  ExpressionBase() = delete;
  explicit ExpressionBase(TypeRef type);
  virtual ~ExpressionBase() = default;

  TypeRef type() const;

  const std::vector<std::shared_ptr<ExpressionBase>>& children() const;
  std::shared_ptr<ExpressionBase> child(size_t index) const;
  size_t numChildren() const;
  void replaceChild(size_t index, std::shared_ptr<ExpressionBase> child);

  virtual std::string getName() const = 0;
  virtual std::string getAsString() const = 0;
  virtual DistributedShape shape() const = 0;
  virtual TileMapping tileMapping() const = 0;
  virtual size_t hash() const = 0;

  virtual std::unique_ptr<ExpressionBase> clone() const = 0;

  virtual std::any accept(ExpressionVisitor& visitor, std::any arg = {}) = 0;

 protected:
  void addChild(std::shared_ptr<ExpressionBase> child);
};

class UnaryExpr : public ExpressionBase {
  UnaryOpType op_;

 public:
  UnaryExpr() = delete;
  UnaryExpr(UnaryOpType op, std::shared_ptr<ExpressionBase> arg);
  ~UnaryExpr() override = default;

  ExpressionBase& arg() const;
  UnaryOpType op() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class BinaryExpr : public ExpressionBase {
  BinaryOpType op_;

 public:
  BinaryExpr() = delete;
  BinaryExpr(BinaryOpType op, std::shared_ptr<ExpressionBase> lhs,
             std::shared_ptr<ExpressionBase> rhs);
  ~BinaryExpr() override = default;

  ExpressionBase& lhs() const;
  ExpressionBase& rhs() const;
  BinaryOpType op() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class DotProductExpr : public ExpressionBase {
 public:
  DotProductExpr() = delete;
  DotProductExpr(std::shared_ptr<ExpressionBase> lhs,
                 std::shared_ptr<ExpressionBase> rhs);
  ~DotProductExpr() override = default;

  ExpressionBase& lhs() const;
  ExpressionBase& rhs() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class CrossProductExpr : public ExpressionBase {
 public:
  CrossProductExpr() = delete;
  CrossProductExpr(std::shared_ptr<ExpressionBase> lhs,
                   std::shared_ptr<ExpressionBase> rhs);
  ~CrossProductExpr() override = default;

  ExpressionBase& lhs() const;
  ExpressionBase& rhs() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class InputExpr : public ExpressionBase {
  poplar::Tensor tensor_;
  poplar::Tensor dynamicSize_;

  // TODO: Add support for a dynamic first dimension

 public:
  InputExpr() = delete;
  InputExpr(poplar::Tensor tensor, TypeRef type = nullptr);

  const poplar::Tensor& tensor() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class CastExpr : public ExpressionBase {
 public:
  CastExpr() = delete;
  CastExpr(std::shared_ptr<ExpressionBase> arg, TypeRef type);
  ~CastExpr() override = default;

  ExpressionBase& arg() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class ConstExpr : public ExpressionBase {
  std::any value_;
  std::string str_;
  TensorShape shape_;

 public:
  ConstExpr() = delete;
  ConstExpr(std::any value, std::string str, TypeRef type, TensorShape shape);

  template <DataType T>
  explicit ConstExpr(T value);

  template <DataType T>
  explicit ConstExpr(const std::initializer_list<T>& values,
                     std::optional<TensorShape> shape = std::nullopt);

  ~ConstExpr() override = default;

  template <DataType T>
  T value() const;

  std::string valueAsString() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class PermuteExpr : public ExpressionBase {
  std::vector<size_t> permutation_;

 public:
  PermuteExpr() = delete;
  PermuteExpr(std::shared_ptr<ExpressionBase> arg,
              std::vector<size_t> permutation);
  ~PermuteExpr() override = default;

  ExpressionBase& arg() const;
  const std::vector<size_t>& permutation() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

class BroadcastExpr : public ExpressionBase {
  DistributedShape shape_;

 public:
  BroadcastExpr() = delete;
  BroadcastExpr(std::shared_ptr<ExpressionBase> arg, DistributedShape shape);
  ~BroadcastExpr() override = default;

  ExpressionBase& arg() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

}  // namespace graphene::detail