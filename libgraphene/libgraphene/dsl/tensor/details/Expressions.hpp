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
class InputExpr;
class CastExpr;
class ConstExpr;
class PermuteExpr;
class BroadcastExpr;

struct ExpressionVisitor {
  virtual std::any visit(UnaryExpr& expr, std::any arg);
  virtual std::any visit(BinaryExpr& expr, std::any arg);
  virtual std::any visit(InputExpr& expr, std::any arg);
  virtual std::any visit(CastExpr& expr, std::any arg);
  virtual std::any visit(ConstExpr& expr, std::any arg);
  virtual std::any visit(PermuteExpr& expr, std::any arg);
  virtual std::any visit(BroadcastExpr& expr, std::any arg);
};

class ExpressionBase {
  TypeRef type_;

 public:
  ExpressionBase() = delete;
  explicit ExpressionBase(TypeRef type);
  virtual ~ExpressionBase() = default;

  TypeRef type() const;

  virtual std::string getName() const = 0;
  virtual std::string getAsString() const = 0;
  virtual DistributedShape shape() const = 0;
  virtual TileMapping tileMapping() const = 0;
  virtual size_t hash() const = 0;

  virtual std::unique_ptr<ExpressionBase> clone() const = 0;

  virtual std::any accept(ExpressionVisitor& visitor, std::any arg = {}) = 0;
};

class UnaryExpr : public ExpressionBase {
  std::unique_ptr<ExpressionBase> arg_;
  UnaryOpType op_;

 public:
  UnaryExpr() = delete;
  UnaryExpr(UnaryOpType op, std::unique_ptr<ExpressionBase> arg);
  ~UnaryExpr() override = default;

  ExpressionBase* arg() const;
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
  std::unique_ptr<ExpressionBase> lhs_;
  std::unique_ptr<ExpressionBase> rhs_;
  BinaryOpType op_;

 public:
  BinaryExpr() = delete;
  BinaryExpr(BinaryOpType op, std::unique_ptr<ExpressionBase> lhs,
             std::unique_ptr<ExpressionBase> rhs);
  ~BinaryExpr() override = default;

  ExpressionBase* lhs() const;
  ExpressionBase* rhs() const;
  BinaryOpType op() const;

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
  std::unique_ptr<ExpressionBase> arg_;

 public:
  CastExpr() = delete;
  CastExpr(std::unique_ptr<ExpressionBase> arg, TypeRef type);
  ~CastExpr() override = default;

  ExpressionBase* arg() const;

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

 public:
  ConstExpr() = delete;
  ConstExpr(std::any value, std::string str, TypeRef type);

  /// Constructor for integer types
  template <DataType T>
    requires(std::is_integral<T>::value)
  explicit ConstExpr(T value)
      : ConstExpr(value, std::to_string(value), getType<T>()) {}

  /// Constructor for floating point types
  explicit ConstExpr(float value);
  explicit ConstExpr(double value);
  explicit ConstExpr(doubleword value);

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
  std::unique_ptr<ExpressionBase> arg_;
  std::vector<size_t> permutation_;

 public:
  PermuteExpr() = delete;
  PermuteExpr(std::unique_ptr<ExpressionBase> arg,
              std::vector<size_t> permutation);
  ~PermuteExpr() override = default;

  ExpressionBase* arg() const;
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
  std::unique_ptr<ExpressionBase> arg_;
  DistributedShape shape_;

 public:
  BroadcastExpr() = delete;
  BroadcastExpr(std::unique_ptr<ExpressionBase> arg, DistributedShape shape);
  ~BroadcastExpr() override = default;

  ExpressionBase* arg() const;

  std::string getName() const override;
  std::string getAsString() const override;
  DistributedShape shape() const override;
  TileMapping tileMapping() const override;
  size_t hash() const override;

  std::unique_ptr<ExpressionBase> clone() const override;

  std::any accept(ExpressionVisitor& visitor, std::any arg = {}) override;
};

}  // namespace graphene::detail