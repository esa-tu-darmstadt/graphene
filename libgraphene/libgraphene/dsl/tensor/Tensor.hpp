#pragma once

#include <any>
#include <initializer_list>
#include <poplar/Graph.hpp>
#include <typeindex>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/Expression.hpp"
#include "libgraphene/dsl/tensor/details/Expressions.hpp"
#include "libgraphene/matrix/Norm.hpp"

namespace graphene {

class RemoteTensor;
class HostTensor;

/**
 * @brief Represents a mutable tensor, which can be assigned to.
 * This class has value semantics. When assigned or passed-by-value, a copy
 * program for the underlying tensor is generated.
 *
 * @tparam Type The data type of the tensor.
 */
class Tensor : public Expression {
  std::map<std::type_index, std::any> metadata_;

  /// Constructs an uninitialized tensor. See `Tensor::uninitialized`.
  explicit Tensor(TypeRef type, DistributedShape shape, TileMapping tileMapping,
                  std::string name);

  /// Constructs a tensor with an initial value. See `Tensor::withInitialValue`.
  template <DataType Type>
  explicit Tensor(std::initializer_list<Type> values,
                  std::optional<DistributedShape> shape = {},
                  TileMapping tileMapping = {}, std::string name = "");

  /// Constructs a tensor from a poplar tensor. See `Tensor::fromPoplar`.
  explicit Tensor(const poplar::Tensor tensor, TypeRef type);

 public:
  /**
   * @brief Constructs an unitialized mutable tensor.
   *
   * @param shape The shape of the tensor.
   * @param tileMapping The tile mapping of the tensor.
   */
  static Tensor uninitialized(
      TypeRef type, DistributedShape shape = DistributedShape::scalar(),
      TileMapping tileMapping = {}, std::string name = "") {
    return Tensor(type, shape, tileMapping, name);
  }

  /**
   * @brief Constructs a mutable tensor with an initial, scalar value.
   *
   * @param value The initial value.
   * @param name The name of the tensor.
   */
  template <DataType Type>
  static Tensor withInitialValue(Type value, std::string name = "") {
    return Tensor({value}, {}, {}, name);
  }

  /**
   * @brief Constructs a mutable tensor with an initial value.
   *
   * @param values The initial values.
   * @param shape The shape of the tensor.
   * @param tileMapping The tile mapping of the tensor.
   * @param name The name of the tensor.
   */
  template <DataType Type>
  static Tensor withInitialValue(std::initializer_list<Type> values,
                                 std::optional<DistributedShape> shape = {},
                                 TileMapping tileMapping = {},
                                 std::string name = "") {
    return Tensor(values, shape, tileMapping, name);
  }

  /**
   * @brief Constructs a mutable tensor from a poplar tensor.
   *
   * @param tensor The poplar tensor.
   * @param type The type of the tensor.
   */
  static Tensor fromPoplar(poplar::Tensor tensor, TypeRef type) {
    return Tensor(tensor, type);
  }

  /**
   * @brief Constructs a mutable value with the initial value of the expression.
   *
   * @param expr The expression to initialize from.
   */
  Tensor(Expression expr);

  /**
   * @brief Constructs a mutable with the given value as its initial value.
   *
   * @param value The value to copy.
   */
  Tensor(const Tensor &value);

  /**
   * @brief Constructs a mutable value by moving another value.
   *
   * @param value The value to move.
   */
  Tensor(Tensor &&value) : Expression(std::move(value)) {
    spdlog::trace("Move constructing tensor");
  }

  /** Assignment operators. Generates copy programs. */
  Tensor &operator=(const Tensor &value);
  Tensor &operator=(const Expression &expr);

  /**
   * @brief Returns a tensor with the same storage as this tensor. The resulting
   * tensor will always have the same content as this tensor.
   */
  Tensor same() const;

  /**
   * @brief Get the underlying poplar tensor.
   *
   * @param flattenIfScalar If true, flatten the tensor if it's scalar.
   * @return poplar::Tensor The underlying tensor.
   */
  poplar::Tensor tensor(bool flattenIfScalar = false) const;

  /**
   * @brief Get the slice of the underlying poplar tensor mapped to the given
   * tile.
   *
   * @param tile The tile to get the tensor on.
   * @param flattenIfScalar If true, flatten the tensor if it's scalar.
   * @return poplar::Tensor The tensor on the tile.
   */
  poplar::Tensor tensorOnTile(size_t tile, bool flattenIfScalar = false) const;

  /**
   * @brief Print the tensor to the console.
   *
   * @param name Optional name for the tensor.
   * @param fmt Format options for printing.
   * @param stream Optional stream to print to. If not given, prints to
   * std::cout. If given, must be still alive when the tensor is print during
   * execution.
   */
  void print(std::string name = "", poplar::PrintTensorFmt fmt = {},
             std::ostream &stream = std::cout) const;

  /**
   * @brief Rearranges this tensor across tiles. The first dimension is
   * distributed according to the given shape. If no tile mapping is given, the
   * tensor is mapped linearly across the tiles specified in the shape (see \ref
   * TileMapping::linearMappingWithShape(Shape)).
   */
  Tensor rearrange(DistributedShape shape, TileMapping mapping = {}) const;

  /**
   * @brief Copy the value to the remote memory.
   *
   * @return RemoteTensor<Type> The remote value.
   */
  RemoteTensor copyToRemote() const;

  /**
   * @brief Copy the value to the host memory. When the copy is complete, the
   * callback is called with the filled HostTensor. The host tensor is only
   * valid in the callback, and will be deallocated after the callback returns.
   */
  void copyToHost(std::function<void(const HostTensor &)> callback) const;

  /**
   * @brief Get the vector norm of the value.
   *
   * @param type The type of the norm.
   * @return Tensor<Type> The norm of the value.
   */
  Expression norm(VectorNorm type) const;

  /** @brief Stores metadata in the tensor.
   *
   * @tparam MetadataType The type of the metadata.
   * @param metadata The metadata to store.
   */
  template <typename MetadataType>
  void setMetadata(MetadataType metadata) {
    metadata_[std::type_index(typeid(MetadataType))] = metadata;
  }

  /** @brief Gets metadata from the tensor.
   *
   * @tparam MetadataType The type of the metadata.
   * @return MetadataType The metadata.
   */
  template <typename MetadataType>
  MetadataType getMetadata() const {
    return std::any_cast<MetadataType>(
        metadata_.at(std::type_index(typeid(MetadataType))));
  }

  /** @brief Checks if the tensor has metadata of the given type.
   *
   * @tparam MetadataType The type of the metadata.
   * @return bool True if the value has metadata of the given type.
   */
  template <typename MetadataType>
  bool hasMetadata() const {
    return metadata_.find(std::type_index(typeid(MetadataType))) !=
           metadata_.end();
  }

  /** @brief Gets the base expression. The expression is guaranteed to be an
   * \ref detail::InputExpr.
   */
  detail::InputExpr &base() const;
};

/** Unrolls the double word arithmetic tensor. A double word arithmetic
 * tensor uses the long long format, which stores both floating point words
 * in a single element. This functions adds one dimension of size 2 in which
 * it stores the upper and lower floating point words seperatly. */
Tensor unrollDoubleWordValue(const Tensor &value);

}  // namespace graphene