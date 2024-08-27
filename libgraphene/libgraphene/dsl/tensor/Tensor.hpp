#pragma once

#include <any>
#include <initializer_list>
#include <optional>
#include <poplar/Graph.hpp>
#include <popops/Reduce.hpp>
#include <typeindex>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/dsl/tensor/Expression.hpp"
#include "libgraphene/matrix/Norm.hpp"

namespace graphene {
using TileMapping = poplar::Graph::TileToTensorMapping;

template <DataType Type>
class RemoteTensor;

/**
 * @brief Represents a mutable tensor, which can be assigned to.
 * This class has value semantics. When assigned or passed-by-value, a copy
 * program for the underlying tensor is generated.
 *
 * @tparam Type The data type of the tensor.
 */
template <DataType Type>
class Tensor : public Expression<Type> {
  // The tile mapping and shape of the tensor. Cached for performance reasons.
  mutable std::optional<TileMapping> tileMapping_;
  mutable std::optional<std::vector<size_t>> shape_;

  std::map<std::type_index, std::any> metadata_;

 public:
  /**
   * @brief Constructs an unitialized mutable tensor.
   *
   * @param shape The shape of the tensor.
   * @param tileMapping The tile mapping of the tensor.
   */
  explicit Tensor(std::vector<size_t> shape = {}, TileMapping tileMapping = {},
                  std::string name = "");

  /**
   * @brief Constructs a mutable tensor with an initial, scalar value.
   *
   * @param value The initial value.
   */
  explicit Tensor(Type value, std::string name = "");

  /**
   * @brief Constructs a mutable tensor from a list of values.
   */
  explicit Tensor(std::initializer_list<Type> values,
                  std::vector<size_t> shape = {}, TileMapping tileMapping = {},
                  std::string name = "");

  /**
   * @brief Constructs a mutable value with a given poplar tensor.
   *
   * @param tensor The poplar tensor.
   */
  explicit Tensor(const poplar::Tensor tensor);

  /**
   * @brief Constructs a mutable value with the initial value of the expression.
   * As with all expressions, only native poplar types are supported.
   *
   * @param expr The expression to initialize from.
   */
  Tensor(Expression<Type> expr)
    requires PoplarNativeType<Type>;

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
  Tensor(Tensor &&value) = default;

  /** Assignment operators */
  Tensor &operator=(const Tensor &value);

  Tensor &operator=(const Expression<Type> &expr)
    requires PoplarNativeType<Type>;

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
   * @brief Get the tile mapping of the tensor.
   *
   * @return TileMapping The tile mapping.
   */
  const TileMapping &tileMapping() const;

  /**
   * @brief Get the shape of the tensor.
   *
   * @return TileMapping The tile mapping.
   */
  const std::vector<size_t> &shape() const;

  /**
   * @brief Print the tensor to the console.
   *
   * @param name Optional name for the tensor.
   * @param fmt Format options for printing.
   */
  void print(std::string name = "", poplar::PrintTensorFmt fmt = {}) const;

  /**
   * @brief Copy the value to the remote memory.
   *
   * @return RemoteTensor<Type> The remote value.
   */
  RemoteTensor<Type> copyToRemote() const;

  /**
   * @brief Reduce the value along the given dimensions.
   * Not yet implemented for double word arithmetic types.
   *
   * @param dims The dimensions to reduce along.
   * @param params The reduction parameters.
   * @param debugContext The debug context.
   * @return Tensor<Type> The reduced value.
   */
  Tensor<Type> reduce(const std::vector<size_t> dims = {0},
                      popops::ReduceParams params = {}) const
    requires PoplarNativeType<Type>;

  /**
   * @brief Get the vector norm of the value.
   * Not yet implemented for double word arithmetic types.
   *
   * @param type The type of the norm.
   * @return Tensor<Type> The norm of the value.
   */
  Expression<Type> norm(VectorNorm type) const
    requires PoplarNativeType<Type>;

  /** @brief Casts a double word type to a single precision float type.
   *
   * @return Expression<float> The casted expression.
   */
  template <typename DestType>
  Tensor<DestType> cast() const
    requires std::is_same_v<Type, doubleword> && std::is_same_v<DestType, float>
  ;

  /** @brief Casts a double precision type to a single precision float type.
   *
   * @return Expression<float> The casted expression.
   */
  template <typename DestType>
  Tensor<DestType> cast() const
    requires std::is_same_v<Type, double> && std::is_same_v<DestType, float>;

  /** @brief Casts a single precision float type to a double word type.
   * type.
   *
   * @return Expression<float> The casted expression.
   */
  template <typename DestType>
  Tensor<DestType> cast() const
    requires std::is_same_v<Type, float> && std::is_same_v<DestType, doubleword>
  ;

  /** @brief Casts a single precision float type to a double precision type.
   * type.
   *
   * @return Expression<float> The casted expression.
   */
  template <typename DestType>
  Tensor<DestType> cast() const
    requires std::is_same_v<Type, float> && std::is_same_v<DestType, double>;

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
};

/** Unrolls the double word arithmetic tensor. A double word arithmetic
 * tensor uses the long long format, which stores both floating point words
 * in a single element. This functions adds one dimension of size 2 in which
 * it stores the upper and lower floating point words seperatly. */
Tensor<float> unrollDoubleWordValue(const Tensor<doubleword> &value);

}  // namespace graphene