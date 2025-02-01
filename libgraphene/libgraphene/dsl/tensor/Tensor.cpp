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

#include "libgraphene/dsl/tensor/Tensor.hpp"

#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Type.hpp>
#include <poputil/VertexTemplates.hpp>
#include <stdexcept>
#include <type_traits>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Shape.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Traits.hpp"
#include "libgraphene/dsl/tensor/Expression.hpp"
#include "libgraphene/dsl/tensor/HostTensor.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/PrintTensorFormat.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/dsl/tensor/details/Expressions.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"
#include "libgraphene/util/Tracepoint.hpp"
#include "libtwofloat/twofloat.hpp"

using namespace graphene;

namespace {
Tensor unrollTwoFloatValue(const Tensor &value) {
  assert(value.type() == Type::TWOFLOAT32);

  GRAPHENE_TRACEPOINT();
  DebugInfo di("Tensor", DI_ARGS(value.tensor()));
  auto &graph = Context::graph();

  // Add a new dimension of size 2 to the end of the shape
  DistributedShape resultShape = value.shape();
  resultShape.push_back(2);

  // Double the size of each interval in the tile mapping
  TileMapping resultMapping = value.tileMapping().scaleUp(2);

  Tensor unrolled =
      Tensor::uninitialized(Type::FLOAT32, resultShape, resultMapping);

  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < value.tileMapping().maxTile(); ++tile) {
    if (value.tileMapping()[tile].empty()) continue;

    // Get in and out tensors on the tile and flatten to a vector
    poplar::Tensor tileTensor =
        value.tensorOnTile(tile).flatten(0, value.shape().rank());
    poplar::Tensor unrolledTileTensor =
        unrolled.tensorOnTile(tile).flatten(0, unrolled.shape().rank());

    std::string codeletName =
        poputil::templateVertex("graphene::ops::UnrollDoubleWordVertex");
    poplar::VertexRef v = graph.addVertex(cs, codeletName);

    graph.connect(v["in"], tileTensor);
    graph.connect(v["out"], unrolledTileTensor);
    graph.setPerfEstimate(v, tileTensor.numElements() * 2 + 100);
    graph.setTileMapping(v, tile);
  }

  Context::program().add(poplar::program::Execute(cs, di));

  return unrolled;
}
}  // namespace

namespace graphene {

/// Create an uninitialized tensor
Tensor::Tensor(TypeRef type, DistributedShape shape, TileMapping tileMapping,
               std::string name)
    : Expression(Context::graph().addVariable(
                     type->poplarEquivalentType()->poplarType(),
                     shape.globalShape().toPoplar(), name),
                 type) {
  if (tileMapping.empty())
    tileMapping = TileMapping::linearMappingWithShape(shape);

  if (!tileMapping.isCompatibleWithShape(shape))
    throw std::invalid_argument("Tile mapping is not compatible with shape");

  Context::graph().setTileMapping(tensor(), tileMapping.toPoplar());
}

/// Create an initialized tensor
template <DataType Type>
Tensor::Tensor(std::initializer_list<Type> values,
               std::optional<DistributedShape> shape, TileMapping tileMapping,
               std::string name)
    : Expression(Context::graph().addVariable(
                     getType<Type>()->poplarEquivalentType()->poplarType(),
                     shape ? shape->globalShape().toPoplar()
                           : std::vector<size_t>{{values.size()}},
                     name),
                 getType<Type>()) {
  auto &graph = Context::graph();

  // Remember that we cannot use Graph::setInitialValue, because the initial
  // value is only set once. If the value is used in a loop, the value would not
  // be reset to the initial value in each iteration.

  // Transform the values to the host type if necessary
  using HostType = typename Traits<Type>::PoplarHostType;
  poplar::ArrayRef<HostType> hostValueArrayRef;
  std::vector<HostType> hostValues;
  hostValues.reserve(values.size());
  for (auto value : values) {
    hostValues.push_back(toPoplarHostType(value));
  }
  hostValueArrayRef = poplar::ArrayRef<HostType>(hostValues);

  if (!shape) {
    if (!tileMapping.empty())
      throw std::invalid_argument(
          "Shape must be provided if tile mapping is provided");
    shape = DistributedShape::onSingleTile({values.size()}, 0);
  }

  if (tileMapping.empty()) {
    tileMapping = TileMapping::linearMappingWithShape(*shape);
  }

  if (!tileMapping.isCompatibleWithShape(*shape))
    throw std::invalid_argument("Tile mapping is not compatible with shape");

  poplar::Tensor constant = graph.addConstant(
      Traits<Type>::PoplarType, this->tensor().shape(), hostValueArrayRef);

  graph.setTileMapping(constant, tileMapping.toPoplar());
  graph.setTileMapping(this->tensor(), tileMapping.toPoplar());

  Context::program().add(poplar::program::Copy(constant, this->tensor()));
}

Tensor::Tensor(const poplar::Tensor tensor, TypeRef type)
    : Expression(tensor, type) {
  if (tensor.containsConstant())
    throw std::runtime_error("Tensor must not contain constants");
}

Tensor::Tensor(const Expression expr) : Tensor(materializeExpression(expr)) {}

Tensor &Tensor::operator=(const Expression &expr) {
  // Materialize the expression into this tensor
  materializeExpression(expr, *this);
  return *this;
}

Tensor &Tensor::operator=(const Tensor &value) {
  // Copy one tensor to another. This might be more effective than materializing
  // the tensor into this tensor.
  Context::program().add(poplar::program::Copy(value.tensor(), tensor()));
  return *this;
}

Tensor::Tensor(const Tensor &value)
    : Tensor(Context::graph().clone(value.tensor()), value.type()) {
  // Use the assignment operator to copy the tensor
  *this = value;
}

Tensor Tensor::same() const { return Tensor::fromPoplar(tensor(), type()); }

detail::InputExpr &Tensor::base() const {
  detail::InputExpr *inputExpr =
      dynamic_cast<detail::InputExpr *>(&Expression::base());
  assert(inputExpr);
  return *inputExpr;
}

poplar::Tensor Tensor::tensor(bool flattenIfScalar) const {
  poplar::Tensor tensor = base().tensor();
  if (flattenIfScalar && tensor.numElements() == 1) return tensor.flatten()[0];
  return tensor;
}

poplar::Tensor Tensor::tensorOnTile(size_t tile, bool flattenIfScalar) const {
  // If the first dimension is 1, the tensor is broadcasted to all tiles
  if (this->shape()[0] == 1) {
    return tensor(flattenIfScalar);
  }

  // If the tensor is neither broadcasted nor mapped to the tile, return an
  // empty tensor
  if (tileMapping()[tile].empty()) return poplar::Tensor();

  poplar::Tensor tensor =
      sliceTensorToTile(this->tensor(), tile, tileMapping());

  if (flattenIfScalar && tensor.numElements() == 1) return tensor.flatten();
  return tensor;
}

Tensor Tensor::rearrange(DistributedShape targetShape,
                         TileMapping tileMapping) const {
  if (shape().globalShape() != targetShape.globalShape())
    throw std::runtime_error(
        "Cannot rearrange tensor to shape with different shape");

  if (tileMapping.empty())
    tileMapping = TileMapping::linearMappingWithShape(targetShape);

  Tensor rearranged = Tensor::uninitialized(type(), targetShape, tileMapping);
  Context::program().add(poplar::program::Copy(tensor(), rearranged.tensor()));
  return rearranged;
}

namespace {
/// Recursively print the values of a tensor. The offset is the index of the
/// first element of the given dimension, in number of elements (not number of
/// bytes)
void printTensorRecursive(std::ostream &stream, const uint8_t *data,
                          TypeRef type, const TensorShape &shape, size_t dim,
                          size_t offset, const PrintTensorFormat &fmt,
                          const std::vector<bool> &dimsToSummarize,
                          std::string indent = "") {
  // If we are at the last dimension, print the values
  if (dim == shape.rank() - 1) {
    stream << indent << "[";
    for (size_t i = 0; i < shape[dim]; ++i) {
      if (i > 0) stream << ", ";
      if (dimsToSummarize[dim] && i == fmt.edgeItems) {
        stream << "..., ";
      }
      size_t byteOffset = (offset + i) * type->size();
      type->prettyPrintValue(data + byteOffset, stream);
    }
    stream << "]";
    return;
  }

  // If we are not at the last dimension, print the values of the next dimension
  stream << indent << "[" << std::endl;
  for (size_t i = 0; i < shape[dim]; ++i) {
    if (i > 0) stream << "\n";
    if (dimsToSummarize[dim] && i == fmt.edgeItems) {
      stream << indent << "  "
             << "..." << std::endl;
    }

    size_t subOffset = offset + i * shape.stride(dim);
    printTensorRecursive(stream, data, type, shape, dim + 1, subOffset, fmt,
                         dimsToSummarize, indent + "  ");
  }
  stream << std::endl << indent << "]";
}
}  // namespace

void Tensor::print(std::string name, PrintTensorFormat fmt,
                   std::ostream &stream) const {
  DebugInfo di("Tensor", DI_ARGS(name));
  if (name.empty()) name = "<unnamed>";

  // Use a custom print function instead of program::PrintTensor to support
  // non-native types (e.g., doubleword)
  auto &runtime = Runtime::instance();
  auto &graph = Context::graph();
  auto &program = Context::program();

  std::string handle = runtime.registerHandle("PrintTensor");
  poplar::Type poplarType = type()->poplarEquivalentType()->poplarType();

  poplar::Tensor tensorToPrint = tensor();
  TensorShape summarizedShape = shape().globalShape();
  std::vector<bool> dimsToSummarize(shape().rank(), false);

  // Slice away what isnt printed due to summarisation so that it isnt copied
  // to the host
  for (size_t i = 0; i < shape().rank(); ++i) {
    if (shape()[i] > fmt.summariseThreshold) {
      tensorToPrint = poplar::concat(
          tensorToPrint.slice(0, fmt.edgeItems, i),
          tensorToPrint.slice(shape()[i] - fmt.edgeItems, shape()[i], i), i);
      summarizedShape[i] = fmt.edgeItems * 2;
      dimsToSummarize[i] = true;
    }
  }

  auto func = graph.addHostFunction(
      handle, {{poplarType, summarizedShape.numElements()}}, {});

  runtime.registerHostFunction(
      handle, [summarizedShape, originalShape = shape().globalShape(),
               name = name, type = type(), fmt, dimsToSummarize,
               &stream](poplar::ArrayRef<const void *> ins,
                        poplar::ArrayRef<void *> /*outs*/) {
        stream << "tensor<" << originalShape.str() << "x" << type->str() << "> "
               << name << " = ";
        switch (fmt.floatFormat) {
          case PrintTensorFormat::FloatFormat::Auto:
            stream << std::defaultfloat;
            break;
          case PrintTensorFormat::FloatFormat::Fixed:
            stream << std::fixed;
            break;
          case PrintTensorFormat::FloatFormat::Scientific:
            stream << std::scientific;
            break;
          case PrintTensorFormat::FloatFormat::HexFloat:
            stream << std::hexfloat;
            break;
        }
        auto oldprecision = stream.precision(fmt.precision);

        printTensorRecursive(stream, reinterpret_cast<const uint8_t *>(ins[0]),
                             type, summarizedShape, 0, 0, fmt, dimsToSummarize,
                             "");
        stream << std::endl << std::defaultfloat;
        stream.precision(oldprecision);
      });

  program.add(poplar::program::Call(func, {tensorToPrint}, {}, di));
}

void Tensor::copyToHost(
    std::function<void(const HostTensor &tensor)> callback) const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Tensor", DI_ARGS(tensor()));

  auto &runtime = Runtime::instance();
  auto &graph = Context::graph();
  auto &program = Context::program();

  std::string handle = runtime.registerHandle("CopyToHost");
  poplar::Type poplarType = type()->poplarEquivalentType()->poplarType();

  auto func = graph.addHostFunction(handle, {{poplarType, numElements()}}, {});

  runtime.registerHostFunction(
      handle,
      [callback = callback, shape = shape(), tileMapping = tileMapping(),
       type = type(),
       name = tensor().getDebugStr()](poplar::ArrayRef<const void *> ins,
                                      poplar::ArrayRef<void *> /*outs*/) {
        HostTensor tensor = HostTensor::createTemporary(
            const_cast<void *>(ins[0]), shape, tileMapping, name, type);
        callback(tensor);
      });

  program.add(poplar::program::Call(func, {tensor()}, {}, di));
}

RemoteTensor Tensor::copyToRemote() const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Tensor", DI_ARGS(tensor()));

  std::unordered_map<size_t, poplar::RemoteBuffer> buffers;
  TileMapping ipuMapping = tileMapping().translateToIPUMapping(
      Context::graph().getTarget().getTilesPerIPU());

  size_t numIPUs = ipuMapping.maxTile() + 1;
  for (size_t ipu = 0; ipu < numIPUs; ++ipu) {
    size_t numElements = ipuMapping.numElementsOnTile(ipu);
    if (numElements == 0) continue;

    std::string handle = Runtime::instance().registerHandle(
        "hostToRemote_ipu" + std::to_string(ipu));
    poplar::RemoteBuffer buffer = Context::graph().addRemoteBuffer(
        handle, type()->poplarEquivalentType()->poplarType(), numElements);
    buffers[ipu] = buffer;
    Context::program().add(poplar::program::Copy(
        sliceTensorToIPU(tensor(), ipu, tileMapping()), buffer, di));
  }

  return RemoteTensor(type(), buffers, tileMapping(), shape(),
                      tensor().getDebugStr());
}

Expression Tensor::norm(VectorNorm type) const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Tensor", DI_ARGS(tensor()));

  switch (type) {
    case VectorNorm::L1:
      return reduce(0, ReduceOperation::ADD);
    case VectorNorm::L2:
      return ops::Sqrt(reduce(0, ReduceOperation::SQUARE_ADD));
    case VectorNorm::LINF:
      return reduce(0, ReduceOperation::MAX);
    default:
      throw std::runtime_error("Invalid vector norm");
  }
}

// Explicit instantiation
#define INSTANTIATE(T)                                                  \
  template Tensor::Tensor(std::initializer_list<T>,                     \
                          std::optional<DistributedShape>, TileMapping, \
                          std::string);

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
#undef INSTANTIATE

}  // namespace graphene