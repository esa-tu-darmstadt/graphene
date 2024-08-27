#include "libgraphene/dsl/tensor/Tensor.hpp"

#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Type.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>
#include <poputil/VertexTemplates.hpp>
#include <type_traits>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Traits.hpp"
#include "libgraphene/dsl/tensor/Expression.hpp"
#include "libgraphene/dsl/tensor/Operators.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/PoplarHelpers.hpp"
#include "libgraphene/util/Runtime.hpp"
#include "libgraphene/util/Tracepoint.hpp"
#include "libtwofloat/twofloat.hpp"

namespace graphene {

template <DataType Type>
Tensor<Type>::Tensor(std::vector<size_t> shape, TileMapping tileMapping,
                     std::string name)
    : Expression<Type>(
          Context::graph().addVariable(Traits<Type>::PoplarType, shape, name)) {
  if (tileMapping.empty()) {
    Context::graph().setTileMapping(tensor(), 0);
  } else {
    Context::graph().setTileMapping(tensor(), tileMapping);
  }
}

template <DataType Type>
Tensor<Type>::Tensor(Type value, std::string name)
    : Tensor<Type>({value}, {}, {}, name) {}

template <DataType Type>
Tensor<Type>::Tensor(std::initializer_list<Type> values,
                     std::vector<size_t> shape, TileMapping tileMapping,
                     std::string name)
    : Expression<Type>(Context::graph().addVariable(
          Traits<Type>::PoplarType,
          shape.empty() ? std::vector<size_t>{values.size()} : shape, name)) {
  auto &graph = Context::graph();

  // Remember that we cannot use Graph::setInitialValue, because the initial
  // value is only set once. If the value is used in a loop, the value would not
  // be reset to the initial value in each iteration.

  // Transform the values to the host type if necessary
  using HostType = typename Traits<Type>::PoplarHostType;
  poplar::ArrayRef<HostType> hostValueArrayRef;
  std::vector<HostType> hostValues;
  if constexpr (!std::is_same_v<Type, typename Traits<Type>::PoplarHostType>) {
    hostValues.reserve(values.size());
    for (auto value : values) {
      hostValues.push_back(toPoplarHostType(value));
    }
    hostValueArrayRef = poplar::ArrayRef<HostType>(hostValues);
  } else {
    hostValueArrayRef = values;
  }

  if (tileMapping.empty()) {
    tileMapping = {{poplar::Interval(0, this->tensor().numElements())}};
  }

  poplar::Tensor constant = graph.addConstant(
      Traits<Type>::PoplarType, this->tensor().shape(), hostValueArrayRef);

  graph.setTileMapping(constant, tileMapping);
  graph.setTileMapping(this->tensor(), tileMapping);

  Context::program().add(poplar::program::Copy(constant, this->tensor()));
}

template <DataType Type>
Tensor<Type>::Tensor(const poplar::Tensor tensor) : Expression<Type>(tensor) {
  if (tensor.containsConstant())
    throw std::runtime_error("Tensor must not contain constants");
}

template <DataType Type>
Tensor<Type>::Tensor(const Expression<Type> expr)
  requires PoplarNativeType<Type>
    : Tensor(materializeExpression(expr)) {}

template <DataType Type>
Tensor<Type> &Tensor<Type>::operator=(const Expression<Type> &expr)
  requires PoplarNativeType<Type>
{
  // Materialize the expression into this tensor
  materializeExpression(expr, *this);
  return *this;
}

template <DataType Type>
Tensor<Type> &Tensor<Type>::operator=(const Tensor &value) {
  Context::program().add(poplar::program::Copy(value.tensor(), tensor()));
  return *this;
}

template <DataType Type>
Tensor<Type>::Tensor(const Tensor &value)
    : Tensor(Context::graph().clone(value.tensor())) {
  // Use the assignment operator to copy the tensor
  *this = value;
}

template <DataType Type>
const TileMapping &Tensor<Type>::tileMapping() const {
  if (!tileMapping_) {
    tileMapping_ = Context::graph().getTileMapping(tensor());
  }
  return *tileMapping_;
}

template <DataType Type>
const std::vector<size_t> &Tensor<Type>::shape() const {
  if (!shape_) {
    shape_ = Expression<Type>::shape();
  }
  return *shape_;
}

template <DataType Type>
poplar::Tensor Tensor<Type>::tensor(bool flattenIfScalar) const {
  assert(this->placeholders().size() == 1);
  poplar::Tensor tensor = this->placeholders()[0];
  if (flattenIfScalar && tensor.numElements() == 1) return tensor.flatten()[0];
  return tensor;
}

template <DataType Type>
poplar::Tensor Tensor<Type>::tensorOnTile(size_t tile,
                                          bool flattenIfScalar) const {
  assert(this->placeholders().size() == 1);

  if (this->shape().empty()) return poplar::Tensor();

  // If the first dimension is 1, the tensor is broadcasted to all tiles
  if (this->shape()[0] == 1) {
    return tensor(flattenIfScalar);
  }

  // If the tensor is neither broadcasted nor mapped to the tile, return an
  // empty tensor
  if (tileMapping()[tile].empty()) return poplar::Tensor();

  poplar::Tensor tensor =
      sliceTensorToTile(this->placeholders()[0], tile, &tileMapping());

  if (flattenIfScalar && tensor.numElements() == 1) return tensor.flatten()[0];
  return tensor;
}

template <DataType Type>
void Tensor<Type>::print(std::string name, poplar::PrintTensorFmt fmt) const {
  DebugInfo di("name", DI_ARGS(name));
  auto ipuIntervals = calculateIPUIntervals(tileMapping());
  if (name.empty()) name = "<unnamed>";

  poplar::Tensor unrolledTensor = tensor();
  if constexpr (std::is_same_v<Type, double>) {
    auto unrolledValue = unrollTwoFloatValue(*this);
    unrolledTensor = unrolledValue.tensor();
  }

  for (size_t ipu = 0; ipu < ipuIntervals.size(); ++ipu) {
    if (ipuIntervals[ipu].size() == 0) continue;
    std::string localName = name;
    if (ipuIntervals.size() > 1) localName += "#ipu" + std::to_string(ipu);
    Context::program().add(poplar::program::PrintTensor(
        localName, sliceTensorToIPU(unrolledTensor, ipu), fmt, di));
  }
}

template <DataType Type>
RemoteTensor<Type> Tensor<Type>::copyToRemote() const {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RemoteValue", DI_ARGS(tensor()));

  poplar::Tensor tensor = this->tensor();
  poplar::Graph::TileToTensorMapping tileMapping =
      Context::graph().getTileMapping(tensor);

  auto ipuIntervals = calculateIPUIntervals(tileMapping);
  std::vector<poplar::RemoteBuffer> buffers;
  for (size_t ipu = 0; ipu < ipuIntervals.size(); ++ipu) {
    std::string handle = Runtime::instance().registerHandle(
        "hostToRemote_ipu" + std::to_string(ipu));
    poplar::RemoteBuffer buffer = Context::graph().addRemoteBuffer(
        handle, Traits<Type>::PoplarType, ipuIntervals[ipu].size());
    Context::program().add(
        poplar::program::Copy(tensor.flatten().slice(ipuIntervals[ipu].begin(),
                                                     ipuIntervals[ipu].end()),
                              buffer, di));
    buffers.push_back(buffer);
  }

  return RemoteTensor<Type>(buffers, tileMapping, tensor.shape(),
                            tensor.getDebugStr());
}

template <DataType Type>
Tensor<Type> Tensor<Type>::reduce(const std::vector<size_t> dims,
                                  popops::ReduceParams params) const
  requires PoplarNativeType<Type>
{
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RemoteValue", DI_ARGS(tensor()));

  poplar::Tensor tensor = this->tensor();
  poplar::Tensor reduced = popops::reduce(Context::graph(), tensor, dims,
                                          params, Context::program(), di);
  di.addOutput(reduced);
  return Tensor<Type>(reduced);
}

template <DataType Type>
Expression<Type> Tensor<Type>::norm(VectorNorm type) const
  requires PoplarNativeType<Type>
{
  GRAPHENE_TRACEPOINT();
  DebugInfo di("RemoteValue", DI_ARGS(tensor()));

  switch (type) {
    case VectorNorm::L1:
      return reduce({0}, popops::ReduceParams(popops::Operation::ADD));
    case VectorNorm::L2:
      return ops::Sqrt(
          reduce({0}, popops::ReduceParams(popops::Operation::SQUARE_ADD)));
    case VectorNorm::LINF:
      return reduce({0}, popops::ReduceParams(popops::Operation::MAX));
    default:
      throw std::runtime_error("Invalid vector norm");
  }
}

Tensor<float> unrollTwoFloatValue(const Tensor<double> &value) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Value", DI_ARGS(value.tensor()));
  auto &graph = Context::graph();

  // Add a new dimension of size 2 to the end of the shape
  std::vector<size_t> resultShape = value.tensor().shape();
  resultShape.push_back(2);

  // Double the size of each interval in the tile mapping
  poplar::Graph::TileToTensorMapping resultMapping = value.tileMapping();
  for (auto &intervals : resultMapping) {
    for (auto &interval : intervals) {
      interval = poplar::Interval(interval.begin() * 2, interval.end() * 2);
    }
  }

  Tensor<float> unrolled(resultShape, resultMapping);

  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < value.tileMapping().size(); ++tile) {
    if (value.tileMapping()[tile].empty()) continue;

    // Get in and out tensors on the tile and flatten to a vector
    poplar::Tensor tileTensor =
        value.tensorOnTile(tile).flatten(0, value.shape().size());
    poplar::Tensor unrolledTileTensor =
        unrolled.tensorOnTile(tile).flatten(0, unrolled.shape().size());

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

template <DataType Type>
template <typename DestType>
Tensor<DestType> Tensor<Type>::cast() const
  requires std::is_same_v<Type, doubleword> && std::is_same_v<DestType, float>
{
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Value", DI_ARGS(tensor()));

  auto &graph = Context::graph();
  auto &program = Context::program();

  Tensor<float> casted(this->shape(), this->tileMapping());

  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < tileMapping().size(); ++tile) {
    if (tileMapping()[tile].empty()) continue;

    // FIXME: Use flattenToVector
    poplar::Tensor tileTensor = tensorOnTile(tile).flatten();
    poplar::Tensor castedTileTensor = casted.tensorOnTile(tile).flatten();

    std::string codeletName =
        poputil::templateVertex("graphene::ops::CastDoubleWordToFloatVertex");
    poplar::VertexRef v = graph.addVertex(cs, codeletName);

    graph.connect(v["in"], tileTensor);
    graph.connect(v["out"], castedTileTensor);
    graph.setPerfEstimate(v, tileTensor.numElements() * 3 + 100);
    graph.setTileMapping(v, tile);
  }

  program.add(poplar::program::Execute(cs, di));

  return casted;
}
template <DataType Type>
template <typename DestType>
Tensor<DestType> Tensor<Type>::cast() const
  requires std::is_same_v<Type, float> && std::is_same_v<DestType, doubleword>
{
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Value", DI_ARGS(tensor()));

  auto &graph = Context::graph();
  auto &program = Context::program();

  Tensor<doubleword> casted(this->shape(), this->tileMapping());

  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < tileMapping().size(); ++tile) {
    if (tileMapping()[tile].empty()) continue;

    // FIXME: Use flattenToVector
    poplar::Tensor tileTensor = tensorOnTile(tile).flatten();
    poplar::Tensor castedTileTensor = casted.tensorOnTile(tile).flatten();

    std::string codeletName =
        poputil::templateVertex("graphene::ops::CastFloatToDoubleWordVertex");
    poplar::VertexRef v = graph.addVertex(cs, codeletName);

    graph.connect(v["in"], tileTensor);
    graph.connect(v["out"], castedTileTensor);
    graph.setPerfEstimate(v, tileTensor.numElements() * 3 + 100);
    graph.setTileMapping(v, tile);
  }

  program.add(poplar::program::Execute(cs, di));

  return casted;
}

template <DataType Type>
template <typename DestType>
Tensor<DestType> Tensor<Type>::cast() const
  requires std::is_same_v<Type, double> && std::is_same_v<DestType, float>
{
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Value", DI_ARGS(tensor()));

  auto &graph = Context::graph();
  auto &program = Context::program();

  Tensor<float> casted(this->shape(), this->tileMapping());

  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < tileMapping().size(); ++tile) {
    if (tileMapping()[tile].empty()) continue;

    // FIXME: Use flattenToVector
    poplar::Tensor tileTensor = tensorOnTile(tile).flatten();
    poplar::Tensor castedTileTensor = casted.tensorOnTile(tile).flatten();

    std::string codeletName = poputil::templateVertex(
        "graphene::ops::CastDoublePrecisionToFloatVertex");
    poplar::VertexRef v = graph.addVertex(cs, codeletName);

    graph.connect(v["in"], tileTensor);
    graph.connect(v["out"], castedTileTensor);
    graph.setPerfEstimate(v, tileTensor.numElements() * 3 + 100);
    graph.setTileMapping(v, tile);
  }

  program.add(poplar::program::Execute(cs, di));

  return casted;
}
template <DataType Type>
template <typename DestType>
Tensor<DestType> Tensor<Type>::cast() const
  requires std::is_same_v<Type, float> && std::is_same_v<DestType, double>
{
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Value", DI_ARGS(tensor()));

  auto &graph = Context::graph();
  auto &program = Context::program();

  Tensor<double> casted(this->shape(), this->tileMapping());

  poplar::ComputeSet cs = graph.addComputeSet(di);
  for (size_t tile = 0; tile < tileMapping().size(); ++tile) {
    if (tileMapping()[tile].empty()) continue;

    // FIXME: Use flattenToVector
    poplar::Tensor tileTensor = tensorOnTile(tile).flatten();
    poplar::Tensor castedTileTensor = casted.tensorOnTile(tile).flatten();

    std::string codeletName = poputil::templateVertex(
        "graphene::ops::CastFloatToDoublePrecisionVertex");
    poplar::VertexRef v = graph.addVertex(cs, codeletName);

    graph.connect(v["in"], tileTensor);
    graph.connect(v["out"], castedTileTensor);
    graph.setPerfEstimate(v, tileTensor.numElements() * 3 + 100);
    graph.setTileMapping(v, tile);
  }

  program.add(poplar::program::Execute(cs, di));

  return casted;
}

// Explicit instantiation
#define INSTANTIATE(T) template class Tensor<T>;

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

// Instantiate the cast methods from double to float and vice versa
#define INSTANTIATE(T, U) template Tensor<U> Tensor<T>::cast<U>() const;

INSTANTIATE(doubleword, float)
INSTANTIATE(float, doubleword)

INSTANTIATE(double, float)
INSTANTIATE(float, double)

#undef INSTANTIATE

}  // namespace graphene