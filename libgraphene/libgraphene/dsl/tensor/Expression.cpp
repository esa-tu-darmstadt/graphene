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

#include "libgraphene/dsl/tensor/Expression.hpp"

#include <spdlog/spdlog.h>

#include <any>
#include <cstddef>
#include <iterator>
#include <stdexcept>

#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Helpers.hpp"
#include "libgraphene/common/TileMapping.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/dsl/code/CodeGen.hpp"
#include "libgraphene/dsl/code/ControlFlow.hpp"
#include "libgraphene/dsl/code/Execute.hpp"
#include "libgraphene/dsl/code/Operators.hpp"
#include "libgraphene/dsl/code/Value.hpp"
#include "libgraphene/dsl/tensor/Execute.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/dsl/tensor/details/Expressions.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"
#include "libgraphene/util/Tracepoint.hpp"

using namespace graphene;
namespace {
struct ReductionOptions {
  size_t reductionDim;
  bool reducePerWorker;
  ReduceOperation op;
};

/// Initializes a Tensor to an identity/initial value suitable for the given
/// reduction operation. For each reduction type, the tensor is filled with a
/// constant that represents the 'neutral' element in the reduction (e.g., 0 for
/// ADD, 1 for MUL, etc.).
void setInitialReductionValue(Tensor &tensor, ReduceOperation op) {
  typeSwitch(tensor.type(), [&]<DataType Type>() {
    switch (op) {
      case ReduceOperation::ADD:
      case ReduceOperation::SQUARE_ADD:
        tensor = (Type)0;
        break;
      case ReduceOperation::MUL:
        tensor = (Type)1;
        break;
      case ReduceOperation::MAX:
        tensor = std::numeric_limits<Type>::min();
        break;
      case ReduceOperation::MIN:
        tensor = std::numeric_limits<Type>::max();
        break;
      case ReduceOperation::LOGICAL_AND:
        tensor = (Type)1;
        break;
      case ReduceOperation::LOGICAL_OR:
        tensor = (Type)0;
        break;
    }
  });
}

/// Visitor that collects the input tensors of an expression
struct CollectInputsVisitor : detail::ExpressionVisitor {
  std::vector<poplar::Tensor> &tensors;
  std::vector<TypeRef> &types;
  std::vector<codedsl::VertexInOutType::Direction> &directions;
  std::vector<DistributedShape> &shapes;

  CollectInputsVisitor(
      std::vector<poplar::Tensor> &tensors, std::vector<TypeRef> &types,
      std::vector<codedsl::VertexInOutType::Direction> &directions,
      std::vector<DistributedShape> &shapes)
      : tensors(tensors),
        types(types),
        directions(directions),
        shapes(shapes) {}

  std::any visit(detail::InputExpr &expr, std::any arg) final {
    tensors.push_back(expr.tensor());
    types.push_back(expr.type());
    directions.push_back(codedsl::VertexInOutType::Direction::Input);
    shapes.push_back(expr.shape());
    return {};
  }
};

/// Calculates the index in a flattened tensor based on per-dimension indices,
/// accounting for broadcasting. The size of the first dimension
///  (`shape[0]`) is not used.
///  If a dimension is broadcasted (size == 1), the index is ignored for that
///  dimension. This function can handle shapes of different ranks by ignoring
///  leading broadcast dimensions that don't appear in the original shape.
codedsl::Value getFlattenedIndex(std::vector<codedsl::Value> indices,
                                 DistributedShape shape) {
  // Allows setting the value to a different value instead of triggering an
  // assignment
  std::unique_ptr<codedsl::Value> flattenedIndex =
      std::make_unique<codedsl::Value>((uint32_t)0);

  uint32_t stride = 1;
  for (int dim = shape.rank() - 1; dim >= 0; --dim) {
    // A dimension of 1 in the input shape means we ignore the index,
    // broadcasting the input value along that dimension
    if (shape[dim] == 1) continue;

    // Ignore the leading dimensions that are present in the output but not in
    // the input. This effectively adds broadcast dimensions to the start of
    // the input shape.
    codedsl::Value index = indices[indices.size() - shape.rank() + dim];

    flattenedIndex =
        std::make_unique<codedsl::Value>(*flattenedIndex + index * stride);

    stride *= shape[dim];
  }

  return *flattenedIndex;
}

/// Tries to find the size of the first dimension of the iteration space on the
/// executing tile. Without reduction, this could be calculated based on the
/// number of elements in the output tensor and its shape. When reducing in the
/// first dimension, we try to find it based on any other input tensor with the
/// same first dimension mapping. This may fail for something like
/// `Expression(0).broadcast({100}).reduce(0)` too, as the shape information is
/// not available in the vertex.
std::optional<codedsl::Value> getFirstDimensionSize(
    std::vector<codedsl::Value> tensors, std::vector<DistributedShape> shapes,
    DistributedShape iterationSpace) {
  assert(tensors.size() == shapes.size());

  ssize_t index = -1;
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (shapes[i].firstDimDistribution() ==
        iterationSpace.firstDimDistribution()) {
      index = (ssize_t)i;
      break;
    }
  }
  if (index == -1) {
    return std::nullopt;
  }

  unsigned stride = shapes[index].stride(0);
  return codedsl::Variable(tensors[index].size() / stride);
}

/// A visitor that translates an expression tree into a CodeDSL expression
struct GenerateCodeForExpressionVisitor : detail::ExpressionVisitor {
  std::vector<codedsl::Value> &inputs;
  unsigned currentInputIndex = 0;

  GenerateCodeForExpressionVisitor(std::vector<codedsl::Value> &inputs)
      : inputs(inputs) {}

  std::any visit(detail::ConstExpr &expr, std::any indices) final {
    // Indices are ignored, i.e., the constant is broadcasted to all elements
    (void)indices;
    return (codedsl::Value)codedsl::Expression(expr.type(),
                                               expr.valueAsString());
  }

  std::any visit(detail::InputExpr &expr, std::any indices) final {
    auto indexVector = std::any_cast<std::vector<codedsl::Value>>(indices);

    DistributedShape inputShape = expr.shape();
    codedsl::Value input = inputs[currentInputIndex++];

    assert(inputShape.rank() <= indexVector.size() &&
           "output shape too small for given indices");

    codedsl::Value flattenedIndex = getFlattenedIndex(indexVector, inputShape);

    return input[flattenedIndex];
  }
  std::any visit(detail::BroadcastExpr &expr, std::any indices) final {
    return expr.arg()->accept(*this, indices);
  }

  std::any visit(detail::BinaryExpr &expr, std::any indices) final {
    auto lhs =
        std::any_cast<codedsl::Value>(expr.lhs()->accept(*this, indices));
    auto rhs =
        std::any_cast<codedsl::Value>(expr.rhs()->accept(*this, indices));
    return codedsl::Value(codedsl::operation(expr.op(), lhs, rhs));
  }
  std::any visit(detail::UnaryExpr &expr, std::any indices) final {
    auto arg =
        std::any_cast<codedsl::Value>(expr.arg()->accept(*this, indices));
    return codedsl::Value(codedsl::operation(expr.op(), arg));
  }
  std::any visit(detail::CastExpr &expr, std::any indices) final {
    auto value =
        std::any_cast<codedsl::Value>(expr.arg()->accept(*this, indices));
    return value.cast(expr.type());
  }
  std::any visit(detail::PermuteExpr &expr, std::any indices) final {
    auto indexVector = std::any_cast<std::vector<codedsl::Value>>(indices);
    std::vector<codedsl::Value> permutedIndices;

    for (size_t i = 0; i < expr.permutation().size(); ++i) {
      // Take broadcasted dimensions into account
      size_t permutedIndex = expr.permutation()[i] + indexVector.size() -
                             expr.permutation().size();

      permutedIndices.push_back(indexVector[permutedIndex]);
    }

    return expr.arg()->accept(*this, permutedIndices);
  }
};

/// Materializes an expression into a poplar tensor. The output shape may differ
/// from `expr.shape()` if reduction is applied.
void materializeExpressionInto(
    const Expression &expr, poplar::Tensor tensor, DistributedShape outputShape,
    std::optional<ReductionOptions> reductionOptions = std::nullopt) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Expression");

  spdlog::trace("Materializing expression: {}", expr.asString());

  detail::ExpressionBase &exprBase = expr.base();

  if (dynamic_cast<detail::ConstExpr *>(&exprBase)) {
    // TODO: Maybe use an optimized "fill" operation
  }

  std::vector<poplar::Tensor> inputsAndOutputTensors;
  std::vector<TypeRef> inputsAndOutputTypes;
  std::vector<codedsl::VertexInOutType::Direction> inputsAndOutputDirections;
  std::vector<DistributedShape> inputsAndOutputShapes;

  // Collect the input tensors
  {
    CollectInputsVisitor visitor(inputsAndOutputTensors, inputsAndOutputTypes,
                                 inputsAndOutputDirections,
                                 inputsAndOutputShapes);
    exprBase.accept(visitor);
  }

  // Add the output tensor
  inputsAndOutputTensors.push_back(tensor);
  inputsAndOutputTypes.push_back(expr.type());
  inputsAndOutputDirections.push_back(
      codedsl::VertexInOutType::Direction::Output);
  inputsAndOutputShapes.push_back(outputShape);

  // FIXME: Find a better way to determine the dimension to partition across
  size_t dimensionToPartitionAcrossWorkers = 0;
  bool multipleWorkers = expr.numElements() > 12;

  if (reductionOptions) {
    multipleWorkers = reductionOptions->reducePerWorker;
    dimensionToPartitionAcrossWorkers = reductionOptions->reductionDim;
  }

  auto code = [&](std::vector<codedsl::Value> args) {
    // If this is a MultiVertex, the first argument is the worker ID
    std::optional<codedsl::Value> workerID;
    auto firstInputIt = args.begin();
    if (multipleWorkers) {
      workerID = args[0];
      firstInputIt++;
    }

    codedsl::Value output = args.back();
    std::vector<codedsl::Value> inputsAndOutputVariables(firstInputIt,
                                                         args.end());
    GenerateCodeForExpressionVisitor visitor(inputsAndOutputVariables);

    // Calculate the shape of the iteration range (aka the input of the
    // expression) on the executing tile. Tensors are distributed across tiles
    // in the first dimension, so the size of the first dimension is different
    // for each tile. Because we dont want to generate different code for each
    // tile, we need to calculate the size of the first dimension on the
    // executing tile based on the number of elements in the input or output
    // tensors mapped to the executing tile.
    std::vector<codedsl::Value> iterationSpaceShape;
    for (size_t dim = 0; dim < expr.shape().rank(); ++dim) {
      if (dim == 0) {
        // Find the size of the first dimension
        auto firstDimSize = getFirstDimensionSize(
            inputsAndOutputVariables, inputsAndOutputShapes, expr.shape());
        if (!firstDimSize) {
          throw std::runtime_error(
              "Could not determine the size of the first dimension of the "
              "output tensor");
        }
        iterationSpaceShape.push_back(*firstDimSize);
      } else
        iterationSpaceShape.push_back(expr.shape()[dim]);
    }

    std::vector<codedsl::Value> inputIndices;
    std::vector<codedsl::Value> outputIndices;

    for (size_t dim = 0; dim < iterationSpaceShape.size(); ++dim) {
      // The outer loop parallelizes the computation across workers
      codedsl::Value numElements = iterationSpaceShape[dim];
      std::optional<codedsl::Value> loopIndex;
      if (multipleWorkers && dim == dimensionToPartitionAcrossWorkers) {
        // Partition this dimension across the 6 workers.
        codedsl::Value chunkSize = numElements / 6u;
        // Align the chunk size, so that larger loads / stores can be done and
        // vectorization can be used
        codedsl::Value alignedChunkSize = (chunkSize / 4u) * 4u;
        codedsl::Variable start(alignedChunkSize * workerID.value());
        codedsl::Variable end(codedsl::Select(
            workerID.value() == 5u, numElements, start + alignedChunkSize));

        loopIndex = codedsl::detail::ForStart(start, end, 1);
      } else {
        loopIndex = codedsl::detail::ForStart(0, numElements, 1);
      }

      inputIndices.push_back(loopIndex.value());
      if (reductionOptions && dim == reductionOptions->reductionDim) {
        // Use either the worker ID or 0 for the output index in the reduction
        // dimension
        outputIndices.push_back(workerID.value_or(0u));
      } else {
        outputIndices.push_back(*loopIndex);
      }
    }

    codedsl::Variable value =
        std::any_cast<codedsl::Value>(exprBase.accept(visitor, inputIndices));
    codedsl::Value outFlattenedIndex =
        getFlattenedIndex(outputIndices, outputShape);

    if (reductionOptions) {
      switch (reductionOptions->op) {
        case ReduceOperation::ADD:
          output[outFlattenedIndex] = output[outFlattenedIndex] + value;
          break;
        case ReduceOperation::MUL:
          output[outFlattenedIndex] = output[outFlattenedIndex] * value;
          break;
        case ReduceOperation::SQUARE_ADD:
          output[outFlattenedIndex] = output[outFlattenedIndex] + value * value;
          break;
        case ReduceOperation::LOGICAL_AND:
          output[outFlattenedIndex] = output[outFlattenedIndex] && value;
          break;
        case ReduceOperation::LOGICAL_OR:
          output[outFlattenedIndex] = output[outFlattenedIndex] || value;
          break;
        default:
          throw std::runtime_error("Unsupported reduction operation");
      }
    } else {
      output[outFlattenedIndex] = value;
    }

    for (size_t dim = 0; dim < iterationSpaceShape.size(); ++dim) {
      codedsl::detail::ForEnd();
    }
  };

  // Collect the member vars
  std::vector<codedsl::Vertex::MemberVarInfo> memberVars;
  for (size_t i = 0; i < inputsAndOutputTensors.size(); ++i) {
    memberVars.push_back(codedsl::Vertex::MemberVarInfo::create(
        inputsAndOutputTypes[i], inputsAndOutputTensors[i],
        inputsAndOutputDirections[i]));
  }
  codedsl::ExecuteAsMapped(memberVars,
                           multipleWorkers ? codedsl::VertexKind::MultiVertex
                                           : codedsl::VertexKind::Vertex,
                           code);
}
}  // namespace

namespace graphene {

Tensor materializeExpression(const Expression &expr) {
  Tensor tensor = Tensor::uninitialized(expr.type(), expr.shape(),
                                        expr.tileMapping(), expr.asString());
  materializeExpressionInto(expr, tensor.tensor(), expr.shape());
  return tensor;
}

void materializeExpression(const Expression &expr, Tensor &value) {
  GRAPHENE_TRACEPOINT();
  DebugInfo di("Expression");

  spdlog::trace("Materializing expression into existing value: {}",
                expr.asString());

  Expression src = expr;

  if (value.type() != expr.type()) {
    spdlog::warn(
        "Implicitly casting expression from {} to {} during assignment",
        expr.type()->str(), value.type()->str());
    src = src.cast(value.type());
  }
  if (value.shape() != expr.shape()) {
    spdlog::warn("Implicitly broadcasting expression during assignment");
    src = src.broadcast(value.shape());
  }

  materializeExpressionInto(src, value.tensor(), value.shape());
}

const TileMapping &Expression::tileMapping() const {
  if (!tileMapping_) {
    tileMapping_ = base().tileMapping();
  }
  return *tileMapping_;
}

const DistributedShape &Expression::shape() const {
  if (!shape_) {
    shape_ = base().shape();
  }
  return *shape_;
}

TypeRef Expression::type() const { return expr_->type(); }
std::string Expression::asString() const { return expr_->getAsString(); }

size_t Expression::rank() const { return shape().rank(); }
size_t Expression::numElements() const { return shape().numElements(); }

Expression Expression::reducePerTile(size_t dim, ReduceOperation op) const {
  // If the dimension is already reduced, just copy into a new tensor. This also
  // prevents reducing along broadcasted dimensions.
  if (shape()[dim] == 1) return *this;

  spdlog::trace("Reducing expression per tile: {}", asString());

  // If the dimension is large enough, reduce per worker thread first
  bool reducePerWorkerFirst = false;
  const size_t minSizeForWorkerReduction = 6;
  if (dim == 0) {
    // Reduce per worker first if profitable for any tile
    reducePerWorkerFirst =
        std::any_of(shape().firstDimDistribution().begin(),
                    shape().firstDimDistribution().end(),
                    [minSizeForWorkerReduction](auto &&pair) {
                      return pair.value > minSizeForWorkerReduction;
                    });
  } else {
    reducePerWorkerFirst = shape()[dim] > minSizeForWorkerReduction;
  }

  Expression intermediate =
      reducePerWorkerFirst ? reducePerWorker(dim, op) : *this;

  // Now, Intermediate has at most 6 elements in the reduction dimension per
  // tile

  // Change the size of the reduction dimension to 1
  DistributedShape outputShape = intermediate.shape();
  if (dim == 0) {
    // If the first dimension is reduced, set the size to 1 for all involved
    // tiles
    size_t numTiles = 0;
    for (auto [tile, size] : outputShape.firstDimDistribution()) {
      outputShape.firstDimDistribution()[tile] = 1;
      numTiles++;
    }
    outputShape.globalShape()[0] = numTiles;
  } else {
    outputShape.globalShape()[dim] = 1;
  }

  TileMapping outputTileMapping =
      TileMapping::linearMappingWithShape(outputShape);

  Tensor output = Tensor::uninitialized(intermediate.type(), outputShape,
                                        outputTileMapping, "reducePerTile");

  // The reduction vertex expects the output to be initialized appropriately
  setInitialReductionValue(output, op);

  materializeExpressionInto(intermediate, output.tensor(), outputShape,
                            ReductionOptions{dim, false, op});

  return output;
}

Expression Expression::reducePerWorker(size_t dim, ReduceOperation op) const {
  // If the dimension is already reduced, just copy into a new tensor. This also
  // prevents reducing along broadcasted dimensions.
  if (shape()[dim] == 1) return *this;

  spdlog::trace("Reducing expression per worker: {}", asString());

  // Change the size of the reduction dimension to 6
  DistributedShape outputShape = shape();
  if (dim == 0) {
    // If the first dimension is reduced, set the size to 6 for all involved
    // tiles
    size_t numTiles = 0;
    for (auto [tile, size] : outputShape.firstDimDistribution()) {
      outputShape.firstDimDistribution()[tile] = 6;
      numTiles++;
    }
    outputShape.globalShape()[0] = 6 * numTiles;

  } else {
    outputShape.globalShape()[dim] = 6;
  }

  TileMapping outputTileMapping =
      TileMapping::linearMappingWithShape(outputShape);

  Tensor output = Tensor::uninitialized(type(), outputShape, outputTileMapping,
                                        "reducePerWorker");

  // The reduction vertex expects the output to be initialized appropriately
  setInitialReductionValue(output, op);

  materializeExpressionInto(*this, output.tensor(), outputShape,
                            ReductionOptions{dim, true, op});

  return output;
}

// FIXME: We need something similiar for reductionGlobal. If we specify a
// coarsness, we can also do multiple stages per IPU reduction.
Expression Expression::reduceGrouped(size_t groupSize,
                                     ReduceOperation op) const {
  if (shape()[0] == 1) return *this;

  // Per-tile reduction is necessary first if any tile has more than one
  // "element" in the first dimension
  bool reducePerTileFirst =
      std::any_of(shape().firstDimDistribution().begin(),
                  shape().firstDimDistribution().end(),
                  [](auto &&pair) { return pair.value > 1; });

  Expression intermediate = reducePerTileFirst ? reducePerTile(0, op) : *this;

  spdlog::trace("Reducing expression per group size {}: {}", groupSize,
                asString());

  // Now, Intermediate has at most 1 element in the reduction dimension per tile

  // Create a rearranged tensor with the same shape as the intermediate tensor
  // but mapped to one tile per group. This collects the results of the per-tile
  // reductions in each group.
  DistributedShape groupedShape =
      intermediate.shape().groupFirstDimension(groupSize);

  // Check if the shape is already grouped
  if (groupedShape == intermediate.shape()) return intermediate;

  Tensor rearranged =
      intermediate.materializeIfNecessary().rearrange(groupedShape);

  // Finally, reduce the aggregated tensors on each IPU
  return rearranged.reducePerTile(0, op);
}

Expression Expression::reduce(size_t dim, ReduceOperation op) const {
  GRAPHENE_TRACEPOINT();
  // DebugInfo di("RemoteValue", DI_ARGS(tensor()));

  if (dim >= rank()) {
    throw std::invalid_argument("Invalid dimension for reduction");
  }

  // Reduce the expression in up to 4 stages:
  // 1. Reduce per worker
  // 2. Reduce per tile (across workers)
  //
  // Tensors are distributed across tiles in the first dimension. Thus, if the
  // reduction dimension is the first dimension, we must reduce across tiles and
  // IPUs too:
  // 3. Reduce per IPU (across tiles)
  // 4. Reduce globally (across IPUs)

  // Reduce on worker threads if the tensor is large enough
  Expression stage1 = *this;
  if (shape()[dim] > 6) {
    stage1 = reducePerWorker(dim, op);
    // Make sure we square only once
    if (op == ReduceOperation::SQUARE_ADD) op = ReduceOperation::ADD;
  }

  // Reduce on tiles
  Expression stage2 = stage1.reducePerTile(dim, op);
  // Make sure we square only once
  if (op == ReduceOperation::SQUARE_ADD) op = ReduceOperation::ADD;

  // Reduce between tiles on the same IPU if first dimension is reduced
  if (dim != 0) return stage2;

  Expression stage3 = stage2.reduceGrouped(1472, op);
  Expression stage4 = stage3.reduceGrouped(999999999, op);

  return stage4;
}

Expression Expression::permute(std::vector<size_t> permutation) const {
  if (permutation.size() != rank()) {
    throw std::runtime_error("Invalid permutation size");
  }

  if (permutation[0] != 0) {
    throw std::runtime_error("Permutation must keep the first dimension");
  }
  return Expression(
      std::make_unique<detail::PermuteExpr>(expr_->clone(), permutation));
}

Expression Expression::broadcast(DistributedShape shape) const {
  if (shape.rank() < rank()) {
    throw std::runtime_error("Cannot broadcast to a smaller rank");
  }
  return Expression(
      std::make_unique<detail::BroadcastExpr>(expr_->clone(), shape));
}

Tensor Expression::materialize() const { return materializeExpression(*this); }

Tensor Expression::materializeIfNecessary() const {
  if (auto expr = dynamic_cast<detail::InputExpr *>(expr_.get())) {
    return Tensor::fromPoplar(expr->tensor(), expr->type());
  }
  return materialize();
}

void Expression::print(std::string name) const {
  materializeIfNecessary().print(name);
}

Expression::Expression(const Expression &expr)
    : expr_(expr.expr_->clone()),
      tileMapping_(expr.tileMapping_),
      shape_(expr.shape_) {}

Expression &Expression::operator=(const Expression &expr) {
  expr_ = expr.expr_->clone();
  shape_ = expr.shape_;
  tileMapping_ = expr.tileMapping_;
  return *this;
}

Expression &Expression::operator=(Expression &&expr) noexcept = default;

Expression::~Expression() = default;

Expression::Expression(const Tensor &tensor)
    : expr_(
          std::make_unique<detail::InputExpr>(tensor.tensor(), tensor.type())),
      tileMapping_(tensor.tileMapping()),
      shape_(tensor.shape()) {}

Expression::Expression(std::unique_ptr<detail::ExpressionBase> expr)
    : expr_(std::move(expr)) {}

Expression::Expression(poplar::Tensor tensor, TypeRef type)
    : expr_(std::make_unique<detail::InputExpr>(tensor, type)) {}

Expression::Expression(Expression &&expr) noexcept = default;

Expression Expression::cast(TypeRef destType) const {
  return Expression(
      std::make_unique<detail::CastExpr>(expr_->clone(), destType));
}

template <DataType T>
Expression::Expression(T value)
    : expr_(std::make_unique<detail::ConstExpr>(value)) {}

// Explicit instantiation
#define INSTANTIATE(T) template Expression::Expression(T value);

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