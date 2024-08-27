#include "libgraphene/dsl/tensor/ControlFlow.hpp"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <poplar/CycleCount.hpp>
#include <poplar/Program.hpp>
#include <poplar/SyncType.hpp>

#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"

using namespace graphene;

void graphene::cf::If(Expression<bool> condition,
                      std::function<void()> trueBody,
                      std::function<void()> falseBody) {
  if (condition.numElements() != 1)
    throw std::runtime_error("Condition must be a scalar");

  auto &program = Context::program();
  DebugInfo di("cf");

  Tensor<bool> predicate = condition;

  poplar::program::Sequence trueSeq, falseSeq, conditionSeq;
  {
    Context context(trueSeq);
    trueBody();
  }

  if (falseBody) {
    Context context(falseSeq);
    falseBody();
  }

  program.add(
      poplar::program::If(predicate.tensor(true), trueSeq, falseSeq, di));
}

void graphene::cf::While(Expression<bool> condition,
                         std::function<void()> body) {
  if (condition.numElements() != 1)
    throw std::runtime_error("Condition must be a scalar");

  auto &program = Context::program();
  DebugInfo di("cf");

  Tensor<bool> predicate;

  poplar::program::Sequence conditionProg;
  {
    Context context(conditionProg);
    predicate = condition;
  }

  poplar::program::Sequence bodyProg;
  {
    Context context(bodyProg);
    body();
  }

  program.add(poplar::program::RepeatWhileTrue(
      conditionProg, predicate.tensor(true), bodyProg, di));
}

void graphene::cf::Repeat(int count, std::function<void()> body) {
  auto &program = Context::program();
  DebugInfo di("cf");
  poplar::program::Sequence seq;

  {
    Context context(seq);
    body();
  }

  program.add(poplar::program::Repeat(count, seq, di));
}

Tensor<unsigned> graphene::cf::Time(std::function<void()> body, size_t tile) {
  DebugInfo di("cf");
  poplar::program::Sequence seq;
  {
    Context context(seq);
    body();
  }

  // The first element of the tensor is the lower 32 bits and the second element
  // is the upper 32 bits.
  auto tensor = poplar::cycleCount(Context::graph(), seq, tile,
                                   poplar::SyncType::GLOBAL, di);

  Context::program().add(seq);

  return Tensor<unsigned>(tensor.slice(0, 1));
}

template <DataType RetType>
std::tuple<Tensor<RetType>, Tensor<unsigned>> graphene::cf::Time(
    std::function<Tensor<RetType>()> body, size_t tile) {
  DebugInfo di("cf");
  std::optional<Tensor<RetType>> result;
  poplar::program::Sequence seq;
  {
    Context context(seq);
    result = body();
  }

  // The first element of the tensor is the lower 32 bits and the second element
  // is the upper 32 bits.
  auto tensor = poplar::cycleCount(Context::graph(), seq, tile,
                                   poplar::SyncType::GLOBAL, di);

  Context::program().add(seq);

  return {*result, Tensor<unsigned>(tensor.slice(0, 1))};
}

// Explicit instantiations
#define INSTANTIATE_TIME_AND_RETURN(TYPE)             \
  template std::tuple<Tensor<TYPE>, Tensor<unsigned>> \
      graphene::cf::Time<TYPE>(std::function<Tensor<TYPE>()>, size_t);

INSTANTIATE_TIME_AND_RETURN(float)