#include "libgraphene/dsl/tensor/ControlFlow.hpp"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <poplar/CycleCount.hpp>
#include <poplar/Program.hpp>
#include <poplar/SyncType.hpp>

#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/DebugInfo.hpp"

using namespace graphene;

void graphene::cf::If(Expression condition, std::function<void()> trueBody,
                      std::function<void()> falseBody) {
  if (condition.numElements() != 1)
    throw std::runtime_error("Condition must be a scalar");
  if (condition.type() != Type::BOOL)
    throw std::runtime_error("Condition must be a boolean expression");

  auto &program = Context::program();
  DebugInfo di("cf");

  Tensor predicate = condition;

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

void graphene::cf::While(Expression condition, std::function<void()> body) {
  if (condition.numElements() != 1)
    throw std::runtime_error("Condition must be a scalar");
  if (condition.type() != Type::BOOL)
    throw std::runtime_error("Condition must be a boolean expression");

  auto &program = Context::program();
  DebugInfo di("cf");

  Tensor predicate = Tensor::uninitialized(Type::BOOL);

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

Tensor graphene::cf::Time(std::function<void()> body, size_t tile) {
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

  return Tensor::fromPoplar(tensor.slice(0, 1), Type::UINT32);
}

std::tuple<Tensor, Tensor> graphene::cf::Time(std::function<Tensor()> body,
                                              size_t tile) {
  DebugInfo di("cf");
  std::optional<Tensor> result;
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

  return {*result, Tensor::fromPoplar(tensor.slice(0, 1), Type::UINT32)};
}