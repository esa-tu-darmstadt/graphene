#include "libgraphene/dsl/ControlFlow.hpp"

#include <spdlog/spdlog.h>

#include <poplar/Program.hpp>

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

  Value<bool> predicate = condition;

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

  Value<bool> predicate;

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