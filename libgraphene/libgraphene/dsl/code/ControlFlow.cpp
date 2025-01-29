#include "ControlFlow.hpp"

#include <spdlog/spdlog.h>

#include <cstdio>
#include <poplar/CodeletFileType.hpp>
#include <poplar/GraphElements.hpp>

#include "CodeGen.hpp"
#include "Operators.hpp"
#include "Value.hpp"

using namespace graphene;
using namespace graphene::codedsl;

void codedsl::Return(Value value) {
  CodeGen::emitCode("return ");
  value.emitValue();
  CodeGen::emitEndStatement();
}

void codedsl::Puts(std::string str) {
  CodeGen::emitCode("puts(\"" + str + "\");\n");
}

void codedsl::If(Value cond, std::function<void()> thenDo,
                 std::function<void()> elseDo) {
  if (cond.type() != BoolType::get()) {
    throw std::runtime_error("Condition must be of type bool");
  }
  CodeGen::emitCode("if (");
  cond.emitValue();
  CodeGen::emitCode(") {\n");
  thenDo();
  CodeGen::emitCode("}\n");
  if (elseDo) {
    CodeGen::emitCode(" else {\n");
    elseDo();
    CodeGen::emitCode("}\n");
  }
}

void codedsl::While(Value cond, std::function<void()> body) {
  if (cond.type() != BoolType::get()) {
    throw std::runtime_error("Condition must be of type bool");
  }
  CodeGen::emitCode("while (");
  cond.emitValue();
  CodeGen::emitCode(") {\n");
  body();
  CodeGen::emitCode("}\n");
}

Variable codedsl::detail::ForStart(Value start, Value end, Value step,
                                   bool reverse, TypeRef iteratorType) {
  CodeGen::emitCode("for (");
  Variable i(iteratorType, start);
  if (reverse) {
    CodeGen::emitStatement((i >= start).expr());
    i.assign(i - step, false);
  } else {
    CodeGen::emitStatement((i < end).expr());
    i.assign(i + step, false);
  }
  CodeGen::emitCode(") {\n");
  return i;
}
void codedsl::detail::ForEnd() { CodeGen::emitCode("}\n"); }

void codedsl::For(Value start, Value end, Value step,
                  std::function<void(Value)> body, TypeRef iteratorType) {
  Variable i = detail::ForStart(start, end, step, false, iteratorType);
  body(i);
  detail::ForEnd();
}

void codedsl::ForReverse(Value start, Value end, Value step,
                         std::function<void(Value)> body,
                         TypeRef iteratorType) {
  Variable i = detail::ForStart(start, end, step, true, iteratorType);
  body(i);
  detail::ForEnd();
}

void codedsl::Break() { CodeGen::emitStatement("break"); }
