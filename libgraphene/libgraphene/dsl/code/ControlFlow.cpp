#include <spdlog/spdlog.h>

#include <cstdio>
#include <poplar/CodeletFileType.hpp>
#include <poplar/GraphElements.hpp>

#include "CodeGen.hpp"
#include "Operators.hpp"
#include "Value.hpp"

namespace graphene::codedsl {

void Return(Value value) {
  CodeGen::emitCode("return ");
  value.emitValue();
  CodeGen::emitEndStatement();
}

void Puts(std::string str) { CodeGen::emitCode("puts(\"" + str + "\");\n"); }

void If(Value cond, std::function<void()> thenDo,
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

void While(Value cond, std::function<void()> body) {
  if (cond.type() != BoolType::get()) {
    throw std::runtime_error("Condition must be of type bool");
  }
  CodeGen::emitCode("while (");
  cond.emitValue();
  CodeGen::emitCode(") {\n");
  body();
  CodeGen::emitCode("}\n");
}

void For(Value start, Value end, Value step, std::function<void(Value)> body,
         bool reverse = false, TypeRef iteratorType = Type::INT32) {
  CodeGen::emitCode("for (");
  Variable i(iteratorType, start);
  CodeGen::emitStatement((i < end).expr());
  i.assign(i + step, false);
  CodeGen::emitCode(") {\n");
  body(i);
  CodeGen::emitCode("}\n");
}

}  // namespace graphene::codedsl