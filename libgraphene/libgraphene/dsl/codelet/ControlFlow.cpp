#include <spdlog/spdlog.h>

#include <cstdio>
#include <poplar/CodeletFileType.hpp>
#include <poplar/GraphElements.hpp>

#include "CodeGen.hpp"
#include "Value.hpp"

namespace graphene::codelet::dsl {

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
  CodeGen::emitCode(") {");
  thenDo();
  CodeGen::emitCode("}");
  if (elseDo) {
    CodeGen::emitCode(" else {");
    elseDo();
    CodeGen::emitCode("}");
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

}  // namespace graphene::codelet::dsl