#include "libgraphene/util/DebugInfo.hpp"

#include <poplar/DebugContext.hpp>

using namespace graphene;

std::stack<DebugInfo *> DebugInfo::debugInfoStack;

DebugInfo::DebugInfo(const std::string &api,
                     const std::vector<poputil::ArgType> &args,
                     poplar::SourceLocation loc)
    : poplar::DebugInfo(DebugInfo::debugContext(getPathName(api, loc)),
                        "graphene") {
  setValue("api", api);
  add("args", args);
  debugInfoStack.push(this);
}

void DebugInfo::add(std::string name,
                    const std::vector<poputil::ArgType> &args) {
  if (args.size() > 0) {
    poplar::ProfileValue::Map argsPV;
    for (auto &a : args) {
      argsPV.insert({a.n, a.pv});
    }
    setValue(std::move(name), argsPV);
  }
}

void DebugInfo::add(std::string name, poplar::ProfileValue pv) {
  setValue(std::move(name), std::move(pv));
}

void DebugInfo::addOutputs(const std::vector<poputil::ArgType> &outputs) {
  add("outputs", outputs);
}

void DebugInfo::addOutput(const poplar::Tensor &output) {
  setValue("output", poputil::toProfileValue(output));
}