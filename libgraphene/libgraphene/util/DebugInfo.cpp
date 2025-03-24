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