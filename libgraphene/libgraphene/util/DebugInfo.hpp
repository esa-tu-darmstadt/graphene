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

#pragma once

#include <poplar/DebugContext.hpp>
#include <poputil/DebugInfo.hpp>
#include <stack>

namespace graphene {

class DebugInfo : public poplar::DebugInfo {
  static std::stack<DebugInfo *> debugInfoStack;

  static std::string getPathName(const std::string &api,
                                 poplar::SourceLocation loc) {
    std::string pathName = api;
    if (loc.isValid()) {
      pathName += "::" + std::string(loc.getFunctionName());
    }
    return pathName;
  }

  static poplar::DebugContext debugContext(
      std::string name = "",
      poplar::SourceLocation loc = poplar::SourceLocation::Current()) {
    if (debugInfoStack.empty()) return poplar::DebugContext(name, loc);
    return poplar::DebugContext(*debugInfoStack.top(), name);
  }

 public:
  DebugInfo(const std::string &api,
            const std::vector<poputil::ArgType> &args = {},
            poplar::SourceLocation loc = poplar::SourceLocation::Current());

  DebugInfo &operator=(const DebugInfo &) = delete;
  DebugInfo(const DebugInfo &) = delete;
  ~DebugInfo() override { debugInfoStack.pop(); }

  void add(std::string name, const std::vector<poputil::ArgType> &args);
  void add(std::string name, poplar::ProfileValue pv);

  void addOutputs(const std::vector<poputil::ArgType> &outputs);

  // Convience method when there is only a single output
  void addOutput(const poplar::Tensor &output);
};

}  // namespace graphene
