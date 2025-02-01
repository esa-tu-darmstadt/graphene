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

#include <cstdint>
#include <pvti/pvti.hpp>

namespace graphene {

// Wrapper class of the pvti Tracepoint for Poplar

// The POPLAR_TRACEPOINT macro uses the __PRETTY_FUNCTION__ which
// returns a full method name. We use the constexpr function to
// remove the return type and arguments.

// This really just wraps pvti::Tracepoint with an additional
// std::string_view constructor which is not provided because it is
// not c++11.
class Tracepoint : public pvti::Tracepoint {
 public:
  Tracepoint(pvti::TraceChannel *channel, const std::string traceLabel)
      : pvti::Tracepoint(channel, traceLabel) {}

  Tracepoint(pvti::TraceChannel *channel, const std::string_view traceLabel,
             pvti::Metadata m)
      : pvti::Tracepoint(channel, traceLabel.begin(), traceLabel.length(), &m) {
  }

  Tracepoint(pvti::TraceChannel *channel, const char *traceLabel)
      : pvti::Tracepoint(channel, traceLabel) {}

  Tracepoint(pvti::TraceChannel *channel, const std::string_view t)
      : pvti::Tracepoint(channel, t.begin(), t.length()) {}

  ~Tracepoint() = default;
};

extern pvti::TraceChannel traceLibFvm;

constexpr std::string_view formatPrettyFunction(std::string_view s) {
  // The following function does not work on all methods as shown below.
  //
  // struct A {
  //    operator int(); // implicit conversion operator: yields `int`
  //    int operator()(); // function call operator:  yields `operator`
  //    void (*)() foo(); // function pointer return type
  // };

  // Find the namespace(s)::class::method substring
  // First locate the start of the arguments
  auto j = s.find_first_of("(");

  // In the case there is no ( in the name set j to the end of the string
  if (j == std::string_view::npos) {
    j = s.length();
  }

  // Second find the last space before the arguments
  // PRETTY_FUNCTION can return "virtual void ...."
  auto i = s.find_last_of(" ", j);

  // If i == npos (-1) then i+1 will equal 0
  return s.substr(i + 1, j - (i + 1));
}

#define GRAPHENE_TRACEPOINT() \
  graphene::Tracepoint pt__(  \
      &graphene::traceLibFvm, \
      ::graphene::formatPrettyFunction(__PRETTY_FUNCTION__))

}  // namespace graphene