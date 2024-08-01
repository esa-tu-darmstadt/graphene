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