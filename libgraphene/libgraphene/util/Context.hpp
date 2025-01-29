#pragma once

#include <deque>
#include <memory>
#include <poplar/Graph.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Program.hpp>

namespace graphene {
class Context {
 private:
  static std::deque<poplar::Graph *> graphs_;
  static std::deque<poplar::program::Sequence *> programs_;
  static poplar::program::Sequence *preludeProgram_;

  bool setsGraph_ = false;
  bool setsProgram_ = false;

 public:
  Context() = default;
  Context(poplar::Graph &graph) : setsGraph_(true) {
    graphs_.push_back(&graph);
  }
  Context(poplar::program::Sequence &program) : setsProgram_(true) {
    programs_.push_back(&program);
  }
  Context(poplar::Graph &graph, poplar::program::Sequence &program)
      : setsGraph_(true), setsProgram_(true) {
    graphs_.push_back(&graph);
    programs_.push_back(&program);
  }
  ~Context() {
    if (setsGraph_) graphs_.pop_back();
    if (setsProgram_) programs_.pop_back();
  }

  bool setsGraph() const { return setsGraph_; }
  bool setsProgram() const { return setsProgram_; }

  static poplar::Graph &graph() {
    assert(graphs_.size() > 0 &&
           "No graph in context. Did you forget to create a context?");
    return *graphs_.back();
  }
  static poplar::program::Sequence &program() {
    assert(programs_.size() > 0 &&
           "No program in context. Did you forget to create a context?");
    return *programs_.back();
  }

  static size_t getNumIPUs() {
    assert(graphs_.size() > 0 &&
           "No graph in context. Did you forget to create a context?");
    return graphs_.back()->getTarget().getNumIPUs();
  }

  static size_t getNumTiles() {
    assert(graphs_.size() > 0 &&
           "No graph in context. Did you forget to create a context?");
    return graphs_.back()->getTarget().getNumTiles();
  }

  static void setPreludeProgram(poplar::program::Sequence &program) {
    preludeProgram_ = &program;
  }

  static bool isPreludeProgramSet() { return preludeProgram_ != nullptr; }

  /// Executes all operations in parallel within the lifetime of this object by
  /// adding them to the same compute set.
  class ExecuteInParallel;

  /// Executes all operations in the prelude program, i.e. before the main
  /// program, within the lifetime of this object.
  class Prelude;
};

class Context::Prelude {
  Context context;

 public:
  Prelude() : context(*Context::preludeProgram_) {
    assert(Context::isPreludeProgramSet() &&
           "No prelude program set. Did you forget to set it?");
  }
};
}  // namespace graphene