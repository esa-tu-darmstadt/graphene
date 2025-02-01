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