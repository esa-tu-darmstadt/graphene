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

#include "libgraphene/util/Context.hpp"

std::deque<poplar::Graph *> graphene::Context::graphs_ =
    std::deque<poplar::Graph *>();
std::deque<poplar::program::Sequence *> graphene::Context::programs_ =
    std::deque<poplar::program::Sequence *>();
poplar::program::Sequence *graphene::Context::preludeProgram_;
std::optional<poplar::ComputeSet> graphene::Context::parallelComputeSet_;