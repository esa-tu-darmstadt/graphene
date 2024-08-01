#pragma once

#include "libgraphene/dsl/Value.hpp"

namespace graphene {
namespace cf {
void If(Expression<bool> condition, std::function<void()> trueBody,
        std::function<void()> falseBody = {});

void While(Expression<bool> condition, std::function<void()> body);

void Repeat(int count, std::function<void()> body);
}  // namespace cf
}  // namespace graphene