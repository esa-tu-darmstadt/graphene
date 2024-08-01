#pragma once

#include <string>
namespace graphene {
enum class VectorNorm { L1, L2, LINF, None };

VectorNorm parseVectorNorm(std::string const& norm);
std::string normToString(VectorNorm norm);

}  // namespace graphene