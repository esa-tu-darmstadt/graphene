#include "CodeGen.hpp"

namespace graphene::codedsl {
std::stringstream CodeGen::vertexStream_;
std::optional<std::stringstream> CodeGen::functionStream_;
std::map<std::string, std::string> CodeGen::functions_;

size_t CodeGen::functionCounter_ = 0;
size_t CodeGen::varCounter_ = 0;
}  // namespace graphene::codedsl