#include "CodeGen.hpp"

namespace graphene::codedsl {
std::stringstream CodeGen::vertexStream_;
std::stack<std::stringstream> CodeGen::functionStreams_;

size_t CodeGen::functionCounter_ = 0;
size_t CodeGen::varCounter_ = 0;
}  // namespace graphene::codedsl