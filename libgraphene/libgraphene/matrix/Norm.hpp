#pragma once

#include <string>
namespace graphene {
enum class VectorNorm { L1, L2, LINF, None };

/**
 * Parses a string representation of a vector norm and returns \ref VectorNorm
 * enum value.
 *
 * @param norm The string representation of the vector norm.
 * @return The \ref VectorNorm enum value corresponding to the given string
 * representation.
 */
VectorNorm parseVectorNorm(std::string const& norm);

/**
 * Converts a VectorNorm enum value to its string representation.
 *
 * @param norm The VectorNorm value to convert.
 * @return The string representation of the VectorNorm value.
 */
std::string normToString(VectorNorm norm);

}  // namespace graphene