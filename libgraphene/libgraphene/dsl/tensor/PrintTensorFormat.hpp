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

#include <optional>
#include <stdexcept>
namespace graphene {
struct PrintTensorFormat {
  /// Floating point format to use when printing a tensor
  enum class FloatFormat {
    Auto = 0,       /* Automatically determine the format */
    Fixed = 1,      /* Use fixed point e.g. -100.00 */
    Scientific = 2, /* Use scientific notation e.g. -1.123e+10 */
    HexFloat = 3,   /* Use hexadecimal floating point e.g. 0x1.234p+10 */
  };

  /// Constructor. Initializes the print format with the given parameters.
  /// @param summariseThreshold If the number of elements in a dimension exceeds
  /// this threshold, only the first \p edgeItems and last \p edgeItems elements
  /// are printed for that dimension.
  /// @param edgeItems The number of elements to print at the beginning and end
  /// of each dimension, if a dimension exceeds the \p summariseThreshold. If
  /// not provided, defaults to \p summariseThreshold / 2, resulting in the
  /// maximum number of elements printed for a dimension being \p
  /// summariseThreshold.
  /// @param precision The precision
  PrintTensorFormat(unsigned summariseThreshold = 100, unsigned precision = 8,
                    FloatFormat floatFormat = FloatFormat::Auto,
                    std::optional<unsigned> edgeItems = {})
      : summariseThreshold(summariseThreshold),
        edgeItems(edgeItems ? *edgeItems : summariseThreshold / 2),
        precision(precision),
        floatFormat(floatFormat) {
    if (edgeItems && (*edgeItems) * 2 > summariseThreshold) {
      throw std::invalid_argument(
          "edgeItems must be less than half of summariseThreshold");
    }
  }

  unsigned summariseThreshold;
  unsigned edgeItems;
  unsigned precision;
  FloatFormat floatFormat;
};

}  // namespace graphene