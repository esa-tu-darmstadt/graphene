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

#include <memory>
#include <nlohmann/json_fwd.hpp>

namespace graphene::matrix::solver {

enum class MultiColorMode { Off, On, Auto };

struct Configuration {
 protected:
  template <typename T>
  void setFieldFromJSON(nlohmann::json const& config, std::string const& field,
                        T& value);

 public:
  virtual std::string solverName() const = 0;
  static std::shared_ptr<Configuration> fromJSON(nlohmann::json const& config);
};

}  // namespace graphene::matrix::solver