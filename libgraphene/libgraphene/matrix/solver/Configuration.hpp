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