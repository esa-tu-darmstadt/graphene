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

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <poplar/PrintTensor.hpp>

#include "CLI/CLI.hpp"
#include "libgraphene/dsl/tensor/RemoteTensor.hpp"
#include "libgraphene/dsl/tensor/Tensor.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"
#include "libgraphene/util/Runtime.hpp"

using namespace graphene;
using namespace matrix;

std::tuple<size_t, size_t, size_t> parsePoissonConfig(std::string config) {
  std::vector<std::string> parts;
  // Split the string by commas
  size_t pos = 0;
  while ((pos = config.find(",")) != std::string::npos) {
    parts.push_back(config.substr(0, pos));
    config.erase(0, pos + 1);
  }
  parts.push_back(config);

  if (parts.size() != 3) {
    throw std::runtime_error(
        "Invalid Poisson matrix config. Expected format: nx,ny,nz");
  }

  return std::make_tuple(std::stoi(parts[0]), std::stoi(parts[1]),
                         std::stoi(parts[2]));
}

void load_vector(Tensor& x, const nlohmann::json& configField, Matrix& A,
                 bool withHalo, std::string name) {
  if (configField.is_number_float()) {
    x = configField.get<float>();
  } else if (configField.is_string()) {
    HostTensor x_host = A.tileLayout().loadVectorFromFile(
        Type::FLOAT32, configField.get<std::string>(), withHalo, name);
    x = x_host.copyToRemote().copyToTile();
  } else {
    throw std::runtime_error(fmt::format(
        "Invalid config value for vector {}: {}", name, configField.dump()));
  }
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);

  struct config_t {
    std::string configPath;
    size_t numTiles;
    std::string matrixPath;
    std::string poissonConfig;
    std::string profileDirectory;
    bool emulate = false;
  } cliConfig;

  // Add command line arguments
  CLI::App app{"Graphene Benchmark Tool"};
  app.add_option("config", cliConfig.configPath, "Path to config (json) file")
      ->required()
      ->check(CLI::ExistingFile);

  app.add_option("-t,--tiles", cliConfig.numTiles, "Number of tiles")
      ->required();

  app.add_option("-e,--emulate", cliConfig.emulate,
                 "Emulate on CPU instead of running on IPUs (default: false)");

  // Either a matrix file or a poisson matrix config must be provided
  auto matrixGroup = app.add_option_group("Matrix source");
  matrixGroup->require_option(1);
  matrixGroup
      ->add_option("-m,--matrix", cliConfig.matrixPath, "Path to matrix file")
      ->check(CLI::ExistingFile);
  std::string poissonConfig;
  matrixGroup->add_option("-p,--poisson", cliConfig.poissonConfig,
                          "Poisson matrix config. Format: nx,ny,nz");

  app.add_option(
         "-d,--profile", cliConfig.profileDirectory,
         "Directory to store profiling data. If not provided, profiling "
         "is disabled")
      ->check(CLI::ExistingDirectory);

  // Parse the command line arguments
  CLI11_PARSE(app, argc, argv);

  // Load the config file
  std::ifstream configFile(cliConfig.configPath);
  nlohmann::json config =
      nlohmann::json::parse(configFile, nullptr, true, true);
  configFile.close();

  // Constructs some file paths
  std::filesystem::path configPath = cliConfig.configPath;

  // Calculate number of IPUs
  size_t numTiles = cliConfig.numTiles;
  size_t numIPUs = (numTiles - 1) / 1472 + 1;
  // Round up to the nearest power of 2
  numIPUs = 1 << static_cast<size_t>(std::ceil(std::log2(numIPUs)));
  spdlog::info("Using {} tiles on {} IPUs", numTiles, numIPUs);

  // Enable profiling if requested
  if (!cliConfig.profileDirectory.empty()) {
    std::filesystem::path profileDirectory = cliConfig.profileDirectory;
    spdlog::info("Storing profiling data in {}", profileDirectory.string());
    Runtime::enableProfiling(profileDirectory);
  }

  // Initialize runtime
  Runtime runtime(numIPUs, cliConfig.emulate);

  spdlog::info("Building data flow graph");

  // Load the matrix and the vectors
  host::HostMatrix hostA;
  if (!cliConfig.matrixPath.empty()) {
    hostA =
        host::loadMatrixFromFile(Type::FLOAT32, cliConfig.matrixPath, numTiles);
  } else if (!cliConfig.poissonConfig.empty()) {
    auto [nx, ny, nz] = parsePoissonConfig(cliConfig.poissonConfig);
    hostA = host::generate3DPoissonMatrix(Type::FLOAT32, nx, ny, nz, numTiles);

  } else {
    throw std::runtime_error(
        "No matrix source provided. Please provide either "
        "a matrix file or a Poisson matrix config.");
  }
  Matrix A = hostA.copyToTile();

  Tensor x = A.createUninitializedVector(Type::FLOAT32, true);
  Tensor b = A.createUninitializedVector(Type::FLOAT32, false);

  // Initialize b
  if (config.contains("b"))
    b = config["b"].get<float>();
  else if (config.contains("x")) {
    x = config["x"].get<float>();
    b = A * x;
  } else
    throw std::runtime_error("Either b or x must be provided in the config");

  // Initialize x
  x = config.value<float>("x0", 0.0f);

  std::string benchmark = config["benchmark"];
  if (benchmark == "solve") {
    b.print("b");
    auto solverConfig = solver::Configuration::fromJSON(config["solver"]);
    A.solve(x, b, solverConfig);
    x.print("x final");
  } else if (benchmark == "spmv") {
    // Perform a sparse matrix-vector multiplication
    (void)(A * x);
  } else {
    throw std::runtime_error(fmt::format("Unknown benchmark: {}", benchmark));
  }

  poplar::Engine engine = runtime.compileGraph();
  runtime.loadAndRunEngine(engine);

  spdlog::info("Done!");
  return 0;
}