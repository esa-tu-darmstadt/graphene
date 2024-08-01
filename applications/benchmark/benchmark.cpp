#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>

#include "libgraphene/dsl/RemoteValue.hpp"
#include "libgraphene/dsl/Value.hpp"
#include "libgraphene/matrix/Matrix.hpp"
#include "libgraphene/matrix/Norm.hpp"
#include "libgraphene/matrix/host/HostMatrix.hpp"
#include "libgraphene/matrix/solver/Configuration.hpp"
#include "libgraphene/util/Runtime.hpp"

using namespace graphene;
using namespace matrix;

void load_vector(Value<float>& x, const nlohmann::json& configField,
                 const std::filesystem::path& baseDirectory, Matrix<float>& A,
                 bool withHalo, std::string name) {
  if (configField.is_number_float()) {
    x = configField.get<float>();
  } else if (configField.is_string()) {
    HostValue<float> x_host = host::loadVectorFromFile<float>(
        baseDirectory / configField.get<std::string>(), A.hostMatrix(),
        withHalo, name);
    x = x_host.copyToRemote().copyToTile();
  } else {
    throw std::runtime_error(fmt::format(
        "Invalid config value for vector {}: {}", name, configField));
  }
}

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::trace);

  // Parse command line arguments
  CLI::App app{"Graphene Benchmark Tool"};
  std::string configPathString;
  app.add_option("config", configPathString, "Path to config (json) file")
      ->required();
  CLI11_PARSE(app, argc, argv);

  // Load the config file
  std::ifstream configFile(configPathString);
  nlohmann::json config =
      nlohmann::json::parse(configFile, nullptr, true, true);
  configFile.close();

  // Constructs some file paths
  std::filesystem::path configPath = configPathString;
  std::filesystem::path baseDirectory = configPath.parent_path().string();
  std::filesystem::path profileDirectory =
      baseDirectory / config["profileDirectory"];
  spdlog::info("Storing profiling data in {}", profileDirectory.string());

  // Calculate number of IPUs
  size_t numTiles = config["tiles"];
  size_t numIPUs = (numTiles - 1) / 1472 + 1;
  spdlog::info("Using {} tiles on {} IPUs", numTiles, numIPUs);

  // Initialize runtime
  Runtime::enableProfiling(profileDirectory);
  Runtime runtime(numIPUs);

  spdlog::info("Building data flow graph");

  // Load the matrix and the vectors
  Matrix<float> A = host::loadMatrixFromFile<float>(
                        baseDirectory / config["matrix"], numTiles)
                        .copyToTile();

  Value<float> x = A.createUninitializedVector<float>(true);
  Value<float> b = A.createUninitializedVector<float>(false);

  load_vector(x, config["x0"], baseDirectory, A, true, "x");
  load_vector(b, config["b"], baseDirectory, A, false, "b");

  std::string benchmark = config["benchmark"];
  if (benchmark == "solve") {
    // Solve the system with the solver specified in the config
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