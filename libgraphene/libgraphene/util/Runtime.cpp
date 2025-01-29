#include "libgraphene/util/Runtime.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <stdlib.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <poplar/OptionFlags.hpp>
#include <poplar/RuntimeOptions.hpp>
#include <stdexcept>

#include "libgraphene/codelet/Codelet.hpp"
#include "libgraphene/common/Concepts.hpp"
#include "libgraphene/common/Traits.hpp"
#include "libgraphene/util/Context.hpp"
#include "libgraphene/util/Tracepoint.hpp"
#include "poplar/DeviceManager.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include "poplar/StreamCallback.hpp"

using namespace graphene;

Runtime *Runtime::instance_ = nullptr;

Runtime::Runtime(size_t numIPUs, std::filesystem::path expressionStorageDir)
    : expressionStorageDir_(expressionStorageDir) {
  assert(instance_ == nullptr && "Runtime already registered");
  instance_ = this;

  // Setup the libtwofloat include directory
  twoFloatSourceDir_ = std::filesystem::path(LIBTWOFLOAT_INCLUDE_DIR);
  if (getenv("LIBTWOFLOAT_INCLUDE") != nullptr) {
    twoFloatSourceDir_ = std::filesystem::path(getenv("LIBTWOFLOAT_INCLUDE"));
    spdlog::info(
        "Setting libtwofloat include directory due to the "
        "LIBTWOFLOAT_INCLUDE environment variable to {}",
        twoFloatSourceDir_.string());
  }
  if (!std::filesystem::exists(twoFloatSourceDir_ /
                               "libtwofloat/twofloat.hpp")) {
    throw std::runtime_error(
        "Could not find libtwofloat include directory. Set it using the "
        "LIBTWOFLOAT_INCLUDE environment variable");
  }

  // Setup the expression storage directory
  if (getenv("EXPRESSION_STORAGE_DIR") != nullptr) {
    expressionStorageDir_ =
        std::filesystem::path(getenv("EXPRESSION_STORAGE_DIR"));
    spdlog::debug(
        "Overriding expression storage directory due to the "
        "EXPRESSION_STORAGE_DIR environment variable");
  }
  if (!std::filesystem::exists(expressionStorageDir_))
    if (!std::filesystem::create_directories(expressionStorageDir_))
      throw std::runtime_error(
          "Could not create expression storage directory: " +
          expressionStorageDir_.string());
  spdlog::debug("Expression storage directory: {}",
                expressionStorageDir_.string());

  // Setup expression dumping options
  if (getenv("EXPRESSION_DUMP_ASM") != nullptr) dumpExpressionAsm_ = true;
  if (getenv("EXPRESSION_DUMP_IR") != nullptr) dumpExpressionIR_ = true;

  init(numIPUs);
}
Runtime::~Runtime() { instance_ = nullptr; }

std::string Runtime::registerHandle(std::string prefix) {
  static int i = 0;
  std::string handle = prefix + "_" + std::to_string(i++);
  return handle;
}

void Runtime::copyToRemoteBuffers(poplar::Engine &engine) {
  for (auto &registration : copiesToRemoteBuffers_) {
    try {
      // A poplar_error is thrown if the remote buffer is never used, a
      // std::out_of_range is thrown if the remote buffer is optimized away.
      engine.copyToRemoteBuffer(registration.data, registration.buffer.handle(),
                                registration.repeatIndex);

    } catch (std::out_of_range &e) {
      // std::cout << "Remote bufer " << registration.buffer.handle()
      //           << " not found. Maybe its not used and was optimized away."
      //           << std::endl;
    }
  }
};

void Runtime::copyFromRemoteBuffers(poplar::Engine &engine) {
  for (auto &registration : copiesFromRemoteBuffers_) {
    // Use the ArrayRef<T> overload instead of the void* overload to allow
    // specifying the number of elements. Requires a type switch to get the
    // correct compile time type.
    typeSwitch(
        registration.type->poplarEquivalentType(), [&]<PoplarNativeType T>() {
          gccs::ArrayRef<T> data(reinterpret_cast<T *>(registration.data),
                                 registration.numElements);
          engine.copyFromRemoteBuffer<T>(registration.buffer.handle(), data,
                                         registration.repeatIndex);
        });
  }
};

void Runtime::connectStreams(poplar::Engine &engine) {
  for (auto &registration : streams_) {
    engine.connectStream(registration.buffer.handle(), registration.data);
  }
  for (auto &registration : streamCallbacks_) {
    poplar::StreamCallbackHandle handle(registration.callback);
    engine.connectStreamToCallback(registration.buffer.handle(),
                                   registration.callback);
  }
}

void Runtime::connectHostFunctions(poplar::Engine &engine) {
  for (auto &registration : hostFunctions_) {
    engine.connectHostFunction(registration.handle, 0,
                               std::move(registration.callback));
  }
}

void Runtime::enableProfiling(std::filesystem::path directory,
                              bool enableDeviceInstrumentation,
                              bool enableHostInstrumentation) {
  if (enableDeviceInstrumentation) {
    poplar::OptionFlags options;
    if (getenv("POPLAR_ENGINE_OPTIONS") != nullptr) {
      poplar::readJSONFromEnv("POPLAR_ENGINE_OPTIONS", options);
    }
    options.set("autoReport.all", "true");
    options.set("autoReport.outputDebugInfo", "true");
    options.set("autoReport.directory", directory.string());
    // options.set("debug.computeInstrumentationLevel", "ipu");

    // External exchange instrumentation takes a significant amount of memory
    // and may even lead to failure of memory allocation, especially when
    // instrumenting halo exchanges.
    options.set("debug.instrumentExternalExchange", "false");

    std::stringstream ss;
    ss << options;
    std::string s = ss.str();
    setenv("POPLAR_ENGINE_OPTIONS", s.data(), true);
  }

  if (enableHostInstrumentation) {
    poplar::OptionFlags options;
    if (getenv("PVTI_OPTIONS") != nullptr) {
      poplar::readJSONFromEnv("PVTI_OPTIONS", options);
    }
    options.set("enable", "true");
    options.set("directory", directory.string());

    std::stringstream ss;
    ss << options;
    std::string s = ss.str();
    setenv("PVTI_OPTIONS", s.data(), true);
  }
}

void Runtime::init(size_t numIPUs) {
  GRAPHENE_TRACEPOINT();

  // The DeviceManager is used to discover IPU devices
  auto manager = poplar::DeviceManager::createDeviceManager();

  // Attempt to attach to the requested number of IPUs
  auto devices = manager.getDevices(poplar::TargetType::IPU, numIPUs);
  spdlog::info("Trying to attach to {} IPU(s)", numIPUs);
  auto it =
      std::find_if(devices.begin(), devices.end(),
                   [](poplar::Device &device) { return device.attach(); });

  if (it == devices.end()) {
    spdlog::error("Error attaching to IPU");
    std::exit(1);
  }

  device_ = std::move(*it);
  spdlog::info("Attached to IPU {}", device_.getId());

  // Create the expression storage directory if it does not exist
  if (!std::filesystem::exists(expressionStorageDir_))
    std::filesystem::create_directories(expressionStorageDir_);

  // Target target = Target::createIPUTarget("IPU-POD16-DA");
  poplar::Target target = device_.getTarget();
  this->graph_ = poplar::Graph(target);
  this->preludeProgram_ = poplar::program::Sequence();
  this->mainProgram_ = poplar::program::Sequence();

  this->context_ = std::make_unique<Context>(graph_, mainProgram_);

  Context::setPreludeProgram(preludeProgram_);
  graphene::addCodelets(graph_, preludeProgram_);
}

void Runtime::loadAndRunEngine(poplar::Engine &engine) {
  GRAPHENE_TRACEPOINT();
  spdlog::stopwatch stopwatch;

  spdlog::info("Loading engine");
  engine.load(device_);

  copyToRemoteBuffers(engine);
  connectStreams(engine);
  connectHostFunctions(engine);
  executionStartTime_ = engine.getTimeStamp();

  engine_ = &engine;
  engine.run();
  engine_ = nullptr;

  copyFromRemoteBuffers(engine);

  spdlog::debug("Finished execution in {} seconds", stopwatch);
}

poplar::Engine Runtime::compileGraph(
    std::function<void(int, int)> progressCallback) {
  GRAPHENE_TRACEPOINT();

  spdlog::info("Compiling graph");
  spdlog::stopwatch stopwatch;
  poplar::OptionFlags options;
  // Set runtime specific options here
  // options.set("opt.internalExchangeOptimisationTarget", "memory");

  poplar::program::Sequence fullProgram_({preludeProgram_, mainProgram_});

  // std::ofstream graphProgramDump("graphProgram.json");
  // poplar::program::dumpProgram(graph_, fullProgram_, graphProgramDump);
  // graphProgramDump.close();

  poplar::Executable exec =
      poplar::compileGraph(graph_, {fullProgram_}, options,
                           [&progressCallback](int progress, int total) {
                             std::cout << "\rCompiling graph: " << progress
                                       << " of " << total << std::flush;
                             if (progressCallback)
                               progressCallback(progress, total);
                           });
  std::cout << "\r";

  spdlog::debug("Graph compilation took {} seconds", stopwatch);
  return poplar::Engine(exec, options);
}

std::string Runtime::getCurrentExecutionTime() const {
  if (!engine_)
    throw std::runtime_error(
        "This function is only available during execution");
  auto currentTime = engine_->getTimeStamp();
  return engine_->reportTiming(executionStartTime_, currentTime);
}