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

#include <any>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <poplar/Graph.hpp>
#include <poplar/HostFunctionCallback.hpp>
#include <string>
#include <vector>

#include "libgraphene/common/Traits.hpp"
#include "libgraphene/common/Type.hpp"
#include "libgraphene/util/TargetType.hpp"
#include "libgraphene/util/Tracepoint.hpp"
#include "poplar/Device.hpp"
#include "poplar/Engine.hpp"
#include "poplar/Program.hpp"

namespace graphene {
class Mesh;
class Context;
class Runtime {
  void init(size_t numIPUs, bool emulate);

  struct RemoteBufferRegistration {
    TypeRef type;
    poplar::RemoteBuffer buffer;
    size_t repeatIndex;
    void *data;
    size_t numElements;
  };

  struct DataStreamRegistration {
    poplar::DataStream buffer;
    void *data;
  };

  struct DataStreamCallbackRegistration {
    poplar::DataStream buffer;
    std::function<void(void *)> callback;
  };

  struct HostFunctionRegistration {
    std::string handle;
    poplar::HostCallbackHandle callback;
  };

  std::vector<RemoteBufferRegistration> copiesToRemoteBuffers_;
  std::vector<RemoteBufferRegistration> copiesFromRemoteBuffers_;

  std::vector<DataStreamRegistration> streams_;
  std::vector<DataStreamCallbackRegistration> streamCallbacks_;
  std::vector<HostFunctionRegistration> hostFunctions_;
  std::vector<std::string> handles_;

  poplar::Device device_;
  poplar::Graph graph_;
  poplar::Engine *engine_;
  poplar::program::Sequence mainProgram_;
  poplar::program::Sequence preludeProgram_;
  std::unique_ptr<Context> context_;

  poplar::Engine::TimerTimePoint executionStartTime_;

  std::filesystem::path expressionStorageDir_;
  std::filesystem::path twoFloatSourceDir_;
  std::filesystem::path ipuThreadSyncSourceDir_;

  bool dumpExpressionAsm_ = true;
  bool dumpExpressionIR_ = false;

  TargetType targetType_;

 public:
  struct HostResource {
    virtual ~HostResource() = default;
    HostResource() = default;
    HostResource(const HostResource &) = delete;
  };

 private:
  std::unordered_set<std::shared_ptr<HostResource>> resources_;

 public:
  // Singleton
  static Runtime *instance_;
  static Runtime &instance() {
    assert(instance_ != nullptr && "No runtime registered");
    return *instance_;
  }

  Runtime() = delete;
  explicit Runtime(
      size_t numIPUs, bool emulate = false,
      std::filesystem::path expressionStorageDir = ".cache/expressions");
  ~Runtime();

  /// \brief Sets poplar  and PVTI environment variables to enable profiling.
  /// This must be invoked before any other interaction with poplar or this
  /// library. Place it at the beginning of your main function.
  static void enableProfiling(std::filesystem::path directory = "./profiling/",
                              bool enableDeviceInstrumentation = true,
                              bool enableHostInstrumentation = true);

  /// \brief Finds and registers an unused handle with a given prefix
  std::string registerHandle(std::string prefix);

  void registerStream(poplar::DataStream fifo, void *data) {
    streams_.push_back({fifo, data});
  }

  /// \brief Register a stream callback. The callback will be called when the
  /// stream is to be read or was written to. The callback must copy the data
  /// to or from the provided buffer!
  void registerStreamCallback(poplar::DataStream fifo,
                              std::function<void(void *)> callback) {
    streamCallbacks_.push_back({fifo, callback});
  }

  /// \brief Register a copy to a remote buffer
  /// \param handle The handle of the remote buffer
  /// \param buffer The buffer to copy from
  void registerCopyToRemoteBuffer(TypeRef type, poplar::RemoteBuffer buffer,
                                  const void *data, size_t numElements,
                                  size_t repeatIndex = 0) {
    assert(data != nullptr && "Data is null");
    copiesToRemoteBuffers_.push_back(
        {type, buffer, repeatIndex, const_cast<void *>(data), numElements});
  }

  /// \brief Register a copy from a remote buffer
  /// \param handle The handle of the remote buffer
  /// \param buffer The buffer to copy to
  void registerCopyFromRemoteBuffer(TypeRef type, poplar::RemoteBuffer buffer,
                                    void *data, size_t numElements,
                                    size_t repeatIndex = 0) {
    assert(data != nullptr && "Data is null");
    copiesFromRemoteBuffers_.push_back(
        {type, buffer, repeatIndex, data, numElements});
  }

  /// \brief Register a host function
  /// \param handle The handle of the host function
  /// \param callback The callback to call
  void registerHostFunction(std::string handle,
                            poplar::HostCallbackHandle callback) {
    hostFunctions_.push_back({handle, std::move(callback)});
  }

  /// \brief Loads the program to the device, connects all streams and host
  /// functions and runs the engine
  void loadAndRunEngine(poplar::Engine &engine);

  /// \brief Compiles a graph
  /// \param graph The graph to compile
  /// \param program The program to compile
  /// \return The compiled engine
  poplar::Engine compileGraph(
      std::function<void(int, int)> progressCallback = {});

  /// \brief Copy all registered buffers to remote buffers
  /// \param engine The engine to use
  void copyToRemoteBuffers(poplar::Engine &engine);

  /// \brief Copy all registered buffers from remote buffers
  /// \param engine The engine to use
  void copyFromRemoteBuffers(poplar::Engine &engine);

  /// \brief Connect all registered streams
  /// \param engine The engine to use
  void connectStreams(poplar::Engine &engine);

  /// \brief Connects all host functions
  /// \param engine The engine to use
  void connectHostFunctions(poplar::Engine &engine);

  std::filesystem::path getExpressionStorageDir() const {
    return expressionStorageDir_;
  }

  enum class RuntimeLib { TwoFloat, IpuThreadSync };
  /// \brief Get the path to the include directory of the given runtime library
  std::filesystem::path getRuntimeLibIncludeDir(RuntimeLib lib) const;

  std::string getCurrentExecutionTime() const;

  template <typename T, typename... Args>
  std::shared_ptr<T> createResource(Args &&...args) {
    auto resource = std::make_shared<T>(std::forward<Args>(args)...);
    resources_.emplace(resource);
    return resource;
  }

  template <typename T>
  bool freeResource(std::shared_ptr<T> resource) {
    return resources_.erase(resource) != 0;
  }

  /// True if the user requested to dump generated expressions as assembly
  bool dumpExpressionAsm() const { return dumpExpressionAsm_; }

  /// True if the user requested to dump generated expressions as IR
  bool dumpExpressionIR() const { return dumpExpressionIR_; }

  TargetType getTargetType() const { return targetType_; }
};
}  // namespace graphene