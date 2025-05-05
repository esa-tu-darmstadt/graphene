from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import apply_conandata_patches, copy, export_conandata_patches, get, rm, rmdir
from conan.tools.microsoft import is_msvc, is_msvc_static_runtime
import os


required_conan_version = ">=2.0.9"


class GrapheneConan(ConanFile):
    name = "graphene"
    version = "0.1.0"
    license = "AGPL-3.0"
    author = "ESA TU Darmstadt"
    description = "Linear algebra framework for GraphCore IPUs"
    topics = ("linear-algebra", "sparse-matrix", "ipu", "graphcore", "hpc")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "ipu_arch_ipu1": [True, False],
        "ipu_arch_ipu2": [True, False],
        "ipu_arch_ipu21": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "ipu_arch_ipu1": False,
        "ipu_arch_ipu2": True,
        "ipu_arch_ipu21": True,
    }
    generators = "CMakeDeps", "CMakeToolchain"
    implements = ["auto_shared_fpic"]


    def export_sources(self):
        copy(self, "CMakeLists.txt", self.recipe_folder, self.export_sources_folder)
        copy(self, "cmake/*",          self.recipe_folder, self.export_sources_folder)
        copy(self, "applications/*",          self.recipe_folder, self.export_sources_folder)
        copy(self, "libgraphene/*",          self.recipe_folder, self.export_sources_folder)

    def layout(self):
        cmake_layout(self, src_folder=".")
        self.cpp.source.includedirs = ["libgraphene"]
        self.cpp.build.libdirs = [
            "libgraphene/libgraphene/codelet",
            "libgraphene/libgraphene/common",
            "libgraphene/libgraphene/dsl/common",
            "libgraphene/libgraphene/dsl/tensor",
            "libgraphene/libgraphene/dsl/code",
            "libgraphene/libgraphene/matrix",
            "libgraphene/libgraphene/util"
        ]

    def requirements(self):
        self.requires("gtest/1.16.0")
        self.requires("nlohmann_json/3.9.1") # Same version as in Poplar SDK
        self.requires("cli11/2.3.2")
        self.requires("spdlog/1.15.1")
        self.requires("fmt/11.1.3")
        self.requires("twofloat/0.2.0", transitive_headers=True)
        self.requires("metis/5.2.1")
        self.requires("fast_matrix_market/1.7.6")

    def build(self):
        # Build list of enabled IPU architectures from options
        enabled_ipu_archs = []
        if self.options.ipu_arch_ipu1:
            enabled_ipu_archs.append("ipu1")
        if self.options.ipu_arch_ipu2:
            enabled_ipu_archs.append("ipu2")
        if self.options.ipu_arch_ipu21:
            enabled_ipu_archs.append("ipu21")
        
        # Make sure at least one architecture is enabled
        if not enabled_ipu_archs:
            self.output.warning("No IPU architectures enabled. Defaulting to ipu2.")
            enabled_ipu_archs = ["ipu2"]
        
        # Join the architectures with commas for the CMake variable
        ipu_archs_str = ";".join(enabled_ipu_archs)
    
        cmake = CMake(self)
        # Pass flag to enable specific IPU architectures based on options
        cmake.configure(variables={
            "POPLIBS_ENABLED_IPU_ARCH_NAMES": ipu_archs_str
        })
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()


    def package_info(self):
        self.cpp_info.requires = ["cli11::cli11", "gtest::gtest"]
        self.cpp_info.system_libs = [ "poplar", "poputil", "tbb"]
        
        # Graphene (main interface library)
        self.cpp_info.components["graphene"].libs = []
        self.cpp_info.components["graphene"].requires = [
            "common",
            "util",
            "tensor-dsl",
            "code-dsl",
            "matrix"
        ]
        
        # GrapheneCodelet
        self.cpp_info.components["codelet"].libs = ["GrapheneCodelet"]
        # Codelet only depends on external Poplar libraries and GrapheneHeaders
        
        # GrapheneCommon
        self.cpp_info.components["common"].libs = ["GrapheneCommon"]
        self.cpp_info.components["common"].requires = [
            "spdlog::spdlog",
            "twofloat::twofloat"
        ]
        
        # GrapheneUtil
        self.cpp_info.components["util"].libs = ["GrapheneUtil"]
        self.cpp_info.components["util"].requires = [
            "common",
            "spdlog::spdlog",
            "codelet"
        ]
        
        # GrapheneDSLCommon
        self.cpp_info.components["dsl-common"].libs = ["GrapheneDSLCommon"]
        self.cpp_info.components["dsl-common"].requires = [
            "common",
            "spdlog::spdlog"
        ]
        
        # GrapheneCodeDSL
        self.cpp_info.components["code-dsl"].libs = ["GrapheneCodeDSL"]
        self.cpp_info.components["code-dsl"].requires = [
            "common",
            "spdlog::spdlog",
            "dsl-common",
            "util"
        ]
        
        # GrapheneTensorDSL
        self.cpp_info.components["tensor-dsl"].libs = ["GrapheneTensorDSL"]
        self.cpp_info.components["tensor-dsl"].requires = [
            "common",
            "util",
            "code-dsl",
            "dsl-common",
            "spdlog::spdlog"
        ]
        
        # GrapheneMatrix
        self.cpp_info.components["matrix"].libs = ["GrapheneMatrix"]
        self.cpp_info.components["matrix"].requires = [
            "common",
            "util",
            "tensor-dsl",
            "nlohmann_json::nlohmann_json",
            "metis::metis",
            "spdlog::spdlog",
            "fmt::fmt"
        ]