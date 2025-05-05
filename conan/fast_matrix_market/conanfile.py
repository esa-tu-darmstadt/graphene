from conan import ConanFile
from conan.tools.files   import copy, get, export_conandata_patches, rmdir
from conan.tools.cmake   import CMake, CMakeDeps, CMakeToolchain, cmake_layout
import os

class FastMatrixMarketConan(ConanFile):
    name             = "fast_matrix_market"
    version          = "1.7.6" 
    license          = "BSD-2-Clause"      # LICENSE.txt :contentReference[oaicite:0]{index=0}
    url              = "https://github.com/alugowski/fast_matrix_market"
    homepage         = url
    description      = "High-performance Matrix Market read/write library"
    topics           = ("matrix-market", "sparse-matrix", "io")
    

    settings         = "os", "arch", "compiler", "build_type"
    options          = {
        "shared"          : [True, False],          # for ryu sub-library
        "with_fast_float" : [True, False],
        "with_dragonbox"  : [True, False],
        "with_ryu"        : [True, False],
    }
    default_options  = {
        "shared"          : False,
        "with_fast_float" : True,
        "with_dragonbox"  : True,
        "with_ryu"        : True,
    }

    package_type     = "library"   # treat as a normal package, not header-only

    # ──────────────────────────────────────────────────────────────────────────
    def layout(self):
        cmake_layout(self)

    # ──────────────────────────────────────────────────────────────────────────
    def source(self):
        # Pull the exact tag requested at `conan create` time
        get(self,
            url=f"{self.homepage}/archive/refs/tags/v{self.version}.tar.gz",
            strip_root=True)

    # ──────────────────────────────────────────────────────────────────────────
    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["FMM_USE_FAST_FLOAT"]        = self.options.with_fast_float
        tc.variables["FMM_USE_DRAGONBOX"]         = self.options.with_dragonbox
        tc.variables["FMM_USE_RYU"]               = self.options.with_ryu
        tc.variables["FAST_MATRIX_MARKET_TEST"]   = False
        tc.variables["FAST_MATRIX_MARKET_BENCH"]  = False
        tc.generate()
        CMakeDeps(self).generate()

    # ──────────────────────────────────────────────────────────────────────────
    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()          # builds ryu / dragonbox objects when enabled

    # ──────────────────────────────────────────────────────────────────────────
    def package(self):
      cmake = CMake(self)
      cmake.install()
      # Upstream has no install() step for the top-level target, so stage files manually
      copy(self, "LICENSE.txt",
            dst=os.path.join(self.package_folder, "licenses"),
            src=self.source_folder)

      copy(self, "include/*",
            dst=self.package_folder,
            src=self.source_folder)
      
      # No cmake install → clean any empty cmake folders left by sub-deps
      rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    # ──────────────────────────────────────────────────────────────────────────
    def package_info(self):
        if self.options.with_dragonbox:
            self.cpp_info.libs.append("dragonbox_to_chars")
        if self.options.with_ryu:
            self.cpp_info.libs.append("ryu")
