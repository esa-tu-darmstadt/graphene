set_property(DIRECTORY PROPERTY INCLUDE_DIRECTORIES "")


add_gp_library(NAME GrapheneCodeletsIPU 
    CPP_SOURCES 
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/tensor/DoubleWord.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/tensor/DoublePrecision.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/matrix/crs/Residual.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/matrix/crs/SpMV.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/matrix/solvers/gauss-seidel.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/matrix/solvers/ilu/ilu-factorize.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/matrix/solvers/ilu/ilu-solve.cpp
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/twofloat/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/ipu-thread-sync/include
    DUMP_ASM True
    )