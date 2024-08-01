# function that creates a GP library that contains ipu and cpu targets. It
# builds each source file and each target in a separate partial gp file and then
# link them all together into the final library to allow for maximal parallelism
# during build time.
#
# this function requires the following global variables to exist (all of which
# are defined in the top level CMakeLists.txt):
#   - POPLIBS_ENABLED_IPU_ARCH_NAMES
#   - POPC_EXECUTABLE
#   - POPC_FLAGS

include(GNUInstallDirs)

function(add_gp_library)
  cmake_parse_arguments(CODELET "" "NAME" "ASM_SOURCES;CPP_SOURCES;HEADERS;INCLUDE_DIRS;DUMP_ASM" ${ARGN})

  set(LIST_OF_TARGETS_JSON_TMP "")
  foreach(variant ${POPLIBS_ENABLED_IPU_ARCH_NAMES})
    list(APPEND LIST_OF_TARGETS_JSON_TMP "\"${variant}\"")
  endforeach()

  # Break all targets into separate JSON lists to allow build
  # them separately
  set(LIST_ALL_TARGETS_JSON "")
  foreach(TARGET IN LISTS LIST_OF_TARGETS_JSON_TMP)
    list(APPEND LIST_ALL_TARGETS_JSON "'[${TARGET}]'")
  endforeach()

  # To build all ASM sources combine POPLIBS_ENABLED_IPU_ARCH_NAMES list into
  # a string of targets. Could use JSON string here as well but for simplicity
  # use just a comma separaed string
  string(REPLACE ";" "," ASM_TARGETS "${POPLIBS_ENABLED_IPU_ARCH_NAMES}")

  # For the final step to combine all gp files it requries to JSON formated
  # string of all targets
  string(REPLACE ";" "," STRING_ALL_TARGETS_JSON_TMP "${LIST_OF_TARGETS_JSON_TMP}")
  set(STRING_ALL_TARGETS_JSON "'[${STRING_ALL_TARGETS_JSON_TMP}]'")

  # Add include directories to the compile command
  set(INCLUDE_DIRS_COMMAND "")
  foreach(DIR IN LISTS CODELET_INCLUDE_DIRS)
    list(APPEND INCLUDE_DIRS_COMMAND "-I${DIR}")
  endforeach()

  set(POPC_OPT_FLAGS "-O3")

  set(COMMAND
    ${CMAKE_COMMAND} -E env ${POPC_ENVIRONMENT}
    ${POPC_EXECUTABLE}
    ${POPC_OPT_FLAGS}
    ${POPC_FLAGS}
    -DNDEBUG #-DNDEBUG
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${INCLUDE_DIRS_COMMAND}
  )

  set(PARTIAL_OUTPUTS)
  set(ASM_OUTPUTS)

  # compile each C++ file in it's own gp file so that we don't have to rebuild
  # the entire library whenever one of those files has changed. for now we
  # add all of the headers as dependencies to all of the partial gp files. a
  # future improvement would be to only pass the required headers to each one.
  #
  # TODO: T10282 Fix dependencies with poplar's headers.
  foreach(CPP_SOURCE ${CODELET_CPP_SOURCES})
    get_filename_component(FILE ${CPP_SOURCE} NAME_WE)
    foreach(TARGET IN LISTS LIST_ALL_TARGETS_JSON)
      # Ideally we want to extract target name and add it into a filename
      # but for now use SHA1 for the file names
      string(SHA1 MAGIC_NUMBER ${TARGET})
      set(PARTIAL_GP_NAME "${CODELET_NAME}_${FILE}_${MAGIC_NUMBER}.gp")
      add_custom_command(
        OUTPUT
          ${PARTIAL_GP_NAME}
        COMMAND
          ${COMMAND}
          -o ${PARTIAL_GP_NAME}
          --target ${TARGET}
          ${CPP_SOURCE}
        DEPENDS
          ${CPP_SOURCE}
          ${CODELET_HEADERS}
          popc_bin
      )
      list(APPEND PARTIAL_OUTPUTS ${PARTIAL_GP_NAME})
      if (CODELET_DUMP_ASM)
        set(PARTIAL_ASM_NAME "${CODELET_NAME}_${FILE}_${MAGIC_NUMBER}.S")
        add_custom_command(
          OUTPUT
            ${PARTIAL_ASM_NAME}
          COMMAND
            ${COMMAND}
            -o ${PARTIAL_ASM_NAME}
            --target ${TARGET}
            --S
            ${CPP_SOURCE}
          DEPENDS
            ${CPP_SOURCE}
            ${CODELET_HEADERS}
            popc_bin
        )
        list(APPEND ASM_OUTPUTS ${PARTIAL_ASM_NAME})
      endif()
    endforeach()
  endforeach()

  
  # compile all the assembly into a separate partial gp object if assembly sources are given.
  if (CODELET_ASM_SOURCES)
    set(ASM_GP_NAME "${CODELET_NAME}_asm.gp")
    add_custom_command(
      OUTPUT
        ${ASM_GP_NAME}
      COMMAND
        ${COMMAND}
        -o ${ASM_GP_NAME}
        --target ${ASM_TARGETS}
        ${CODELET_ASM_SOURCES}
      DEPENDS
        ${CODELET_ASM_SOURCES}
        ${CODELET_HEADERS}
        popc_bin
    )
    list(APPEND PARTIAL_OUTPUTS ${ASM_GP_NAME})
  endif()

  # compile all of the partial gp files into the actual final library objects
  set(NAME "${CODELET_NAME}.gp")
  add_custom_command(
    OUTPUT
      ${NAME}
    COMMAND
      ${COMMAND}
      -o ${NAME}
      --target ${STRING_ALL_TARGETS_JSON}
      ${PARTIAL_OUTPUTS}
    DEPENDS
      ${PARTIAL_OUTPUTS}
      ${ASM_OUTPUTS}
      popc_bin
  )
  set(OUTPUTS ${NAME})

  add_custom_target(${CODELET_NAME}.ipu ALL DEPENDS ${OUTPUTS}
    SOURCES
      ${CODELET_CPP_SOURCES}
      ${CODELET_ASM_SOURCES}
      ${CODELET_HEADERS}
  )

  add_library(${CODELET_NAME} INTERFACE)
  add_dependencies(${CODELET_NAME} ${CODELET_NAME}.ipu)

  # Generate a additional dummy lib so that the compiler
  # invocations are added to the compile_commands.json file
  add_library(${CODELET_NAME}.dummy STATIC EXCLUDE_FROM_ALL ${CODELET_CPP_SOURCES})
  set_property(TARGET ${CODELET_NAME}.dummy PROPERTY INCLUDE_DIRECTORIES "")
  target_include_directories(${CODELET_NAME}.dummy PRIVATE 
    ${poplar_DIR}/../../../include
    ${poplar_DIR}/../../../lib/graphcore/include
    ${poplar_DIR}/../../../lib/graphcore/include/c++
    ${poplar_DIR}/../../../lib/clang/16.0.0/include/
    ${poplar_DIR}/../../../lib/graphcore/include
    ${poplar_DIR}/../../../lib/clang/16.0.0/include
    ${poplar_DIR}/../../../lib/graphcore/include/poplar)
  target_include_directories(${CODELET_NAME}.dummy PRIVATE ${CODELET_INCLUDE_DIRS})
  target_compile_definitions(${CODELET_NAME}.dummy PRIVATE __POPC__=1)
  target_compile_options(${CODELET_NAME}.dummy PRIVATE -Wno-ignored-attributes -nostdinc -nostdinc++)
  set_property(TARGET ${CODELET_NAME}.dummy PROPERTY CXX_STANDARD 17)
  message(STATUS "Adding codelet dummy library ${CODELET_NAME}.dummy")

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${NAME}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT ${CODELET_NAME})

endfunction()