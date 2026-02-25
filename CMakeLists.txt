# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.18)
# One of the dependency when compiled in manylinux requires ASM and it cannot
# find it if we do not force it from here.
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX ASM)

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)  # Abseil needs at least C++17.

# Hide all the internal symbols.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# As we pack a wheel and we need to enforce hermeticity we prefer to build the
# static libraries. We manually control our targets but this variable will
# affect our dependencies.
set(BUILD_SHARED_LIBS OFF)

include(GNUInstallDirs)

# Override the default installation directory for CMake. This will affect only
# the dependent targets as we will specify our own installation directory later.
# This is the only way to block dependent targets that do not expose options to
# install their artefacts (we will later remove them).
# Note that we need to define the single components and not the general install
# prefix as that will also affect our installation.
set(CMAKE_INSTALL_LIBDIR "deps/lib" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_BINDIR "deps/bin" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR "deps/include" CACHE PATH "" FORCE)
set(CMAKE_INSTALL_DATAROOTDIR "deps/share" CACHE PATH "" FORCE)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(SafariDependencies)

# Generate the protobuf messages.
# Source the CMake Protobuf macro.
FetchContent_GetProperties(Protobuf SOURCE_DIR Protobuf_SOURCE_DIR)
include(${Protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)

file(GLOB_RECURSE _SAFARI_PROTO_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/protos/*.proto")

add_library(safari_protos STATIC)
protobuf_generate(
  TARGET safari_protos
  LANGUAGE cpp
  PROTOS ${_SAFARI_PROTO_FILES}
  IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_SOURCE_DIR}/src/
  )

target_link_libraries(safari_protos
  protobuf::libprotobuf
  )
target_include_directories(safari_protos PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  )

# We also need TF example and feature protos. We build them manually.
FetchContent_GetProperties(tensorflow)
add_library(tf_protos STATIC)
protobuf_generate(
  TARGET tf_protos
  LANGUAGE cpp
  PROTOS ${tensorflow_SOURCE_DIR}/tensorflow/core/example/example.proto
         ${tensorflow_SOURCE_DIR}/tensorflow/core/example/feature.proto
  IMPORT_DIRS ${tensorflow_SOURCE_DIR}
  )

target_link_libraries(tf_protos
  protobuf::libprotobuf
  )
target_include_directories(tf_protos PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  )

# Python protos.
protobuf_generate(LANGUAGE python
                  PROTOS ${_SAFARI_PROTO_FILES}
                  OUT_VAR _SAFARI_PB_GENERATED_FILES
                  IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${Protobuf_SOURCE_DIR}/src/
                  PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/py_pb)
# We create a custom target so that we can trigger the file generation automatically.
# We do not need this target for anything else as custom_target does not have any INSTALL associated.
add_custom_target(safari_py_protos DEPENDS ${_SAFARI_PB_GENERATED_FILES})
# Make sure the Python proto target gets built to generate the files.
add_dependencies(safari_protos safari_py_protos)

set(_mcap_include_dir ${mcap_SOURCE_DIR}/cpp/mcap/include)
file(GLOB _mcap_installed_headers ${_mcap_include_dir}/mcap/*.hpp)
add_library(mcap STATIC)
target_include_directories(mcap PUBLIC
  "$<BUILD_INTERFACE:${_mcap_include_dir}>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_sources(mcap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mcap/mcap.cpp)
target_link_libraries(mcap lz4 libzstd_static)


add_library(log_writer_cc STATIC)
target_sources(log_writer_cc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/base_mcap_file_handle_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/episode_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/episode_data.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/log_data_serializer_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/log_data_serializer_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle_factory.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_file_handle.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/mcap_write_op.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/thread_pool_log_writer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/thread_pool_log_writer.cc
)

target_include_directories(log_writer_cc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                          ${OPENCV_MODULE_opencv_core_LOCATION}/include
                          ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
                          ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include)


target_link_libraries(log_writer_cc PUBLIC safari_protos
                                        absl::core_headers
                                        absl::any_invocable
                                        absl::status
                                        absl::statusor
                                        absl::strings
                                        absl::string_view
                                        absl::span
                                        absl::log
                                        absl::flat_hash_map
                                        absl::fixed_array
                                        absl::flat_hash_set
                                        absl::memory
                                        absl::synchronization
                                        absl::time
                                        opencv_core
                                        opencv_imgcodecs
                                        opencv_imgproc
                                        mcap
                                        protobuf::libprotobuf
                                        tf_protos)



# Python bindings
pybind11_add_module(log_writer ${CMAKE_CURRENT_SOURCE_DIR}/safari_sdk/logging/cc/python/log_writer.cc)
target_link_libraries(log_writer PUBLIC log_writer_cc
                                        safari_protos
                                        absl::log
                                        absl::status
                                        absl::statusor
                                        absl::strings
                                        absl::string_view
                                        absl::absl_check
                                        pybind11_abseil::absl_casters
                                        pybind11_abseil::status_casters
                                        pybind11_protobuf::pybind11_native_proto_caster
                                        )

# As we build with static libraries we do not need to install dependencies nor to update
# the RPATH.
install(TARGETS log_writer DESTINATION ${SKBUILD_PROJECT_NAME}/logging/cc/python)

# We install the full output directory of the Python proto to preserve the full path.
# Note: The trailing slash is important as this will force CMake to copy the content of the source directory in the destination
# without creating an additional `safari_sdk` folder.
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/py_pb/safari_sdk/ DESTINATION ${SKBUILD_PROJECT_NAME})

# TODO: At the moment pybind11_abseil does not provide a Pip package.
# We need to install the targets manually. Note that this might conflict with other
# library that might be installed by the user.
install(TARGETS status_py_extension_stub ok_status_singleton DESTINATION pybind11_abseil)

# Now remove the dependencies that we installed but we do not need.
install(CODE "
    message(STATUS \"Removing dependencies installation artifacts...\")
    file(REMOVE_RECURSE \"\${CMAKE_INSTALL_PREFIX}/deps\")
")
