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

# Explicitly find Python. This is to avoid issues when PyBind tries to find it
# (incorrectly).
find_package(Python3 COMPONENTS Interpreter Development)

# Declaring and fetching dependencies.
include(FetchContent)

FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20250814.1
)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v3.0.1
)

FetchContent_Declare(
  pybind11_abseil
  GIT_REPOSITORY https://github.com/pybind/pybind11_abseil.git
  GIT_TAG c55fdc9c53d26af70fa8c2314a683abef62fa3f0  # 25 Feb 2025.
)
# pybind11_abseil internally forces interprocedural optimization to off. But
# this is not propagated up to the top level CMakeLists.txt, which means that
# there will be incompatibility between our targets and the one defined by
# pybind11_abseil.
# We force it to on here.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "" FORCE)
set(PYBIND11_PROTOBUF_BUILD_TESTING OFF)
set(BUILD_TESTING OFF)
set(protobuf_INSTALL OFF)
set(protobuf_BUILD_TESTS OFF)

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v32.1
)

FetchContent_Declare(
  pybind11_protobuf
  GIT_REPOSITORY https://github.com/pybind/pybind11_protobuf.git
  GIT_TAG 4825dca68c8de73f5655fc50ce79c49c4d814652 # 29 Oct 2025.
)

# MCAP and its dependencies. Note that MCAP does not have a CMake and we will create
# a target in our setup.
FetchContent_Declare(
  mcap
  GIT_REPOSITORY https://github.com/foxglove/mcap.git
  GIT_TAG releases/mcap-cli/v0.0.60
)

FetchContent_Declare(
  lz4
  GIT_REPOSITORY https://github.com/lz4/lz4.git
  GIT_TAG v1.10.0
  SOURCE_SUBDIR  build/cmake
)

set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
# Build zstd as a static library.
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  zstd
  GIT_REPOSITORY https://github.com/facebook/zstd.git
  GIT_TAG v1.5.7
  SOURCE_SUBDIR build/cmake
)

# We only download TF.
FetchContent_Declare(
  tensorflow
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG v2.19.1
)


# Disable all the unneeded OpenCV options
set(WITH_WEBP OFF CACHE BOOL "" FORCE)
set(WITH_TIFF OFF CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_V4L OFF CACHE BOOL "" FORCE)
set(WITH_PROTOBUF OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_GIF OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_HDR OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_SUNRASTER OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_PXM OFF CACHE BOOL "" FORCE)
set(WITH_IMGCODEC_PFM OFF CACHE BOOL "" FORCE)
set(WITH_FLATBUFFERS OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(INSTALL_PYTHON_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSTALL_C_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
# Specify only the required modules.
set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "Selected OpenCV modules" FORCE)
set(OPENCV_SETUPVARS_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}" CACHE PATH "" FORCE)

FetchContent_Declare(
  opencv
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG 4.13.0
)

FetchContent_MakeAvailable(abseil-cpp protobuf
                           pybind11 pybind11_abseil pybind11_protobuf
                           opencv lz4 zstd mcap
                           tensorflow)
