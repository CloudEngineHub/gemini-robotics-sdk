#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e
SAFARI_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
OUTPUT_DIR="${OUTPUT_DIR:-$(realpath "${SAFARI_DIR}/..")}"
VENV_DIR="$(mktemp -d)"

trap 'rm -rf "${VENV_DIR}"' EXIT

cd "${SAFARI_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install -q --upgrade pip
pip install -q build cibuildwheel

echo "Building sdist..."
python -m build --sdist --outdir "${OUTPUT_DIR}/dist"

echo "Building wheel with cibuildwheel..."
CIBW_BUILD="cp311-* cp312-* cp313-*" \
CIBW_SKIP="*-musllinux_*" \
  cibuildwheel --platform linux --output-dir "${OUTPUT_DIR}/wheelhouse"

echo "Sdist files:"
ls -l "${OUTPUT_DIR}/dist"
echo "Wheel files:"
ls -l "${OUTPUT_DIR}/wheelhouse"