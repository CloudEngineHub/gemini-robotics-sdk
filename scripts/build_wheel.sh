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
SAFARI_DIR="$(realpath "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/..")"
VENV_DIR="$(mktemp -d)"
UPLOAD_TARGET=""
UPLOAD_WHL=false
DOCKER_IMAGE=""

function _usage() {
  echo "Usage: $0 [-h|--help] [--no-smoke-test] [--upload]"
  echo "  -h|--help: Show this help message and exit."
  echo "  --repository-url: Upload the wheel to this repository URL."
  echo "  --docker-image: Use this docker image for building the wheel."
  echo "                  If not specified, a new image will be built from the"
  echo "                  Dockerfile in this directory."
  echo "  --upload: Upload the wheel to PyPi."
}

function cleanup() {
  echo "Cleaning up..."
  rm -rf "${SAFARI_DIR}"/dist/*
  rm -rf "${SAFARI_DIR}"/wheelhouse/*
  rm -rf "${VENV_DIR}"
  if command -v deactivate &> /dev/null; then
    deactivate
  fi
}

trap cleanup EXIT

while (( $# > 0 )) ; do
  case "$1" in
    -h|--help) _usage; exit 1 ;;
    --upload) UPLOAD_WHL=true ; shift ;;
    --docker-image) DOCKER_IMAGE=$2; shift 2 ;;
    --repository-url) UPLOAD_TARGET="--repository-url $2"; UPLOAD_WHL=true; shift 2 ;;
    *) echo "Unknown option: $1"; _usage; exit 1 ;;
  esac
done
cd ${SAFARI_DIR}
# Generate the revision info file. This is a no-op if not built from a git repository.
${SAFARI_DIR}/scripts/generate_revision_info.sh "${SAFARI_DIR}/safari_sdk/revision_info.txt" || true
if [[ -z "${DOCKER_IMAGE}" ]]; then
  echo "No docker image provided"
  echo "${DOCKER_IMAGE}"
  DOCKER_IMAGE="safari-sdk-wheel-build:latest"
  docker build -t "${DOCKER_IMAGE}" \
   -f  kokoro/gcp_ubuntu_docker/Dockerfile .
fi
docker run --rm \
  -u "$(id -u):$(id -g)" \
  -v "${SAFARI_DIR}":/safari \
   "${DOCKER_IMAGE}" /safari/scripts/kokoro_build_wheel.sh

echo "Wheel files should be in the wheelhouse folder."
ls -l "${SAFARI_DIR}/wheelhouse"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
# Upload the wheel to gcloud or PyPI if desired.
if [[ ${UPLOAD_TARGET} == *python.pkg.dev* ]]; then
  # Install the keyrings to allow authentication with Artifact Registry.
  pip install keyrings.google-artifactregistry-auth
  echo "Uploading whl to gCloud (${UPLOAD_TARGET})."
elif [[ -n "${UPLOAD_TARGET}" ]]; then
  echo "Uploading whl to ${UPLOAD_TARGET}."
elif ${UPLOAD_WHL}; then
  echo "Uploading whl to PyPI."
fi

if ${UPLOAD_WHL}; then
  pip install twine
  twine upload --verbose ${SAFARI_DIR}/wheelhouse/*.whl ${UPLOAD_TARGET}
fi

deactivate
