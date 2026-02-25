#!/bin/bash

# Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
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
# Generates a file containing source code revision information from git.
#
# usage:
#   ./generate_version_info.sh output_file
#
# The given output file must be in a directory that is part of a git repository.
# That file will be populated with the following information:
# - The git hash of the most recent commit.
# - The date of the most recent commit.
# - The Piper CL number corresponding to that commit, from the PiperOrigin-RevId
#   tag in the commit message, if present.
# - The Gerrit change id corresponding to that commit, from the Change-Id tag in
#   the commit message, if present.

output_file=$1
if [[ -z "$output_file" ]]; then
  echo "Usage: $0 <output_file>"
  exit 1
fi

git_dir=$(dirname "$output_file")
if ! git_hash=$(git -C "$git_dir" log -n 1 --format="%H"); then
  echo "Warning: Failed to get commit data from git. Is $git_dir a git repository?"
  exit 1
fi
commit_date=$(git -C "$git_dir" log -n 1 --format="%cD")
gerrit_id=$(git -C "$git_dir" log -n 1 --format="%b" | awk '/Change-Id:/ {print $2}')
piper_cl=$(git -C "$git_dir" log -n 1 --format="%b" | awk '/PiperOrigin-RevId:/ {print $2}')

printf "commit_date: %s\ngit_hash: %s\ngerrit_id: %s\npiper_cl: cl/%s\n" \
       "$commit_date" "$git_hash" "$gerrit_id" "$piper_cl" > "$output_file"

echo "Generated the following revision info in ${output_file}:"
cat "${output_file}"
