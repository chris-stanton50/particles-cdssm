#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

shopt -s nullglob
config_files=(config/*.json)

if (( ${#config_files[@]} == 0 )); then
    echo "No JSON config files found in $script_dir/config" >&2
    exit 1
fi

for config_file in "${config_files[@]}"; do
    config_name="${config_file##*/}"
    config_name="${config_name%.json}"
    python smoothing_experiment.py -c "$config_name"
done