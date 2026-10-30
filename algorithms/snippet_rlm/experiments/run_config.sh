#!/usr/bin/env bash
# Run one Experiments.xlsx row of the Snippet RLM from its config file (not in the original repository).
#   experiments/run_config.sh configs/<ROW>.env
# Loads configs/paths.env, then the row config (which exports the row's environment and names LAUNCHER and
# LAUNCHER_ARGS), then runs the launcher with those arguments. The launcher itself sets every behaviour-selecting
# variable exactly as the original wrapper did; the config adds only what the original run had implicitly (see the
# "implicit environment" section of each config).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
CFG=$(realpath "${1:?usage: run_config.sh configs/<ROW>.env}")
. "$CFG"
for v in "${LAUNCHER_ARGS[@]}"; do
  case $v in *UNKNOWN*) echo "ABORT: $CFG has an UNKNOWN launcher argument"; exit 2 ;; esac
done
echo "$(date '+%F %T') row=$ROW launcher=$LAUNCHER args=${LAUNCHER_ARGS[*]:-}"
exec "$ALG/$LAUNCHER" "${LAUNCHER_ARGS[@]}"
