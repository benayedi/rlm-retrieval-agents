#!/usr/bin/env bash
# The snippet REPL agent (src/run_obliq.py through experiments/obliq_shard.sh) with Qwen3.6-35B-A3B on OBLIQ-Twitter:
# top-k 20, 12 turns, the Qwen3.6 switches (REPL-fence cue, tolerant fences, presence penalty 1.5) and search tagging.
# TWQ-1 = ARM ctrl (plain dense search, Qwen3-Embedding-8B on :8010); SP-6 = ARM rr (the novelty + reranker proxy
# with WIRE=mine on :8775, started by studies/search_policy/experiments/sp6_obliq_snippet_qwen36_rerank.sh).
# Two slices, run at different times with slightly different settings, reproduced exactly:
#   SLICE=dev100   the 100 dev questions, 10 workers, LLM on the shard default :8030. RECONSTRUCTED (no launcher file):
#                  one command line of 2026-10-13 09:19 in the Claude Code session transcript 8ba31816, both arms:
#                    env EMBEDDING_SERVER_URL=http://localhost:<8010|8775> LOCAL_LLM_MODEL=qwen36-35b-a3b OBLIQ_REPL_CUE=1
#                      OBLIQ_FENCE_TOLERANT=1 OBLIQ_PRESENCE_PENALTY=1.5 OBLIQ_TAG_SEARCH=1 LOCAL_LM_TIMEOUT=1800
#                      RLM_TRAJ_DIR=logs/obliq_mine_rr/traj_<arm> bash scripts/run_obliq_shard.sh
#                      data/obliq_splits/twitter_dev.json logs/obliq_mine_rr/<arm>_dev100 20 12 10
#   SLICE=rest181  the other 181 questions over both GPUs (ids rest181_g0/g1, 8 workers each, LLM :8030 / :8031).
#                  ctrl: the SNIP branch of <original repo>/scripts/obliq_best_chain.sh, snip_run 12 data/obliq_fd/rest181
#                  logs/obliq_mine_rr/snip_rest181 (2026-10-20; copy in original/RLM_2026-11-11/RLM/scripts/), which the
#                  dev100 pick (src/obliq_pick_best.py) chose. rr: RECONSTRUCTED from one command line of 2026-10-21 12:34
#                  in transcript 1a0da3c7 (the same variables plus EMBEDDING_SERVER_URL=http://localhost:8775 and
#                  RLM_TRAJ_DIR=logs/obliq_mine_rr/traj_rr_rest181).
# Locations: experiments/obliq_shard.sh runs in $RUNS_DIR, so the original relative paths logs/obliq_mine_rr/... become
# $RUNS_DIR/obliq_mine_rr/...; the id files are read from IDS_DIR (default $RLM_DATA_DIR: obliq_fd/rest181_g{0,1}.json)
# and the repository copy of twitter_dev.json. Needs OBLIQ_BENCH_ROOT, the :8010 search server and vLLM on :8030 (+ :8031).
#   ARM=ctrl|rr SLICE=dev100|rest181 experiments/obliq_twitter_qwen36.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ARM=${ARM:?set ARM=ctrl or rr}; SLICE=${SLICE:?set SLICE=dev100 or rest181}
case "$ARM" in ctrl) SEARCH=http://localhost:8010 ;; rr) SEARCH=http://localhost:8775 ;; *) echo "ARM: ctrl or rr" >&2; exit 2 ;; esac
curl -s -m10 -o /dev/null "$SEARCH/health" || { echo "ABORT: search server $SEARCH down" >&2; exit 1; }
SHARD=$ALG/experiments/obliq_shard.sh
mkdir -p "${RUNS_DIR:?}/obliq_mine_rr"
if [ "$SLICE" = dev100 ]; then
  env EMBEDDING_SERVER_URL=$SEARCH LOCAL_LLM_MODEL=qwen36-35b-a3b OBLIQ_REPL_CUE=1 OBLIQ_FENCE_TOLERANT=1 \
    OBLIQ_PRESENCE_PENALTY=1.5 OBLIQ_TAG_SEARCH=1 LOCAL_LM_TIMEOUT=1800 RLM_TRAJ_DIR=obliq_mine_rr/traj_$ARM \
    bash "$SHARD" "$ART/studies/prompt_optimisation/splits/obliq_splits/twitter_dev.json" obliq_mine_rr/${ARM}_dev100 20 12 10 \
    > "$RUNS_DIR/obliq_mine_rr/shard_$ARM.log" 2>&1
elif [ "$SLICE" = rest181 ]; then
  IDS_DIR=${IDS_DIR:-${RLM_DATA_DIR:?set RLM_DATA_DIR or IDS_DIR (directory holding obliq_fd/rest181_g0.json)}}
  case "$ARM" in ctrl) NAME=snip_rest181; TRAJ=obliq_mine_rr/traj_snip_rest181 ;;
                 rr)   NAME=rr_rest181;   TRAJ=obliq_mine_rr/traj_rr_rest181 ;; esac
  for g in 0 1; do
    ( env OBLIQ_REPL_CUE=1 OBLIQ_FENCE_TOLERANT=1 OBLIQ_PRESENCE_PENALTY=1.5 OBLIQ_TAG_SEARCH=1 LOCAL_LM_TIMEOUT=1800 \
        EMBEDDING_SERVER_URL=$SEARCH LOCAL_LLM_MODEL=qwen36-35b-a3b LOCAL_LLM_BASE_URL=http://localhost:803$g/v1 \
        RLM_TRAJ_DIR=$TRAJ OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
        "$SHARD" "$(realpath "$IDS_DIR/obliq_fd/rest181_g$g.json")" obliq_mine_rr/${NAME}_g$g 20 12 8 \
        > "$RUNS_DIR/${NAME}_g$g.log" 2>&1 ) & eval P$g=\$!
  done
  wait $P0 $P1
else
  echo "SLICE: dev100 or rest181" >&2; exit 2
fi
