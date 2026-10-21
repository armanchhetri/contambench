#!/usr/bin/env bash
# GPU smoke test for `contam-bench generate`: a small generation job per method, then
# `contam-bench export --pairs` on each result.
#
# Needs a CUDA GPU, `pip install 'contambench[gen]'`, and access to the model on Hugging
# Face (Llama 3.2 is gated: run `huggingface-cli login` first). Run it only against
# models you are authorized to test.
#
#   bash scripts/gpu_smoke_generate.sh --model meta-llama/Llama-3.2-3B-Instruct --method both --out smoke_gen
#
# Options (defaults in brackets):
#   --model ID               Hugging Face model id             [meta-llama/Llama-3.2-3B-Instruct]
#   --method M               loopllm | engorgio | both         [both]
#   --out DIR                output directory                  [smoke_gen]
#   --n-prompts N            benign prompts (pairs)            [2]
#   --steps N                LoopLLM steps per prompt          [5]
#   --engorgio-steps N       Engorgio iterations               [200]
#   --max-length N           LoopLLM generation cap, tokens    [512]
#   --engorgio-max-length N  Engorgio generation cap, tokens   [256]
#   --trigger-len N          Engorgio trigger length           [16]
#   --prompts SRC            alpaca | sharegpt | FILE.json     [alpaca]
#   --c C                    contamination ratio for export    [0.10]
set -euo pipefail

MODEL=meta-llama/Llama-3.2-3B-Instruct
METHOD=both
OUT=smoke_gen
N_PROMPTS=2
STEPS=5
ENG_STEPS=200
MAX_LEN=512
ENG_MAX_LEN=256
TRIGGER_LEN=16
PROMPTS=alpaca
C=0.10

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --method) METHOD=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --n-prompts) N_PROMPTS=$2; shift 2 ;;
        --steps) STEPS=$2; shift 2 ;;
        --engorgio-steps) ENG_STEPS=$2; shift 2 ;;
        --max-length) MAX_LEN=$2; shift 2 ;;
        --engorgio-max-length) ENG_MAX_LEN=$2; shift 2 ;;
        --trigger-len) TRIGGER_LEN=$2; shift 2 ;;
        --prompts) PROMPTS=$2; shift 2 ;;
        --c) C=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$METHOD" in loopllm|engorgio|both) ;; *) echo "--method must be loopllm, engorgio, or both" >&2; exit 2 ;; esac

mkdir -p "$OUT"
echo "== $(contam-bench --version), model $MODEL, method $METHOD, output in $OUT"

if [ "$METHOD" = loopllm ] || [ "$METHOD" = both ]; then
    echo "== LoopLLM: $N_PROMPTS prompts, $STEPS steps each, cap $MAX_LEN tokens"
    contam-bench generate --method loopllm --model "$MODEL" --out "$OUT/loopllm" --prompts "$PROMPTS" \
        --n-prompts "$N_PROMPTS" --steps "$STEPS" --max-length "$MAX_LEN" --seed 23 2>&1 | tee "$OUT/loopllm.log"
    contam-bench export --pairs "$OUT/loopllm/pairs.json" --c "$C" --k "$N_PROMPTS" --out "$OUT/loopllm_workload.csv"
fi

if [ "$METHOD" = engorgio ] || [ "$METHOD" = both ]; then
    echo "== Engorgio: $ENG_STEPS iterations, trigger $TRIGGER_LEN tokens, cap $ENG_MAX_LEN tokens, $N_PROMPTS prompts"
    contam-bench generate --method engorgio --model "$MODEL" --out "$OUT/engorgio" --prompts "$PROMPTS" \
        --n-prompts "$N_PROMPTS" --steps "$ENG_STEPS" --trigger-len "$TRIGGER_LEN" --max-length "$ENG_MAX_LEN" \
        --seed 23 2>&1 | tee "$OUT/engorgio.log"
    contam-bench export --pairs "$OUT/engorgio/pairs.json" --c "$C" --k "$N_PROMPTS" --out "$OUT/engorgio_workload.csv"
fi

echo "== done: pairs in $OUT/*/pairs.json, workloads in $OUT/*_workload.csv"
