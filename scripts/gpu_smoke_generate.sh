#!/usr/bin/env bash
# GPU smoke test for `contam-bench generate`: a small LoopLLM job, a small Engorgio job
# (if this build includes the Engorgio port), then `contam-bench export --pairs` on each.
#
# Needs: a CUDA GPU, `pip install 'contambench[gen]'` in the active environment, and
# access to the model on Hugging Face (Llama 3.2 is gated: `huggingface-cli login` or
# set HF_TOKEN). Run only against models you are authorized to test.
#
#   bash scripts/gpu_smoke_generate.sh [OUT_DIR]          # default OUT_DIR=smoke_gen
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/gpu_smoke_generate.sh
set -euo pipefail

OUT=${1:-smoke_gen}
MODEL=${MODEL:-meta-llama/Llama-3.2-3B-Instruct}
N_PROMPTS=${N_PROMPTS:-2}
CB=${CONTAM_BENCH:-contam-bench}
mkdir -p "$OUT"

echo "== contambench $($CB --version) on $(hostname), model $MODEL, output in $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "== LoopLLM: $N_PROMPTS prompts, 5 steps each, cap 512 tokens"
$CB generate --method loopllm --model "$MODEL" --out "$OUT/loopllm" \
    --n-prompts "$N_PROMPTS" --steps 5 --max-length 512 --seed 23 2>&1 | tee "$OUT/loopllm.log"
$CB export --pairs "$OUT/loopllm/pairs.json" --c 0.10 --k "$N_PROMPTS" --out "$OUT/loopllm_c0p10.csv"

if python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('contambench.gen.engorgio') is None)"; then
    echo "== Engorgio: 200 iterations, trigger 16 tokens, cap 256 tokens, $N_PROMPTS prompts"
    $CB generate --method engorgio --model "$MODEL" --out "$OUT/engorgio" \
        --n-prompts "$N_PROMPTS" --steps 200 --trigger-len 16 --max-length 256 --seed 23 2>&1 | tee "$OUT/engorgio.log"
    $CB export --pairs "$OUT/engorgio/pairs.json" --c 0.10 --k "$N_PROMPTS" --out "$OUT/engorgio_c0p10.csv"
else
    echo "== Engorgio port not in this build; skipped (see THIRD_PARTY_NOTICES.md)"
fi

python - "$OUT" <<'PY'
import json, sys, pathlib
for f in sorted(pathlib.Path(sys.argv[1]).glob("*/pairs.json")):
    d = json.loads(f.read_text())
    p = d["provenance"]
    print(f"{f}: {d['pairs']} pairs, mean success_rate={d['average_asr']}, mean avg_len={d['average_len']}, "
          f"generator={p['generator']}, suffix_tokens={p['suffix_tokens']}, seed={p['seed']}, "
          f"contambench={p['versions']['contambench']}")
PY
echo "== done; workloads: $(ls "$OUT"/*.csv | tr '\n' ' ')"
