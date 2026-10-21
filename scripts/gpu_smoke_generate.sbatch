#!/bin/bash -l
# SLURM wrapper for scripts/gpu_smoke_generate.sh (same header as the research repo's
# schedule_main.py). Submit from the contambench checkout:
#
#   mkdir -p slurm && sbatch scripts/gpu_smoke_generate.sbatch
#
# Overrides: CONTAMBENCH_DIR (checkout, default: submit directory), GEN_VENV (venv
# with contambench[gen], created with uv on first use), OUT, MODEL, N_PROMPTS.
#SBATCH --job-name=contambench_gen_smoke
#SBATCH --mail-user=slack:@ac7240
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=cost-llm
#SBATCH --partition=tier3
#SBATCH --output=slurm/gen_smoke_%x_%j.out
#SBATCH --error=slurm/gen_smoke_%x_%j.err
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=48g
#SBATCH --gres=gpu:a100:1
set -euo pipefail

source ~/.conda/etc/profile.d/conda.sh
conda activate llm_cost_env

CONTAMBENCH_DIR=${CONTAMBENCH_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
GEN_VENV=${GEN_VENV:-$HOME/.venvs/contambench-gen}
OUT=${OUT:-$CONTAMBENCH_DIR/smoke_gen/$SLURM_JOB_ID}
cd "$CONTAMBENCH_DIR"

# A separate venv keeps the research environment's packages untouched.
if [ ! -x "$GEN_VENV/bin/contam-bench" ]; then
    uv venv "$GEN_VENV" --python 3.11
    uv pip install --python "$GEN_VENV/bin/python" -e "$CONTAMBENCH_DIR[gen]"
fi
source "$GEN_VENV/bin/activate"

bash scripts/gpu_smoke_generate.sh "$OUT"
