# contambench

`contambench` is the Python package for the ContamBench data record: stressor suites,
labeled contaminated workloads, and per-request reference measurements for testing
open-weight LLM serving stacks.

A *stressor* is a request whose suffix drives decoding to the output cap. *Workload
contamination* is the share `c` of stressors in a request stream; the term is unrelated
to training-data contamination. A small fraction of stressors can push co-located
benign requests past their latency objectives, and stress tests that replay ordinary
prompts do not exercise this case.

The package fetches and verifies the record, writes labeled workloads at a chosen `c`,
generates suites for new models, runs YAML-configured campaigns against an endpoint you
operate, and scores any run with the SLO rule used for the published results. Its
workloads also run unchanged in existing stress testers, as section 5 shows for [GuideLLM](https://github.com/vllm-project/guidellm).

- Data record: Zenodo, [doi:10.5281/zenodo.23605512](https://doi.org/10.5281/zenodo.23605512) (data version 0.2.0)
- Project page: [contambench.pages.dev](https://contambench.pages.dev)
- Paper: *ContamBench: Reusable Stress Workloads for Shared LLM Serving*, ICSE 2027 Tool Demonstrations and Data Showcase (under review)

## Install

```bash
pip install contambench                 # Python 3.10+, CPU only, no torch
pip install 'contambench[gen]'          # optional: stressor generation (torch, GPU), section 6
```

## 1. Get the data

```bash
contam-bench data fetch                 # all three products, ~630 MB, into ~/.cache/contambench/0.2.0
contam-bench data fetch --subset d1     # one product only: d1, d2, or d3
contam-bench data fetch --from DIR      # copy from a local download of the record
contam-bench data status                # what is present, and where
```

Every file is checked against the record's `SHA256SUMS`. Set `CONTAMBENCH_DATA=DIR` to
use an extracted record somewhere else.

## 2. Reproduce the published tables

```bash
contam-bench report
```

`report` recomputes the stressor-suite table, the contamination sweep at the high
operating point of each deployment, and the GuideLLM reuse table from the request logs,
then compares the sweep with the published values. It runs on a CPU in a few seconds.

## 3. Work with the data, no GPU needed

```bash
# a labeled workload at c = 0.10 from the Llama-3.2-3B suite; writes workload.csv and workload.manifest.json
contam-bench export --suite llama-3.2-3b-instruct --c 0.10 --out workload.csv

# score any run in the D3 format: violation rate, benign-only rate, benign tail inflation, Stage B
contam-bench score ~/.cache/contambench/0.2.0/d3_execution/runs/aws_qwen7b/rq2_sweep --arrival-rate 0.17

# copy the example campaign configurations into ./examples
contam-bench examples

# an offline campaign with synthetic latencies, to see the workflow end to end
contam-bench run --config examples/simulate.yaml
contam-bench score contambench-results/raw/<run_id>
```

Validated suites: `llama-3.2-3b-instruct`, `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct`,
`mistral-7b-instruct-v0.3`. Generated suites with Stage A only (1024-token cap, 8-bit):
`phi-4-mini-instruct`, `granite-4.1-8b`, `seed-coder-8b-instruct`, `qwen2.5-coder-7b-instruct`,
`qwen3-4b-instruct-2507`, `qwen3-8b`, `glm-4-9b-0414`, `deepseek-r1-distill-qwen-7b`. `export --k` sets how many top prompts by
Stage-A success rate the stressors are drawn from (default 50); `--benign FILE` uses your
own benign pool with the columns `prompt_id, prompt, success_rate`.

## 4. Test your own deployment

These steps send traffic to a live OpenAI-compatible endpoint and need the GPU behind
it. Run them only against systems you operate or are authorized to test. Set
`OPENAI_API_KEY` if your endpoint requires a key; keys are never written to the run logs.

```bash
# optional: record vLLM scheduler state (queue depth, KV-cache use, preemptions) during the campaign
contam-bench telemetry --endpoint http://HOST:8000 --interval 1 --out contambench-results/vllm/metrics.jsonl &

# step 1: benign capacity calibration -> capacity and operating points at 60/80/90% of the plateau
contam-bench calibrate --config examples/calibrate.yaml --endpoint http://HOST:8000

# step 2: contamination sweep at the calibrated rates (edit arrival_rate in the YAML first)
contam-bench run --config examples/sweep.yaml --endpoint http://HOST:8000

# step 3: score it
contam-bench score contambench-results/raw/<run_id>
```

Report the Stage-B line of `score` together with any result. A suite generated for a model can
still fail to reach the cap on your deployment; the Mistral-7B suite in the record is an
example. Its Stage-A run used greedy decoding, so `report` prints no Stage-A rate for it.

### Example configurations (`contam-bench examples` copies them to `./examples`)

| File | Campaign |
|---|---|
| `simulate.yaml` | offline run with synthetic latencies |
| `calibrate.yaml` | benign arrival-rate sweep, 50 clients |
| `sweep.yaml` | c ∈ {0, 0.02, 0.05, 0.10, 0.15} at three arrival rates, three trials |
| `control_semantic_suffix.yaml` | a plain-language suffix in place of the optimized one |
| `output_caps.yaml` | the same workload under caps of 512, 1024, 2048, 4096 tokens |

The main keys are `attack_model` (resolves the D1 suite for that model in the fetched
record), `attack_file` (a file path, or `contambench:<path inside the record>`, for
example `contambench:d1_stressors/controls/random_suffix.json`), `workload_size`,
`contamination_ratio`, `num_clients`, `arrival_rate`, `arrival_model`,
`requests_per_client`, `num_trials`, `max_tokens`, `temperature`, `seed`, and
`auto_slo: true`, which takes the SLO budgets from the c = 0 trials of the same run.
Command-line options such as `--endpoint` override the file.

Each run writes `contambench-results/raw/<run_id>/` with `requests.parquet`,
`requests.csv`, and `metadata.json`, in the same format as the record's D3 runs.

## 5. Use the workloads with GuideLLM

[GuideLLM](https://github.com/vllm-project/guidellm) is the vLLM project's load and
SLO benchmarking tool (Apache-2.0). contambench does not include or wrap it: install it
separately and run it as published, with a ContamBench export as its dataset file. Then
convert its report with `import-guidellm` and score it like any other run.

```bash
pip install guidellm==0.7.1             # separate environment recommended; it pulls in torch

contam-bench export --suite llama-3.2-3b-instruct --c 0    --out benign.csv
contam-bench export --suite llama-3.2-3b-instruct --c 0.10 --out workload.csv

for w in benign workload; do
  guidellm run \
    --backend '{"kind": "openai_http", "target": "http://HOST:8000", "request_format": "/v1/chat/completions",
                "stream": true, "extras": {"body": {"temperature": 0.7, "max_tokens": 4096}}}' \
    --profile kind=poisson,rate=13.5 --constraint kind=max_requests,count=1000 \
    --data "{\"kind\": \"csv_file\", \"path\": \"$w.csv\", \"load_kwargs\": {\"split\": \"train\"}}" \
    --data-column-mapper '{"kind": "generative_column_mapper", "column_mappings": {"text_column": "prompt"}}' \
    --output kind=json,path=guidellm_$w.json
  contam-bench import-guidellm guidellm_$w.json --labels $w.csv --max-tokens 4096 --out runs/guidellm_$w
done

contam-bench score runs/guidellm_benign runs/guidellm_workload              # budgets from the benign run
contam-bench score runs/guidellm_workload --budgets other_run/slo_budgets.json --budget-rate 0.27
```

`import-guidellm` labels each request by matching its prompt with the export, so
stressor and benign requests can be scored separately. GuideLLM does not report a
finish reason; the importer counts a request as reaching the cap when its output length
equals `--max-tokens`. Score each report at one rate: run GuideLLM once per rate.

The GuideLLM runs in the data record were made with GuideLLM at upstream commit
`90bc9de` (July 2026, after release 0.7.1). To use exactly that code:
`pip install "guidellm @ git+https://github.com/vllm-project/guidellm@90bc9de035cdc8c6a125e8b9f40bab3225f511ec"`.
The commands above were checked with the released 0.7.1.

## 6. Generate a suite for a new model

`contam-bench generate` optimizes stressor suffixes against a Hugging Face checkpoint
and writes a D1 pairs file with Stage-A records. It needs a CUDA GPU (an A100 40 GB
fits a 3B to 8B model; use `--quantize 8` or `4` for larger ones) and the optional
`gen` extra, which pulls in torch. The base install does not.

Tested with torch 2.8.0 (CUDA 12.8 wheels) and transformers 4.57.6 on an NVIDIA A100 40 GB, so the
extra pins torch below 2.9. Newer torch releases compile Triton kernels at run time and
need a working C compiler on the GPU node.

**Responsible use.** Generate and use stressors only against models and serving
systems that you operate or are explicitly authorized to test. A suite is a load-test
input: its purpose is to show whether a deployment keeps its latency objectives when
some requests run to the output cap.

```bash
pip install 'contambench[gen]'          # torch, transformers, accelerate, bitsandbytes, datasets, ...
contam-bench generate --list-models     # models with shipped metadata (output limit, stop ids)

# LoopLLM, as used for the released suites: 195 Alpaca prompts, 30-token suffix, 100 steps per prompt
contam-bench generate --model meta-llama/Llama-3.2-3B-Instruct --out suites/llama3b

# Engorgio (Dong et al., ICLR 2025), one trigger search, triggers appended to the benign prompts
contam-bench generate --method engorgio --model meta-llama/Llama-3.2-3B-Instruct --out suites/llama3b-engorgio

contam-bench export --pairs suites/llama3b/pairs.json --c 0.10 --out workload.csv
```

| Option | Meaning |
|---|---|
| `--method` | `loopllm` (default) or `engorgio` |
| `--out` | a directory (writes `pairs.json` and `work/`) or a `.json` pairs file (work files go to `<name>.work/`) |
| `--prompts` | benign prompts: `alpaca` (default), `sharegpt`, or a `.json` file (list of strings, list of `{"prompt": ...}`, or `{"prompts": [...]}`) |
| `--n-prompts` | number of benign prompts, one pair each (default 195) |
| `--steps` | LoopLLM steps per prompt (default 100); Engorgio iterations (default 3000) |
| `--seed` | random seed (default 23) |
| `--quantize` | load with bitsandbytes `4` or `8` bit |
| `--max-length` | generation cap in tokens for optimization and Stage A (default: the model's limit, else 4096) |
| `--temperature` | Stage-A sampling temperature (default 0.7). Sampling is always on, also for checkpoints whose `generation_config.json` ships no sampling settings |
| `--trigger-len`, `--gpu-threshold`, `--cpuoffload` | Engorgio trigger length (32), and activation offloading above a GPU-memory percentage |
| `--model-info` | JSON with `name, limit, stop_ids, eot, ...` for a model that is not listed |

A model that is not listed runs with defaults (its own chat template, output limit
4096) and a warning. LoopLLM writes one `res_<i>.json` per prompt into the work
directory, and rerunning the same command resumes from the last one. The pairs file
has the D1 keys (`prompt, adv_suffix, adv_prompt, answer, success_rate, avg_len,
repet`) and a `provenance` block (method, model, suffix length, seed, steps, prompt
source, output limit, package and library versions). `export --pairs` and the harness
`attack_file` option read it directly.

**Stage B is still to be measured.** `success_rate` is Stage A: whether the suffix
drives 16 sampled decodes on the generation GPU to the cap. Whether the stressors
reach the cap on your deployment (Stage B) depends on its serving stack, decoding
settings, and output limit. Measure it with `calibrate`, `run`, and `score`
(section 4) before relying on the suite, and report the Stage-B line with any result.

`scripts/gpu_smoke_generate.sh` runs a small generation job and exports a workload
from it, as a quick check on a GPU machine:
`bash scripts/gpu_smoke_generate.sh --model meta-llama/Llama-3.2-3B-Instruct --method both --out smoke_gen`.

## The oracle

For each arrival rate, the budget of end-to-end latency, time to first token, and time
between tokens is the 99th percentile over the c = 0 requests of the same run. A request
violates when it exceeds any of the three. Rates are computed per trial and averaged,
with a Student-t 95% interval over trials.

## Development

```bash
pip install -e ".[test]"
pytest                                   # CPU tests; set CONTAMBENCH_DATA to also run the data checks
```

## Acknowledgments

The GuideLLM comparison and the `import-guidellm` reader build on
[GuideLLM](https://github.com/vllm-project/guidellm) by the vLLM project, used
under the Apache License 2.0; contambench supplies labeled workloads that it
can run. The serving targets were run with
[vLLM](https://github.com/vllm-project/vllm).

The released stressors were generated with
[LoopLLM](https://github.com/neuron-insight-lab/LoopLLM) (Li et al., *LoopLLM:
Transferable Energy-Latency Attacks in LLMs via Repetitive Generation*, AAAI 2026,
[arXiv:2511.07876](https://arxiv.org/abs/2511.07876)), which `contambench.gen`
vendors under the Apache License 2.0 with its license file. The `engorgio` method is
adapted from [Engorgio](https://github.com/jianshuod/Engorgio-prompt) (Dong et al., *An
Engorgio Prompt Makes Large Language Model Babble on*, ICLR 2025,
[arXiv:2412.19394](https://arxiv.org/abs/2412.19394)). See `THIRD_PARTY_NOTICES.md`
for licenses and changes.

## License

Code: Apache-2.0. The data record is CC BY-NC 4.0, because its benign prompts come from
Stanford Alpaca.
