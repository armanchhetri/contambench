# contambench

`contambench` is the Python package for the ContamBench data record: stressor suites,
labeled contaminated workloads, and per-request reference measurements for testing
open-weight LLM serving stacks.

A *stressor* is a request whose suffix drives decoding to the output cap. *Workload
contamination* is the share `c` of stressors in a request stream; the term is unrelated
to training-data contamination. A small fraction of stressors can push co-located
benign requests past their latency objectives, and capacity tests that replay benign
prompt pools do not exercise this case.

The package fetches and verifies the record, writes labeled workloads at a chosen `c`,
runs YAML-configured campaigns against an endpoint you operate, and scores any run with
the SLO oracle used for the published results. It does not introduce a new
load-generation technique. Its workloads also run unchanged on existing load generators;
section 5 shows this for [GuideLLM](https://github.com/vllm-project/guidellm).

- Data record: Zenodo, [doi:10.5281/zenodo.23482023](https://doi.org/10.5281/zenodo.23482023)
- Project page: [contambench.pages.dev](https://contambench.pages.dev)
- Paper: *ContamBench: Stressor Suites and Contaminated Workload Traces for Testing LLM Serving*, ICSE 2027 Tool Demonstrations and Data Showcase (under review)

## Install

```bash
pip install contambench                 # Python 3.10+, CPU only, no torch
```

## 1. Get the data

```bash
contam-bench data fetch                 # all three products, ~630 MB, into ~/.cache/contambench/0.1.0
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
contam-bench score ~/.cache/contambench/0.1.0/d3_execution/runs/aws_qwen7b/rq2_sweep --arrival-rate 0.17

# copy the example campaign configurations into ./examples
contam-bench examples

# an offline campaign with synthetic latencies, to see the workflow end to end
contam-bench run --config examples/simulate.yaml
contam-bench score contambench-results/raw/<run_id>
```

Suites: `llama-3.2-3b-instruct`, `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct`,
`mistral-7b-instruct-v0.3`, `qwen2.5-math-7b`. `export --k` sets how many top prompts by
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

Report the Stage-B line of `score` together with any result. A suite that saturates in
isolation can still fail to reach the cap on your deployment; the Mistral-7B suite in
the record is an example.

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

## The oracle

For each arrival rate, the budget of end-to-end latency, time to first token, and time
between tokens is the 99th percentile over the c = 0 requests of the same run. A request
violates when it exceeds any of the three. Rates are computed per trial and averaged,
with a Student-t 95% interval over trials.

## Not in this package

Stressor generation, LoopLLM suffix optimization against a new checkpoint, needs a GPU
workstation and is not included. A new suite needs Stage-A records from generation and
Stage-B records from a contamination sweep on the target deployment.

## Development

```bash
pip install -e ".[test]"
pytest                                   # CPU tests; set CONTAMBENCH_DATA to also run the data checks
```

## Acknowledgments

The GuideLLM comparison and the `import-guidellm` reader build on
[GuideLLM](https://github.com/vllm-project/guidellm) by the vLLM project, used
unmodified under the Apache License 2.0; contambench supplies labeled workloads that it
can run. The serving targets were run with
[vLLM](https://github.com/vllm-project/vllm), and the stressors were generated with
[LoopLLM](https://arxiv.org/abs/2511.07876).

## License

Code: Apache-2.0. The data record is CC BY-NC 4.0, because its benign prompts come from
Stanford Alpaca.
