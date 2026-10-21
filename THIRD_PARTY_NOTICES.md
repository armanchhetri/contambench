# Third-party notices

contambench is licensed under the Apache License 2.0 (see `LICENSE`). The optional
stressor generator (`contambench.gen`, installed with `pip install 'contambench[gen]'`)
contains code adapted from the projects below.

## LoopLLM

- Code: `src/contambench/gen/loopllm/` (attack and `utils/`), vendored from
  <https://github.com/neuron-insight-lab/LoopLLM>.
- License: Apache License 2.0; the full text ships next to the code in
  `src/contambench/gen/loopllm/LICENSE`, and the upstream README in
  `src/contambench/gen/loopllm/README.upstream.md`.
- Changes: adapted to contambench's `Attacker`/`Model` wrappers; selectable loss
  configurations (the cycle loss, used for the released suites, is the default); a
  configurable Stage-A success threshold; prompts from local JSON files; prompt sources
  that only existed in the authors' research repository removed; an option to skip the
  final in-memory re-evaluation; import paths. Each modified file says so in its header.
- Paper: Xingyu Li, Xiaolei Liu, Cheng Liu, Yixiao Xu, Kangyi Ding, Bangzhou Xin, and
  Jia-Li Yin. *LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive
  Generation.* Proceedings of the AAAI Conference on Artificial Intelligence 40(38),
  2026. [doi:10.1609/aaai.v40i38.40445](https://doi.org/10.1609/aaai.v40i38.40445),
  [arXiv:2511.07876](https://arxiv.org/abs/2511.07876).
- LoopLLM itself builds on GCG, <https://github.com/llm-attacks/llm-attacks>
  (Zou et al., 2023).

## Engorgio

- Code: `src/contambench/gen/engorgio/` (attack loop, `ica_utils/loss.py`,
  `ica_utils/prepare.py`, `ica_utils/ctl.py`), adapted from
  <https://github.com/jianshuod/Engorgio-prompt> (commit `5fb50dd`, August 2025).
- License: the upstream repository has **no license file** and its README states no
  license terms (checked 2026-10-21). Without a license, redistribution rights are not
  granted, so this port is not part of public contambench builds until the authors grant
  permission or add a license. Builds without it report this when
  `--method engorgio` is requested.
- Changes: soft-prompt optimization driven through contambench's `Model` wrapper and the
  model's chat template; multiple end-of-sequence ids; an optional second loss version;
  CPU offloading of activations above a GPU-memory threshold; the trigger tokens kept
  with each candidate so pairs can be built.
- Paper: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang,
  Hewu Li, Qi Li, Chao Zhang, and Ke Xu. *An Engorgio Prompt Makes Large Language Model
  Babble on.* ICLR 2025. <https://openreview.net/forum?id=m4eXBo0VNc>,
  [arXiv:2412.19394](https://arxiv.org/abs/2412.19394).

## Prompt data

The default benign prompts are downloaded at run time from Stanford Alpaca
(<https://huggingface.co/datasets/tatsu-lab/alpaca>, CC BY-NC 4.0) or, with
`--prompts sharegpt`, from <https://huggingface.co/datasets/shibing624/sharegpt_gpt4>.
They are not shipped with the package; check their terms before redistributing a suite.
