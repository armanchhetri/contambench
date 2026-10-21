# Vendored from LoopLLM (https://github.com/neuron-insight-lab/LoopLLM), Apache-2.0; see LICENSE.
# Modified by the ContamBench authors: adapted to the research codebase's Attacker/Model
# wrappers, configurable loss and success threshold, and (for contambench) import paths.
from contambench.gen import require_extra

require_extra()

import gc  # noqa: E402
import json
import os
import time
from typing import List

import torch
from tqdm import tqdm
from transformers import set_seed

from contambench.gen.attacker import Attacker, AttackParams
from contambench.gen.models import Model
from contambench.gen.stats import run_multiinference

from .utils import (
    DEFAULT_CYCLIC_SEGMENT_LEN,
    DEFAULT_LOSS_CONFIG,
    SuffixManager,
    generate_str,
    get_all_losses,
    get_chat_prompt,
    get_filtered_cands,
    get_gradients,
    get_nonallowed_toks,
    is_entropy_low,
    read_data,
    sample_control,
    test_suffix,
    normalized_lz_complexity,
)


class LoopLLMAttackParams(AttackParams):
    results_dir: str
    seed: int = 23
    data_name: str = "alpaca"
    adv_len: int = 30
    cyclic_segment_len: int = DEFAULT_CYCLIC_SEGMENT_LEN
    steps: int = 50
    topk: int = 64
    num_candidate: int = 128
    max_length: int = 1024
    once_forward_batch: int = 8
    eval_interval: int = 1
    dataset_length: int = 195
    no_cuda: bool = False
    loss_config: str = DEFAULT_LOSS_CONFIG
    req_success_rate: float = 0.5
    # contambench: the research run re-evaluated the final prompts with 16 raw (no chat
    # template) generations each and kept the numbers only in memory. The D1 summary
    # comes from the res_*.json files, so `contam-bench generate` turns this off.
    evaluate_collected: bool = True


class LoopLLMAttacker(Attacker):
    def __init__(self, model: Model):
        super().__init__(model)

    def get_attack_prompts(self, count: int = 40) -> List[dict]:
        return self._attack_prompts[:count]

    def populate_attack_prompts(
        self, adv_prompts: List[str], params: LoopLLMAttackParams
    ):
        for prompt_str in adv_prompts:
            prompt_stat = run_multiinference(self.model, prompt_str, total_runs=16)
            prompt_stat = prompt_stat | {
                "model_name": self.model.get_name(),
                "prompt": prompt_str,
                "trigger_len": params.adv_len,
                "max_len": params.max_length,
            }
            self._attack_prompts.append(prompt_stat)
        self._attack_prompts.sort(key=lambda x: x["avg_response_len"], reverse=True)

    def _individual_gcg(
        self,
        model,
        tokenizer,
        prompt: str,
        epoch: int,
        adv_suffix: str,
        cyclic_segment_ids: list,
        params: LoopLLMAttackParams,
        not_allowed_tokens=None,
        save_dir: str | None = None,
        loss_config: str = DEFAULT_LOSS_CONFIG,
    ) -> str:
        device = model.device

        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            instruction=prompt,
            adv_len=params.adv_len,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
        )
        suffix_manager.update(adv_suffix=adv_suffix)
        suffix_manager.adv_token_id = cyclic_segment_ids

        is_success = False
        res = {}
        adv_prompt = f"{prompt} {adv_suffix.strip()}"
        answer, initial_len, _, _ = generate_str(model, tokenizer, adv_prompt)
        res[-1] = {
            "prompt": adv_prompt,
            "answer": answer,
            "total_len": initial_len,
            "repet": normalized_lz_complexity(answer),
        }
        suffix_manager.update(answer=answer)

        if save_dir:
            save_path = os.path.join(save_dir, f"res_{epoch}.json")
            with open(save_path, "w") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)

        for i in tqdm(range(params.steps)):
            start_time = time.time()
            input_ids = suffix_manager.get_input_ids().to(device)
            coordinate_grad = get_gradients(
                model, input_ids, suffix_manager, loss_config=loss_config
            )

            with torch.no_grad():
                adv_suffix_tokens = input_ids[suffix_manager._control_slice]
                new_adv_suffix_toks = sample_control(
                    adv_suffix_tokens,
                    coordinate_grad,
                    params.num_candidate,
                    params.topk,
                    not_allowed_tokens=not_allowed_tokens,
                )
                new_adv_suffixs = get_filtered_cands(
                    tokenizer,
                    new_adv_suffix_toks,
                    adv_suffix_tokens,
                    fill_cand=False,
                )
                losses = get_all_losses(
                    model,
                    tokenizer,
                    input_ids,
                    new_adv_suffixs,
                    suffix_manager,
                    batch_size=params.once_forward_batch,
                    loss_config=loss_config,
                )

                best_id = losses.argmin()
                adv_suffix = new_adv_suffixs[best_id]
                adv_prompt = f"{prompt} {adv_suffix.strip()}"
                current_losses = losses[best_id]

                res[i] = {
                    "prompt": prompt,
                    "adv_suffix": adv_suffix,
                    "adv_prompt": adv_prompt,
                    "current_losses": current_losses.item(),
                    "loss_config": loss_config,
                }

                if (i + 1) % params.eval_interval == 0:
                    input_ids = get_chat_prompt(
                        tokenizer,
                        adv_prompt,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    success_rate, avg_len, answer = test_suffix(
                        model,
                        tokenizer,
                        input_ids,
                        batch=params.once_forward_batch
                    )
                    is_success =  success_rate >= params.req_success_rate
                    
                    res[i]["answer"] = answer
                    res[i]["success_rate"] = success_rate
                    res[i]["avg_len"] = avg_len
                    res[i]["repet"] = normalized_lz_complexity(answer)
                    suffix_manager.update(answer=answer)

                res[i]["time"] = time.time() - start_time

                if save_dir:
                    with open(save_path, "w") as f:
                        json.dump(res, f, indent=4, ensure_ascii=False)

                suffix_manager.update(adv_suffix=adv_suffix)

                if is_success:
                    _input_ids = suffix_manager.get_input_ids().to(device).unsqueeze(0)
                    if is_entropy_low(model, _input_ids, threshold=0.05):
                        break

            del input_ids, coordinate_grad, new_adv_suffix_toks, losses
            gc.collect()
            torch.cuda.empty_cache()

        return adv_prompt

    def attack(
        self,
        params: LoopLLMAttackParams,
        log_prompts: bool = False,
        gpu_threshold=85,
    ):
        del gpu_threshold  # LoopLLM does not use memory tracking
        print(f"Starting LoopLLM attack with\n{params}")

        set_seed(params.seed)
        requested_device = (
            "cuda"
            if torch.cuda.is_available() and not params.no_cuda
            else "cpu"
        )
        self.model.set_device(requested_device)

        model = self.model.hf_model
        device = model.device
        tokenizer = self.model.tokenizer
        model.generation_config.max_length = params.max_length

        not_allowed_tokens = get_nonallowed_toks(tokenizer, model.device) # get_non_ascii_toks

        adv_token_ids = [
            tokenizer.encode("* " * 20)[-5],
            tokenizer.encode("% " * 20)[-5],
            tokenizer.encode("& " * 20)[-5],
            tokenizer.encode("@ " * 20)[-5],
            tokenizer.encode("# " * 20)[-5],
        ][: params.cyclic_segment_len]
        cyclic_segment_ids = adv_token_ids
        adv_suffix = tokenizer.decode(
            (cyclic_segment_ids * params.adv_len)[: params.adv_len]
        )

        save_dir = params.results_dir
        os.makedirs(save_dir, exist_ok=True)

        data = read_data(params.data_name, length=params.dataset_length)

        start_epoch = 0
        result_files = sorted(
            (
                path
                for path in os.listdir(save_dir)
                if path.startswith("res_")
                and path.endswith(".json")
                and path.split(".")[0].split("_")[-1].isdigit()
            ),
            key=lambda x: int(x.split(".")[0].split("_")[-1]),
        )
        if result_files:
            path = result_files[-1]
            start_epoch = int(path.split(".")[0].split("_")[-1])
            if start_epoch >= len(data) - 1:
                print("All epochs already completed.")
                return []

        collected_adv_prompts = []
        for epoch in range(start_epoch, len(data)):
            print(f"\n========epoch {epoch} / {len(data)}========")
            prompt = data[epoch]

            adv_prompt = self._individual_gcg(
                model,
                tokenizer,
                prompt,
                epoch,
                adv_suffix,
                cyclic_segment_ids,
                params,
                not_allowed_tokens=not_allowed_tokens,
                save_dir=save_dir,
                loss_config=params.loss_config,
            )
            collected_adv_prompts.append(adv_prompt)
            if log_prompts:
                print(repr(adv_prompt))

            gc.collect()
            torch.cuda.empty_cache()

        if params.evaluate_collected:
            self.populate_attack_prompts(collected_adv_prompts, params)
        return collected_adv_prompts
