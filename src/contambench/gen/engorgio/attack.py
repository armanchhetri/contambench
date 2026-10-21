# Engorgio (Dong et al., ICLR 2025, "An Engorgio Prompt Makes Large Language Model Babble on").
# Adapted by the ContamBench authors from https://github.com/jianshuod/Engorgio-prompt
# (commit 5fb50dd). The upstream repository carries no license file; see THIRD_PARTY_NOTICES.md.
from contambench.gen import require_extra

require_extra()

import gc  # noqa: E402
import random
import time
import sys
from typing import List

import torch
import torch.nn.functional as F

from contambench.gen.stats import memory_usage, run_multiinference, track_memory

from .ica_utils.loss import total_loss_v1, LossParams, total_loss_v2
from contambench.gen.models import Model
from contambench.gen.attacker import Attacker, AttackParams
from .ica_utils.prepare import TemplateFactory
from .ica_utils.ctl import Converge_check

class EngorgioAttackParams(AttackParams):

    ## memory optimization
    cpuoffload: bool = True
    # [Basic], Experiment Settings
    seed: int = 123
    log_interval: int = 200

    # [Warmup]
    warmup_lr: float = 0.1
    esc_loss_warmup_iters: int = 20
    warmup_alpha: int = 1
    warmup_filter: bool = True
    warmup_initial_coeff: int = 5

    # [Training], Design Settings
    converge_count: int = 50
    lr: float = 0.05
    num_iters: int = 1
    opt_alpha: int = 1
    alpha: float = 0.05
    loss_opt: List[int] = [1, 2]
    esc_loss_version: int = 0
    sm_loss_version: int = 0
    trigger_esc_eos: bool = False
    alternate: bool = False
    loss_version:int = 1

    # [Initialization], Theta Settings
    trigger_token_length: int = 32
    max_length: int = 2048
    initial_coeff: float = 1e-3
    normal_init: bool = False

    # [Inference], Evaluation Settings
    bs: int = 8
    sample_time: int = 80
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    eval_v2: bool = False


class EngorgioAttacker(Attacker):
    def __init__(self, model: Model):
        super().__init__(model)

    def get_attack_prompts(self, count=40) -> List[dict]:
        return self._attack_prompts[:count]

    def populate_attack_prompts(
        self, candidate_triggers: List, params: EngorgioAttackParams, device: str
    ):
        # print(len(candidate_triggers))
        for candidate_ids in candidate_triggers:
            prompt_str = self.model.decode_from_ids(candidate_ids)
            prompt_stat = run_multiinference(self.model, prompt_str, total_runs = 16)
            # contambench: also keep the trigger tokens alone (the user-message part).
            lo = getattr(self, "_prefix_length", None)
            trig = candidate_ids[lo:lo + params.trigger_token_length] if lo is not None else None
            prompt_stat = prompt_stat | {
                "trigger_ids": trig,
                "trigger_text": self.model.decode_from_ids(trig, skip_special_tokens=True) if trig is not None else None,
                "model_name": self.model.get_name(),
                "prompt": prompt_str,
                "trigger_len": params.trigger_token_length,
                "max_len" : params.max_length
            }

            self._attack_prompts.append(prompt_stat)
        self._attack_prompts.sort(key=lambda x: x["avg_response_len"], reverse=True)

    @track_memory
    def attack(self, params: EngorgioAttackParams, log_prompts=False, gpu_threshold = 85):
        if gpu_threshold is None:
            gpu_threshold = 85
        print(f"Starting engorgio attack with gpu_threshold = {gpu_threshold} and \n{params}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.set_device(device)
        hf_model = self.model.hf_model
        tokenizer = self.model.tokenizer
        total_vocab_size = self.model.total_vocab_size

        eos_token_id = self.model.eos_token_id
        eot_token_id = self.model.eot_token_id
        all_eos_token_ids = getattr(self.model, 'all_eos_token_ids', [eos_token_id])

        embeddings = (
            hf_model.get_input_embeddings()(
                torch.arange(0, total_vocab_size).to(device)
            )
            .detach()
            .to(device)
        )
        trigger_seq_length = params.trigger_token_length
        global_triggers = []
        losse1s = []
        losse2s = []
        losses = []
        # -----------------[Init the Env]------------------
        template_fac = TemplateFactory(
            self.model, self.model.get_name(), trigger_seq_length, tokenizer, embeddings
        )
        # template_fac.add_additional_prompt(prefix_sentence)
        template_len = template_fac.template_length
        self._prefix_length = template_fac.prefix_length
        theta_length = params.max_length - template_len
        # params.theta_length = theta_length
        checker = Converge_check(params.converge_count)
        # -----------------[Init the Trigger Theta]------------------
        log_coeffs = torch.zeros(theta_length, total_vocab_size, dtype=embeddings.dtype)

        trigger_ids = [
                random.randint(0, total_vocab_size - 1) for _ in range(theta_length)
            ]
        indices = torch.arange(len(trigger_ids)).long()
        log_coeffs[indices, torch.LongTensor(trigger_ids)] = params.initial_coeff
        log_coeffs = log_coeffs.to(device)
        log_coeffs.requires_grad = True

        trigger_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).tolist()

        # -----------------[Training]------------------
        optimizer = torch.optim.Adam([log_coeffs], lr=params.lr)
        loss_options = params.loss_opt
        s_time = time.time()
        for i in range(params.num_iters):
            # Warmup using <EOS> Escape Loss
            if i == params.esc_loss_warmup_iters and params.warmup_filter:
                log_top1 = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).tolist()
                with torch.no_grad():
                    log_coeffs.zero_()
                    log_coeffs[indices, torch.LongTensor(log_top1)] = (
                        params.warmup_initial_coeff
                    )
            if i < params.esc_loss_warmup_iters:
                params.alpha = params.warmup_alpha
                loss_options = [2]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = params.warmup_lr
            else:
                params.alpha = params.opt_alpha
                loss_options = params.loss_opt
                for param_group in optimizer.param_groups:
                    param_group["lr"] = params.lr
            if params.alternate:
                if i % 2 == 0:
                    loss_options = [2]
                else:
                    loss_options = [1]
            if params.trigger_esc_eos:
                with torch.no_grad():
                    log_coeffs[:trigger_seq_length, eos_token_id] = torch.finfo(
                        log_coeffs.dtype
                    ).min
            optimizer.zero_grad(set_to_none=True)


    
            with torch.no_grad():
                if torch.isnan(log_coeffs).any():
                    print("NaNs found in log_coeffs")

                if torch.isinf(log_coeffs).any():
                    print("Infs found in log_coeffs")

                torch.nan_to_num(
                    log_coeffs, nan=0.0, posinf=1e6, neginf=-1e6, out=log_coeffs
                )
                log_coeffs.clamp_(min=-1e6, max=1e6)

                if not torch.isfinite(log_coeffs).all():
                    print("Still non-finite values after cleanup")

            coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0), hard=False)
            inputs_embeds = coeffs @ embeddings

            inputs_embeds_x = template_fac.get_input_embeddings(inputs_embeds)

            mem_use = memory_usage()
            if params.cpuoffload and mem_use > gpu_threshold : ## memory greater than 85%
                print(f"[Warning] GPU memory = {mem_use : .3f}%, offloading activations to cpu")
                with torch.autograd.graph.save_on_cpu(): ## this one move intermediate activations to CPU memory during gradient calculation. Slows down the computation though.
                    outputs = hf_model(inputs_embeds=inputs_embeds_x,  output_hidden_states=True,  return_dict=True,)
                    pred = outputs.logits
                    pred_t = pred.contiguous().view(-1, total_vocab_size)
                    target_t = F.softmax(log_coeffs, dim=1)
            else:
                outputs = hf_model(inputs_embeds=inputs_embeds_x,  output_hidden_states=True,  return_dict=True,)
                pred = outputs.logits
                pred_t = pred.contiguous().view(-1, total_vocab_size)
                target_t = F.softmax(log_coeffs, dim=1)

            eos_ids_for_loss = all_eos_token_ids if len(all_eos_token_ids) > 1 else eos_token_id
            loss_params = LossParams(
                max_length=params.max_length,
                alpha=params.alpha,
                total_vocab_size=total_vocab_size,
                eos_token_id=eos_ids_for_loss,
                eot_token_id=eot_token_id,
                esc_loss_version=params.esc_loss_version,
                loss_options=loss_options,
            )
            if params.loss_version == 1:
                loss, (loss1,loss2) = total_loss_v1(
                    pred_t,
                    target_t,
                    template_fac=template_fac,
                    need_detail=True,
                    args=loss_params,
                )
            else:
                print("Using loss version 2")
                loss = total_loss_v2(outputs, loss_params, i)
                
            gc.collect()
            # torch.cuda.empty_cache()

            losses.append(loss.item())
            # print_memory_usage("After loss calculation, before loss.backward ")
            loss.backward()
            optimizer.step()
            trigger_ids = (
                F.gumbel_softmax(log_coeffs, hard=True)
                .argmax(1)
                .tolist()[:trigger_seq_length]
            )
            if checker(trigger_ids):
                break
            if (i + 1) % params.log_interval == 0:
                torch.cuda.empty_cache()
                in_tokens = template_fac.get_input_tokens(checker.new_state)
                global_triggers.append(in_tokens)
                if log_prompts:
                    print(repr(self.model.decode_from_ids(in_tokens)))

            print(
                f"[Epoch {i}]({checker.counter}), loss:{loss.item()})"

            )

            if i == 100:
                print(f"Time Cost: {time.time() - s_time}")

            del coeffs, inputs_embeds, inputs_embeds_x, pred
            gc.collect()

            sys.stdout.flush()

        self.populate_attack_prompts(global_triggers, params, device)
        return losses
