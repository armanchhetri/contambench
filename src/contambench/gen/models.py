"""Model wrapper used by both attacks.

Ported from the research repo's ``src/models.py`` (text-only path). Removed: the
vision-language branches, DeepSpeed loading, and the dependence on the global argparse
``args``; ``get_model`` takes ``quantize`` directly. Loading, chat templating, and
generation are unchanged.
"""
from __future__ import annotations

from contambench.gen import require_extra

require_extra()

import logging  # noqa: E402
import time  # noqa: E402
from typing import List  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from contambench.gen.modelsinfo import ModelInfo  # noqa: E402


class GenerationStopper(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Model:
    def __init__(self, model, tokenizer, model_info: ModelInfo, is_quantized=False):
        self.hf_model = model
        self.is_quantized = is_quantized
        self.is_deepspeed = False
        self.tokenizer = tokenizer
        self.processor = None
        self.is_vlm = False
        self.hf_model.eval()
        self.info = model_info
        self.add_gen_kwargs = {}
        if model_info.stop_ids:
            self.add_gen_kwargs["stopping_criteria"] = StoppingCriteriaList([GenerationStopper(model_info.stop_ids)])
        try:
            self.total_vocab_size = model.get_output_embeddings().out_features
        except Exception as e:
            logging.debug(f"exception on out_feature {e}")
            self.total_vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab().keys())
        self.eos_token_id = tokenizer.eos_token_id
        self.eot_token_id = model_info.eot

        # Handle multiple EOS tokens (important for meta-llama)
        self.all_eos_token_ids = []
        if hasattr(self.hf_model.config, "eos_token_id"):
            eos_cfg = self.hf_model.config.eos_token_id
            if isinstance(eos_cfg, list):
                self.all_eos_token_ids = eos_cfg
            elif isinstance(eos_cfg, int):
                self.all_eos_token_ids = [eos_cfg]
        if tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # necessary for batch generation

    def decode_from_ids(self, ids, **kwargs) -> str:
        return self.tokenizer.decode(ids, **kwargs)

    def add_chat_template(self, prompt: str, sys_prompt=None) -> str:
        """Prompt with a system message and the model's chat template (text-only path)."""
        sys_prompt = ("You are a helpful conversation AI assistant and answer to human in clear and concise way."
                      if sys_prompt is None else sys_prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except ValueError:
            return sys_prompt + "\n" + self.info.sys_prompt.roles[1] + prompt + "\n" + self.info.sys_prompt.roles[0]

    def get_name(self) -> str:
        return self.info.name

    def get_response_limit(self) -> int:
        return self.info.limit or 4096

    def set_device(self, device="cpu"):
        if self.is_quantized or self.is_deepspeed:
            return
        # Models loaded with device_map="auto" are dispatched via accelerate hooks
        # and cannot be moved with .to(device) when layers are CPU-offloaded.
        if getattr(self.hf_model, "hf_device_map", None):
            return
        self.hf_model = self.hf_model.to(device)

    def generate(self, *args, **kwargs):
        kwargs.update(self.add_gen_kwargs)
        if "pad_token_id" not in kwargs:
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        with torch.no_grad():
            kwargs["max_length"] = kwargs.get("max_length", self.get_response_limit())
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True
            start = time.perf_counter()
            output = self.hf_model.generate(*args, **kwargs)
            cost = time.perf_counter() - start
            entropies = []
            for logits in output.scores:
                probs = F.softmax(logits, dim=-1)
                entropies.append(-torch.sum(probs * torch.log(probs + 1e-9), dim=-1))
            return output.sequences, entropies, cost

    def batch_infer(self, input_texts: List[str], max_new_tokens=4096, temperature=0.7):
        self.hf_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer(input_texts, padding=True, padding_side="left", return_tensors="pt",
                                add_special_tokens=False).to(device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(
                **inputs,
                temperature=temperature,
                max_length=self.get_response_limit(),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_lens = inputs.attention_mask.sum(dim=1).tolist()
        response_len = (output_ids != self.tokenizer.pad_token_id).sum(dim=1).tolist()
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return decoded, input_lens, response_len

    def print_details(self):
        tok, cfg = self.tokenizer, self.hf_model.generation_config
        print(f"\n{'=' * 70}\nModel: {self.get_name()}\n{'=' * 70}")
        print(f"  eos={tok.eos_token_id} bos={getattr(tok, 'bos_token_id', None)} pad={tok.pad_token_id} "
              f"vocab={len(tok)} total_vocab={self.total_vocab_size}")
        print(f"  generation_config: max_length={cfg.max_length} eos={cfg.eos_token_id} pad={cfg.pad_token_id} "
              f"do_sample={cfg.do_sample} temperature={getattr(cfg, 'temperature', None)}")
        print(f"  chat_template: {getattr(tok, 'chat_template', None) is not None}  "
              f"stop_ids={self.info.stop_ids}  response_limit={self.get_response_limit()}")
        print(f"  type={type(self.hf_model).__name__} quantized={self.is_quantized} device={self.hf_model.device}")


def _uses_mxfp4(model_name: str) -> bool:
    config = AutoConfig.from_pretrained(model_name)
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    if isinstance(qc, dict):
        return qc.get("quant_method") == "mxfp4"
    return getattr(qc, "quant_method", None) == "mxfp4"


def bitsandbytes_load_kwargs(quantize: int | None) -> dict:
    """HF ``from_pretrained`` kwargs for BitsAndBytes loading (text-only models)."""
    kwargs: dict = {"torch_dtype": torch.bfloat16}
    if quantize is None:
        kwargs["device_map"] = "auto"
        return kwargs
    if quantize == 8:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == 4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="fp4")
        kwargs["device_map"] = "auto"
    else:
        raise ValueError("quantize must be None, 4, or 8")
    return kwargs


def get_model(info: ModelInfo, quantize: int | None = None) -> Model:
    model_name = info.name
    print(f"--- Loading Model: {model_name} ---")
    AutoModel = AutoModelForCausalLM
    if info.tokenizer == "LlamaTokenizer":
        from transformers import LlamaForCausalLM, LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        AutoModel = LlamaForCausalLM
    elif info.tokenizer:
        raise ValueError(f"unsupported tokenizer class {info.tokenizer!r} for {model_name}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    is_quantized = quantize is not None
    if _uses_mxfp4(model_name):
        from transformers import Mxfp4Config

        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        # GPT-OSS ships with native MXFP4 MoE weights; BitsAndBytes is incompatible.
        if quantize is None:
            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
            is_quantized = False
        else:
            load_kwargs["quantization_config"] = Mxfp4Config()
    else:
        load_kwargs = bitsandbytes_load_kwargs(quantize)

    model = AutoModel.from_pretrained(model_name, **load_kwargs)
    return Model(model, tokenizer, model_info=info, is_quantized=is_quantized)
