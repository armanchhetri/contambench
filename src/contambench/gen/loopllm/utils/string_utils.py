# Vendored from LoopLLM (https://github.com/neuron-insight-lab/LoopLLM), Apache-2.0; see ../LICENSE.
# Modified by the ContamBench authors: local JSON prompt files, the stationary-collapse
# helpers, and (for contambench) the research-repo-relative prompt sources removed.
import torch
import random
import json
import pandas as pd
import numpy as np
from transformers import StoppingCriteria

class RepetitionStoppingCriteria(StoppingCriteria):
    """Custom stop criterion: halt when the same token is continuously generated more than the threshold times"""
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.last_tokens = None
        self.counters = 0
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        
        # Initialization state (at the first call)
        if self.last_tokens is None:
            self.last_tokens = input_ids[:, -1].clone()
            self.counters = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
            return torch.zeros_like(self.counters, dtype=torch.bool)

        current_tokens = input_ids[:, -1]
        # Update counter: add 1 for the same position, reset to 1 for different positions
        self.counters = torch.where(current_tokens == self.last_tokens, 
                                  self.counters + 1, 
                                  torch.ones_like(self.counters))
        
        self.last_tokens = current_tokens.clone()

        return self.counters >= self.threshold


def _load_prompts_from_json(path: str, length: int) -> list:
    """Load prompts from a local JSON file.

    Accepts:
      - ``{"prompts": [{"prompt": "..."}, ...]}`` (stress_test / summary style)
      - ``["prompt", ...]`` (plain list of strings)
      - ``[{"prompt": "..."}, ...]`` (list of objects)
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw
    if isinstance(raw, dict):
        entries = raw.get("prompts", raw.get("data", []))
    if not isinstance(entries, list):
        raise ValueError(f"Unsupported prompt JSON shape in {path}")

    data = []
    for ins in entries:
        if isinstance(ins, str):
            prompt = ins
        elif isinstance(ins, dict):
            prompt = ins.get("prompt")
        else:
            continue
        if prompt is None or not str(prompt).strip():
            continue
        data.append(str(prompt))
        if len(data) >= length:
            break
    if not data:
        raise ValueError(f"No prompts found in {path}")
    return data


def read_data(dataset_name, length=100):
    data = []
    # Local JSON prompt file (absolute or relative path ending in .json)
    if isinstance(dataset_name, str) and (
        dataset_name.endswith(".json") or dataset_name.endswith(".jsonl")
    ):
        return _load_prompts_from_json(dataset_name, length)

    from datasets import load_dataset

    if dataset_name == 'alpaca':
        # https://huggingface.co/datasets/tatsu-lab/alpaca
        dataset = load_dataset("tatsu-lab/alpaca", split='train')

        dataset = dataset.shuffle()
        for ins in dataset:
            prompt = ins['instruction']
            if ins['input'].strip() != '':
                prompt += '\n' + ins['input']
            data.append(prompt)
            if len(data) >= length:
                break

    elif dataset_name == 'sharegpt':
        # https://huggingface.co/datasets/shibing624/sharegpt_gpt4
        dataset = load_dataset("shibing624/sharegpt_gpt4", split='train')

        dataset = dataset.shuffle()
        for ins in dataset:
            prompt = ins['conversations'][0]['value']
            data.append(prompt)
            if len(data) >= length:
                break

    else:
        raise NotImplementedError(
            f"Unknown prompt source {dataset_name!r}. "
            "Use alpaca|sharegpt, or a path to a .json prompt file."
        )
    
    return data[:length]


def get_chat_prompt(tokenizer, user_content, assistant_content=None, add_generation_prompt=False, is_tokenize=True, return_tensors=None, use_template=True):
    prompt = None
    if use_template:
        message = [
                {'role': 'user', 'content': user_content}
            ]
        if assistant_content is not None:
            message.append({'role': 'assistant', 'content': assistant_content})
            
        prompt = tokenizer.apply_chat_template(message, tokenize=is_tokenize, add_generation_prompt=add_generation_prompt, return_tensors=return_tensors)    #return_tensors="pt"
    else:
        prompt = user_content
        if return_tensors == 'pt':
            if assistant_content is not None:
                prompt += f' {assistant_content.strip()}'
            prompt = tokenizer(prompt, return_tensors=return_tensors).input_ids
        elif is_tokenize:
            prompt = tokenizer.encode(prompt)
        
    return prompt


def generate_str(model, tokenizer, user_prompt):

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    prompt = get_chat_prompt(tokenizer, user_prompt, add_generation_prompt=True, is_tokenize=False)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    attn_masks = torch.ones_like(input_ids)

    # stopping_criteria = [
    #     RepetitionStoppingCriteria(threshold=50)
    # ]
    stopping_criteria = None


    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                pad_token_id=pad_token_id,
                                stopping_criteria=stopping_criteria)[0]
    
    gen_str = tokenizer.decode(output_ids[input_ids.size(-1): ], skip_special_tokens=True).strip()
    
    return gen_str, len(output_ids[input_ids.size(-1): ]), len(output_ids), output_ids



@torch.no_grad
def test_suffix(model, tokenizer, prompt_ids, batch=16, sample_times=16):
    assert len(prompt_ids.shape) == 2

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    remain_times = sample_times
    len_list = []
    success_count = 0
    output_answer = []
    # stopping_criteria = [
    #     RepetitionStoppingCriteria(threshold=50)
    # ]
    stopping_criteria = None

    while remain_times > 0:
        batch_size = min(batch, remain_times)

        input_ids = prompt_ids.repeat(batch_size, 1).to(model.device)
        attention_mask = torch.ones_like(input_ids)

        out = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pad_token_id=pad_token_id,
                            stopping_criteria=stopping_criteria)

        output_answer.extend(tokenizer.batch_decode(out[:, input_ids.size(1):], skip_special_tokens=True))
        
        x = out.ne(pad_token_id).int().sum(dim=-1)
        len_list.extend(x.tolist())

        success_count += x.ge(model.generation_config.max_length-5).int().sum().item()

        remain_times -= batch_size

    answer = output_answer[np.argmax(len_list)]

    avg_len = sum(len_list) / sample_times
    success_rate = success_count / sample_times
    # is_success = success_rate >= 0.125

    return success_rate, avg_len, answer


def get_nonascii_toks(tokenizer, device='cuda'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    nonascii_toks.extend(tokenizer.all_special_ids)
    
    # record blank
    token = '* '
    s = set()
    while True:
        t = tokenizer(token, add_special_tokens=False).input_ids
        t = t[-1]
        if t not in s:
            s.add(t)
            # print(tokenizer.decode([t]), t)
            nonascii_toks.append(t)
        else:
            break
        token += ' '
 
    return torch.tensor(nonascii_toks, device=device)


def get_nonallowed_toks(tokenizer, device='cuda', max_codepoint=0x02AF):
    """Return token ids whose decoded text is outside the allowed Unicode range."""

    def is_allowed(s):
        return all(c.isprintable() and ord(c) <= max_codepoint for c in s)

    nonallowed_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_allowed(tokenizer.decode([i])):
            nonallowed_toks.append(i)

    nonallowed_toks.extend(tokenizer.all_special_ids)

    # record blank
    token = '* '
    s = set()
    while True:
        t = tokenizer(token, add_special_tokens=False).input_ids
        t = t[-1]
        if t not in s:
            s.add(t)
            nonallowed_toks.append(t)
        else:
            break
        token += ' '

    return torch.tensor(nonallowed_toks, device=device)


class SuffixManager:
    def __init__(self, tokenizer, instruction, adv_len, eos_token_id, pad_token_id, target=None):
        
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.target = target
        
        self.adv_len = adv_len
        self.adv_token_id = self.tokenizer.encode('* ' * 20)[-5]

        self.adv_suffix = self.tokenizer.decode([self.adv_token_id] * self.adv_len)

        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.init()


    def init(self):
        adv_prompt = f'{self.instruction} {self.adv_suffix.strip()}'
        prompt_ids = get_chat_prompt(self.tokenizer, adv_prompt, add_generation_prompt=True)

        prefix_len = max(index for index, item in enumerate(prompt_ids) if item == self.adv_token_id) - self.adv_len + 1

        self._control_slice = slice(prefix_len, prefix_len + self.adv_len)

        ins_len = len(self.tokenizer(self.instruction, add_special_tokens=False).input_ids)
        self._goal_slice = slice(prefix_len-ins_len, prefix_len)
        
        assert all([x == self.adv_token_id for x in prompt_ids[self._control_slice]])

        self._target_slice = slice(len(prompt_ids), None)


    def get_input_ids(self):
        adv_prompt = f'{self.instruction} {self.adv_suffix.strip()}'
        input_ids = get_chat_prompt(self.tokenizer, adv_prompt,
                                    assistant_content=self.target, return_tensors='pt')[0]
        
        return input_ids


    def update(self, adv_suffix=None, answer=None, truncation=1024):
        if adv_suffix is not None:
            self.adv_suffix = adv_suffix
        
        if answer is not None:
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) > truncation:
                answer = self.tokenizer.decode(ids[:truncation])
            self.target = answer


def normalized_lz_complexity(text: str) -> float:
    """Normalized Lempel-Ziv complexity of text (lower = more repetitive)."""
    from antropy import lziv_complexity

    if not text:
        return 0.0
    if len(text) == 1:
        return 1.0
    return float(lziv_complexity(text, normalize=True))