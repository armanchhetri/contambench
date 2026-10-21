# Vendored from LoopLLM (https://github.com/neuron-insight-lab/LoopLLM), Apache-2.0; see ../LICENSE.
# Modified by the ContamBench authors (loss configurations, import paths).
import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .collapse_losses import (
    DEFAULT_LOSS_CONFIG,
    LossConfig,
    compute_attack_loss,
    get_loss_config,
)

# from attack_manager import get_embedding_matrix, get_embeddings


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='auto', load_in_8bit=True, **kwargs):
    if 'phi' in model_path.lower():
        kwargs['trust_remote_code'] = True

    if load_in_8bit and device != 'cpu' and 'quantization_config' not in kwargs:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

    print(model_path)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            **kwargs
        ).eval()
    
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        # use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id if model.generation_config.pad_token_id is None else model.generation_config.pad_token_id

    if not model.generation_config.do_sample:
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.6
        model.generation_config.top_p = 0.9
        
    # greedy decode
    # model.generation_config.do_sample = False
    # model.generation_config.temperature = 1
    # model.generation_config.top_p = 1
    # model.generation_config.top_k = None
        
    return model, tokenizer


def get_gradients(model, input_ids, suffix_manager, loss_config: LossConfig | str | None = None):
    control_slice = suffix_manager._control_slice

    # embed_weights = get_embedding_matrix(model)
    embed_weights = model.get_input_embeddings().weight
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    
    one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights)
    
    embeds = embed_weights[input_ids].detach()
    full_embeds = torch.cat(
        [
            embeds[:control_slice.start, :], 
            input_embeds, 
            embeds[control_slice.stop:, :]
        ], 
        dim=0).unsqueeze(0)
    
    logits = model(inputs_embeds=full_embeds).logits
    print(f"Logits.shape = {logits.shape}")

    if loss_config is None:
        loss_config = get_loss_config(DEFAULT_LOSS_CONFIG)
    elif isinstance(loss_config, str):
        loss_config = get_loss_config(loss_config)

    loss = compute_attack_loss(
        logits, suffix_manager, loss_config, per_sample=False
    )
    del logits
    torch.cuda.empty_cache()

    model.zero_grad()
    loss.backward(retain_graph=False)

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    one_hot.grad.zero_()

    return grad


def model_forward(
    model, input_ids, suffix_manager, batch_size=32, loss_config: LossConfig | str | None = None
):
    if loss_config is None:
        loss_config = get_loss_config(DEFAULT_LOSS_CONFIG)
    elif isinstance(loss_config, str):
        loss_config = get_loss_config(loss_config)

    losses = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        logits = model(input_ids=batch_input_ids).logits
        loss = compute_attack_loss(
            logits, suffix_manager, loss_config, per_sample=True
        )
        # batch_loss = out["loss"] if isinstance(out, dict) else out
        losses.append(loss)

    del batch_input_ids, logits
    gc.collect()
    torch.cuda.empty_cache()

    return torch.cat(losses, dim=0)


def get_all_losses(
    model,
    tokenizer,
    input_ids,
    new_adv_suffixs,
    suffix_manager,
    batch_size=32,
    loss_config: LossConfig | str | None = None,
):
    control_slice = suffix_manager._control_slice

    if isinstance(new_adv_suffixs[0], str):
        max_len = control_slice.stop - control_slice.start
        
        test_ids = tokenizer(new_adv_suffixs, add_special_tokens=False, max_length=max_len,
                              padding=True, truncation=True, return_tensors="pt").input_ids.to(model.device)
        
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(new_adv_suffixs)}")

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )

    return model_forward(
        model, ids, suffix_manager, batch_size=batch_size, loss_config=loss_config
    )



def sample_control(control_toks, grad, batch_size, topk, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens] = np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    
    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    new_control_toks = new_control_toks.unique(dim=0)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, adv_suffix_tokens, fill_cand=False):
    cands, count = [], 0
    s = set()
    length = len(adv_suffix_tokens)
    curr_control = tokenizer.decode(adv_suffix_tokens, skip_special_tokens=True)

    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        
        if decoded_str != curr_control and \
            len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == length and  \
            decoded_str not in s:
                cands.append(decoded_str)
                s.add(decoded_str)
        else:
            count += 1
    if len(cands) == 0:
        raise Exception('get_filtered_cands() get zero candidate')

    if fill_cand and count > 0:
        cands = cands + [cands[-1]] * count
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
        
    return cands

def is_entropy_low(model, input_ids, threshold=0.1):
    # Calculate the output entropy
    with torch.no_grad():
        logits = model(input_ids).logits
        logits_ans = logits[0, -100:, :]
        probs = torch.softmax(logits_ans, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        ans_entropy = entropy.mean(dim=-1).item()
    return ans_entropy < threshold