# Engorgio (Dong et al., ICLR 2025, "An Engorgio Prompt Makes Large Language Model Babble on").
# Adapted by the ContamBench authors from https://github.com/jianshuod/Engorgio-prompt
# (commit 5fb50dd). The upstream repository carries no license file; see THIRD_PARTY_NOTICES.md.
import numpy as np
import torch
import torch.nn.functional as F
from pydantic.dataclasses import dataclass

from typing import List

from .prepare import TemplateFactory

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    cel_list = torch.mul(log_likelihood, target)
    if reduction == 'average':
        loss = torch.sum(cel_list) / batch
    else:
        loss = torch.sum(cel_list)
    return loss

def SelfMentorLoss(pred_t, target_t, args):
    loss1 = SoftCrossEntropy(pred_t, target_t, reduction='sum')
    return loss1


def TokenProbLoss(pred_t, token_id):
    normalized_pred_t = F.softmax(pred_t, dim=1)
    loss = torch.sum(normalized_pred_t[:, token_id])
    return loss


@dataclass
class LossParams:
    max_length: int
    alpha: float
    total_vocab_size: int
    eos_token_id: List[int] | int  # Can be single ID or list (for meta-llama with multiple EOS tokens)
    eot_token_id: int | None # end of thinking
    esc_loss_version: int 
    loss_options: List[int]


def total_loss_v1(pred_t, target_t, args: LossParams, template_fac: TemplateFactory, need_detail=False):
    '''
        pred_t    \in   (max_length, V)
        target_t  \in   (max_length - template_length, V)
    '''

    # reshape if not shaped, need sequence_len x vocab size
    pred_t = pred_t.view(-1, args.total_vocab_size)

    prefix_length = template_fac.prefix_length
    trigger_seq_length = template_fac.trigger_token_length
    response_length = template_fac.response_offset
    output_length = template_fac.template_w_trigger_length
    theta_length = args.max_length - template_fac.template_length

    # Self-Mentor Loss
    # Part 1 -- Trigger Part [prefix_length + 1, response_offset)
    loss1_1 = SelfMentorLoss(pred_t[prefix_length:response_length - 1], target_t[1:trigger_seq_length], args)
    # Part 2 -- Output Part
    loss1_2 = SelfMentorLoss(pred_t[output_length - 1:-1], target_t[trigger_seq_length:], args)
    loss1 = (loss1_1 + loss1_2) / theta_length

    ## calculate EOS and EOT losses
    # Handle both single EOS token and multiple EOS tokens (meta-llama case)
    if isinstance(args.eos_token_id, list):
        eos_loss = torch.tensor(0.0, device=pred_t.device)
        for eos_id in args.eos_token_id:
            eos_loss += TokenProbLoss(pred_t, eos_id)
    else:
        eos_loss = TokenProbLoss(pred_t, args.eos_token_id)

    eot_loss = 0 

    if args.eot_token_id:
        eot_loss = TokenProbLoss(pred_t, args.eot_token_id) # end of thinking
        # print(f"end of token loss = {eot_loss.item()}")

    loss2 =  (eos_loss + eot_loss)
    total_loss = loss1 + args.alpha * loss2
    
    return  total_loss, ((loss1.item(), loss2.item()) if need_detail else (0.0,0.0))

def total_loss_v2(outputs, args: LossParams, step: int):
    logits = outputs.logits.squeeze() # (L, V)
    
    # 1. Uncertainty Loss (KL Divergence from Uniform)
    log_probs = F.log_softmax(logits, dim=-1)
    vocab_size = logits.size(-1)
    target_log_prob = -np.log(vocab_size) 
    kl_diff = target_log_prob - log_probs
    loss_uncertainty = (1.0 / vocab_size) * kl_diff.sum()

    # 2. EOS Suppression Loss (MAXIMIZE LENGTH)
    # Using softmax probability instead of log_probs.
    # Minimizing the probability directly is stable and has a floor at 0.
    probs = F.softmax(logits, dim=-1)
    # If eos_token_id is a list, sum their probabilities first
    if isinstance(args.eos_token_id, list):
        loss_eos = probs[:, args.eos_token_id].sum(dim=-1).mean()
    else:
        loss_eos = probs[:, args.eos_token_id].mean()

    # 3. Diversity Loss (Nuclear Norm)
    hidden_states = outputs.hidden_states
    # Take the last layer hidden states or stack them
    hidden_states = torch.stack(hidden_states, dim=0).squeeze().float()
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    
    # Negative Nuclear Norm: maximizing variance/rank in hidden states
    loss_diversity = -torch.norm(hidden_states, p="nuc") * 1e-4

    # 4. Dynamic Weights (Safeguarded)
    # We detach to calculate the ratio without affecting the gradient flow
    l_unc_val = loss_uncertainty.detach().abs()
    l_eos_val = loss_eos.detach().abs()
    l_div_val = loss_diversity.detach().abs()

    eps = 1e-8
    # Balancing coefficients over time
    ratio1 = 10.0 * np.log(step + 1).item() - 20.0
    if abs(ratio1) < eps: ratio1 = eps
    
    ratio2 = 0.5 * np.log(step + 1).item() + 1.0
    
    # Calculate lambdas to keep loss components in similar scales
    cur_lambda1 = l_unc_val / (l_eos_val + eps) / abs(ratio1)
    cur_lambda2 = l_unc_val / (l_div_val + eps) / ratio2

    # EMA for lambda stability (Note: last_lambda should be stored outside in a real class)
    # For this standalone function, we'll use the current calculated lambda
    lambda1 = cur_lambda1 
    lambda2 = cur_lambda2

    # 5. Total Loss
    # All components are added such that MINIMIZING 'loss' does the following:
    # - Minimizes KL divergence (Uncertainty)
    # - Minimizes EOS probability (Length)
    # - Minimizes negative nuclear norm -> Maximizes rank (Diversity)
    loss = loss_uncertainty + (lambda1 * loss_eos) + (lambda2 * loss_diversity)

    return loss
