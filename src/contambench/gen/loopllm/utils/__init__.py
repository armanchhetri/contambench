# Vendored from LoopLLM (https://github.com/neuron-insight-lab/LoopLLM), Apache-2.0; see ../LICENSE.
# Modified by the ContamBench authors (loss configurations, import paths).
from contambench.gen import require_extra

require_extra()


from .opt_utils import load_model_and_tokenizer
from .collapse_losses import (
    DEFAULT_CYCLIC_SEGMENT_LEN,
    DEFAULT_LOSS_CONFIG,
    LOSS_CONFIGS,
    compute_attack_loss,
    cycle_loss,
    get_loss_config,
    resolve_loss_config,
)
from .opt_utils import get_gradients
from .opt_utils import get_all_losses
from .opt_utils import sample_control
from .opt_utils import get_filtered_cands
from .opt_utils import is_entropy_low

from .string_utils import read_data
from .string_utils import get_chat_prompt
from .string_utils import generate_str
from .string_utils import get_nonascii_toks
from .string_utils import get_nonallowed_toks
from .string_utils import test_suffix
from .string_utils import SuffixManager
from .string_utils import normalized_lz_complexity

__all__ = [
    "DEFAULT_CYCLIC_SEGMENT_LEN",
    "DEFAULT_LOSS_CONFIG",
    "LOSS_CONFIGS",
    "MODEL_PATHS",
    "SuffixManager",
    "compute_attack_loss",
    "cycle_loss",
    "generate_str",
    "get_all_losses",
    "get_chat_prompt",
    "get_filtered_cands",
    "get_gradients",
    "get_loss_config",
    "resolve_loss_config",
    "get_nonallowed_toks",
    "get_nonascii_toks",
    "is_entropy_low",
    "load_model_and_tokenizer",
    "normalized_lz_complexity",
    "read_data",
    "sample_control",
    "test_suffix",
]

MODEL_PATHS = {
    'llama2-13b': 'Your/Model/Path/',  # meta-llama/Llama-2-13b-chat-hf
    'glm4-9b': '',    #THUDM/glm-4-9b-chat-hf
    'llama2-7b': '',     # meta-llama/Llama-2-7b-chat-hf
    'llama3-8b': '',   # meta-llama/Llama-3.1-8B-Instruct
    'mistral-7b': '',      # mistralai/Mistral-7B-Instruct-v0.3
    'falcon3-7b': '',     #tiinae/Falcon3-7B-Instruct
    'vicuna-7b': '',   #lmsys/vicuna-7b-v1.5
    'phi4-mini': '',   #microsoft/Phi-3.5-mini-instruct
    'llama3-3b': 'meta-llama/Llama-3.2-3B-Instruct', #meta-llama/Llama-3.2-3B-Instruct
    'qwen2.5-3b': '',  #Qwen/Qwen2.5-3B-Instruct
    'stablelm-3b': '', #stabilityai/stablelm-zephyr-3b
    'gemma2-2b': '',     #google/gemma-2-2b-it
    'llama3-1b': 'meta-llama/Llama-3.2-1B-Instruct',    #meta-llama/Llama-3.2-1B-Instruct
}