from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

def generate(
    tokens: dict[str, Tensor] | dict[str | list],
    model: PreTrainedModel
):
    raise NotImplementedError

def generate_text(
    prompt: str
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    raise NotImplementedError

def generate_chat(
    prompt_chat: dict[str, str],
    model: PreTrainedModel,
    tokenizer: PreTraiendTokenizer
):
    raise NotImplementedError