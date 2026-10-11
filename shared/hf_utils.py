from torch import Tensor
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase

def generate_text(
    tokens: BatchEncoding,
    model: GenerationMixin,
    generation_config: dict[str, any]
):
    input_ids = get_attr(tokens, "input_ids", None)
    attention_mask = get_attr(tokens, "attention_mask", None)

    if input_ids or attention_mask:
        raise ValueError(f"input_ids and attention mask must be provided. got {tokens}")

    if not (isinstance(input_ids, Tensor) and isinstance(attention_mask, Tensor):
        raise ValueError(
            "input_ids and attention mask must be of type Tensor. Provided Type:\n"
             f"Input_ids: {type(input_ids)}\nAttenion Mask: {type(attention_mask)}"
        )

    return model.generate(**tokens, **generation_config)

def generate_text(
    prompt: str
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: dict[str, any]
):
    raise NotImplementedError

def generate_chat(
    prompt_chat: dict[str, str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: dict[str, any]
):
    raise NotImplementedError