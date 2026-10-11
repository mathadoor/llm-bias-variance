from torch import Tensor
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase

def validate_model_and_tokenizer(
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase
) -> bool:
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"Incorrect tokenizer type: {type(tokenizer)}")

    if not isinstance(model, GenerationMixin):
        raise TypeError(f"The model cannot generate. Type: {type(model)}")

    return True

def _generate_text(
    tokens: BatchEncoding,
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: dict[str, any]
):
    input_ids = get_attr(tokens, "input_ids", None)
    attention_mask = get_attr(tokens, "attention_mask", None)

    if input_ids or attention_mask:
        raise ValueError(f"input_ids and attention mask must be provided. got {tokens}")

    if not (isinstance(input_ids, Tensor) and isinstance(attention_mask, Tensor):
        raise TypeError(
            "input_ids and attention mask must be of type Tensor. Provided Type:\n"
             f"Input_ids: {type(input_ids)}\nAttenion Mask: {type(attention_mask)}"
        )

    generated_ids = model.generate(**tokens, **generation_config)
    return tokenizer.decode(generated_ids, skip_special_tokens=True) 

def generate_text(
    prompt: str | list[str]
    model: GenerationMixin,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: dict[str, any]
):
    validate_model_tokenizer(model, tokenizer)
    if isinstance(prompt, str):
        prompt = [prompt]

    if not isinstance(prompt, list):
        raise ValueError(f"Expected prompt type: list or str, given type: {type(prompt)}")

    tokens = tokenizer([full_prompt], return_tensors="pt")
    return _generate_text(tokens, model, generation_config)
    
def generate_chat(
    prompt_chat: dict[str, str] | list[dict[str, str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: dict[str, any]
):
    validate_model_tokenizer(model, tokenizer)
    tokens = tokenizer.apply_chat_template(prompt_chat, return_tensors="pt")
    return _generate_text(tokens, model, generation_config)
    