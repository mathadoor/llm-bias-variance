from typing import Literal
import torch
from torch import Tensor
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase

class HFGenerator:
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizerBase,
        device: Literal["cpu", "cuda", "mps"] 
    ):
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"Incorrect tokenizer type: {type(tokenizer)}")
    
        if not isinstance(model, GenerationMixin):
            raise TypeError(f"The model cannot generate. Type: {type(model)}")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _generate_text(
        self,
        tokens: BatchEncoding,
        generation_config: dict[str, any]
    ):    
        with torch.no_grad():
            generated_ids = self.model.generate(**tokens, **generation_config)
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True) 

    def generate_text(
        self,
        prompt: str | list[str],
        generation_config: dict[str, any],
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
    
        if not isinstance(prompt, list):
            raise ValueError(f"Expected prompt type: list or str, given type: {type(prompt)}")
    
        tokens = self.tokenizer([full_prompt], return_tensors="pt", padding=True).to(device)
        return self._generate_text(tokens, generation_config)
    
    def generate_chat(
        prompt_chat: dict[str, str] | list[dict[str, str]],
        generation_config: dict[str, any],
    ):
        tokens = self.tokenizer.apply_chat_template(prompt_chat, return_tensors="pt", padding=True).to(device)
        return self._generate_text(tokens, generation_config)
    