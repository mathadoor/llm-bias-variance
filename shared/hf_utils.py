from typing import Literal
import torch
from torch import Tensor
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM

class HFGenerator:
    def __init__(
        self,
        config: dict[str, any]
    ):
        self.device = config["device"]
        self.model = AutoModelForCausalLM.from_pretrained(
            config["name"],
            **config["model"]
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["name"],
            **config["tokenizer"]
        )

        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"Incorrect tokenizer type: {type(self.tokenizer)}")
    
        if not isinstance(self.model, GenerationMixin):
            raise TypeError(f"The model cannot generate. Type: {type(self.model)}")
    
    def generate_chat(
        self,
        prompt_chat: dict[str, str] | list[dict[str, str]],
        generation_config: dict[str, any],
    ) -> dict[str, any]:
        generation_config = generation_config.copy()
        generation_config["return_dict_in_generate"] = True
        tokens = self.tokenizer.apply_chat_template(prompt_chat, return_tensors="pt", padding=True, return_dict=True).to(self.device)
        with torch.no_grad():
            generated_output = self.model.generate(**tokens, **generation_config)

        logits = getattr(generated_output, "logits", None)
        generated_ids = getattr(generated_output, "sequences", None)
        if generated_ids is None:
            raise ValueError("There was no output from generate call")
        output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return_dict = {"output_text": output, "input_tokens": tokens}

        if logits is not None:
            return_dict["logits"] = logits

        if generated_ids is not None:
            return_dict["output_tokens"] = generated_ids

        return return_dict 
