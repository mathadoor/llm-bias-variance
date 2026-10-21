from typing import Literal, Sequence
import torch
from torch import Tensor, softmax
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor

class HFLogitProcessor(LogitsProcessor):
    def __init__(
        self,
        label_index: Sequence[int]
    ):
        self.probs = []
        self.label_index = label_index.copy()

        if label_index is None or len(label_index) == 0:
            raise ValueError("Label index must not be empty.")

    def reset(self):
        self.probs = []

    def get_probs(self):
        return torch.stack(
            self.probs,
            axis=0
        ) if len(self.probs) != 0 else Tensor([])
    
    def __call__(
        self,
        input_ids: Tensor,
        scores: Tensor
    ):
        probs = softmax(scores, axis=1)
        self.probs.append(
            probs[:, self.label_index].detach().cpu()
        )
        return scores
        


class HFGenerator:
    def __init__(
        self,
        config: dict[str, any],
        generation_config: dict[str, any],
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

        generation_config = generation_config.copy()
        generation_config["return_dict_in_generate"] = True
        store_probs = generation_config.get("store_probs")
        labels = generation_config.get("labels")

        if store_probs:
            if len(labels) == 0:
                updated_gen_config["output_probs"] = True
            else:
                updated_gen_config["output_probs"] = False
                updated_gen_config["label_index"] = [0, 1]

        self.gen_config = generation_config

        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"Incorrect tokenizer type: {type(self.tokenizer)}")
    
        if not isinstance(self.model, GenerationMixin):
            raise TypeError(f"The model cannot generate. Type: {type(self.model)}")
    
    def generate_chat(
        self,
        prompt_chat: dict[str, str] | list[dict[str, str]],
    ) -> dict[str, any]:
        tokens = self.tokenizer.apply_chat_template(prompt_chat, return_tensors="pt", padding=True, return_dict=True).to(self.device)
        with torch.no_grad():
            generated_output = self.model.generate(**tokens, **self.generation_config)

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


