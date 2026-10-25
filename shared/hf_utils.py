from typing import Literal, Sequence
import torch
from torch import Tensor, softmax
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import datasets

def load_hf_dataset(
    name: str,
    train_split: float, 
    seed: int
):
    if not isinstance(train_split, float):
        raise TypeError(f"Train split must be a float. got {type(train_split)}")

    if not (0 < train_split <= 1): 
        raise ValueError(f"train_split must be in (0, 1], got {train_split}")

    dataset = datasets.load_dataset(name)
    if train_split == 1:
        return {
            "train": dataset["train"],
            "val": None,
            "test": dataset["test"]
        }

    split = dataset["train"].train_test_split(train_size=train_split, seed=seed)
    return {
        "train": split['train'],
        "val": split['test'],
        "test": dataset['test']
    }

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
        # (batch, steps, labels), same layout as the output_logits path
        return torch.stack(
            self.probs,
            dim=1
        ) if len(self.probs) != 0 else Tensor([])
    
    def __call__(
        self,
        input_ids: Tensor,
        scores: Tensor
    ):
        probs = softmax(scores, axis=1)
        self.probs.append( probs[:, self.label_index].detach().cpu())
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

        del generation_config["store_probs"]
        del generation_config["labels"]


        flag = store_probs and (len(labels) == 0)
        generation_config["output_logits"] = flag 
        label_tokens = self.tokenizer(labels).input_ids if store_probs and len(labels) > 0 else []
        # labels are single characters (0-9, A-Z): each must map to exactly one token
        for label, token_ids in zip(labels, label_tokens):
            if len(token_ids) != 1:
                raise ValueError(f"Label {label!r} must tokenize to exactly one token, got {token_ids}")
        label_index = sum(label_tokens, [])
                
        self.label_index = label_index
        self.generation_config = generation_config
        self.logit_processor = LogitsProcessorList([HFLogitProcessor(label_index)]) if len(label_index) != 0 else None 

        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"Incorrect tokenizer type: {type(self.tokenizer)}")
    
        if not isinstance(self.model, GenerationMixin):
            raise TypeError(f"The model cannot generate. Type: {type(self.model)}")
    
    def generate_chat(
        self,
        prompt_chat: dict[str, str] | list[dict[str, str]],
    ) -> dict[str, any]:
        tokens = self.tokenizer.apply_chat_template(
            prompt_chat, 
            return_tensors="pt", 
            padding=True, 
            return_dict=True
        ).to(self.device)

        with torch.no_grad():
            generated_output = self.model.generate(
                **tokens, 
                **self.generation_config, 
                logits_processor=self.logit_processor
            )

        logits = getattr(generated_output, "logits", None)
        generated_ids = getattr(generated_output, "sequences", None)
        if generated_ids is None:
            raise ValueError("There was no output from generate call")
        # sequences include the prompt; left padding makes the prompt length identical across the batch
        prompt_length = tokens["input_ids"].shape[1]
        output = self.tokenizer.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True)

        return_dict = {"output_text": output, "input_tokens": tokens}

        if logits is not None:
            return_dict["logits"] = torch.softmax(torch.stack(logits, dim=1), axis=-1)
       
        if generated_ids is not None:
            return_dict["output_tokens"] = generated_ids[:, prompt_length:]
 
        if self.logit_processor is not None:
            return_dict["logits"] = self.logit_processor[0].get_probs()
            self.logit_processor[0].reset()

        return return_dict 

    def get_label_logits(
        self,
        label_header: str,
        output_dict: dict
    ) -> Tensor:
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True)
        rows = []
        for tokens, probs in zip(output_dict["output_tokens"], output_dict["logits"]):
            end = decode(tokens).index(label_header) + len(label_header)
            for i in range(len(tokens)):
                if len(decode(tokens[:i])) >= end and decode(tokens[:i + 1])[end:].strip():
                    rows.append(probs[i])
                    break
            else:
                raise ValueError(f"No token found after header {label_header!r}")
        return torch.stack(rows)