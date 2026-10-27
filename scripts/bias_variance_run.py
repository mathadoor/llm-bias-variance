import argparse
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from shared import utils, hf_utils

ANSWER_HEADER = "### output"

def has_answer(text: str) -> bool:
    # the header must be present with something after it, otherwise there is no answer token to read
    return ANSWER_HEADER in text and text.split(ANSWER_HEADER)[-1].strip() != ""

def run_prompt(
    induced_prompt: str,
    num_k: int,
    hf_generator: hf_utils.HFGenerator,
    hf_dataset: Dataset,
    config: dict
):
    with mlflow.start_run():
        
        # tokenize and generate
        mlflow.log_params(dict(num_samples=num_k, **config["model"], **config.get("generation", {})))
    
        # log prompts
        mlflow.log_text(induced_prompt, "induced_prompt.txt")
        
        # Validation
        val_batch_dataset = hf_dataset.batch(
            batch_size=config.get("validation", {}).get("batch_size", 32),
            drop_last_batch=False,
        )
        with torch.no_grad():
            completed_prompt = []
            ground_truth = []
            predictions = []
            answered = []
            probs = []
            for batch in tqdm(val_batch_dataset):
                # Prepare inputs
                reviews = [
                    utils.prep_messages(f"{induced_prompt}\nYou must answer question in this format. '{ANSWER_HEADER}\n[sentiment label]'", sample)
                    for sample in batch['review']
                ]
                output_dict = hf_generator.generate_chat(reviews)
                full_text = output_dict["output_text"]

                # only samples where the header was produced have a prediction and label probabilities
                batch_answered = [has_answer(x) for x in full_text]
                valid = [i for i, ok in enumerate(batch_answered) if ok]
                curr_probs = iter(
                    hf_generator.get_label_logits(
                        ANSWER_HEADER,
                        {key: output_dict[key][valid] for key in ("output_tokens", "logits")}
                    )
                    if valid else []
                )

                # merge
                ground_truth.extend(batch['label'])
                completed_prompt.extend(full_text)
                answered.extend(batch_answered)
                predictions.extend(
                    utils.coerce_numeric(x.split(ANSWER_HEADER)[-1].strip()) if ok else None
                    for x, ok in zip(full_text, batch_answered)
                )
                probs.extend(next(curr_probs) if ok else None for ok in batch_answered)

        answered = np.array(answered)
        mlflow.log_metric("failure_rate", float(1 - answered.mean()))

        if answered.any():
            valid_truth = np.array(ground_truth)[answered]
            valid_predictions = np.array([p for p in predictions if p is not None])
            mlflow.log_metric("accuracy", float((valid_truth == valid_predictions).mean()))

            # renormalize over the label tokens, then use the probability of the positive label (last one)
            label_probs = torch.stack([p for p in probs if p is not None])
            renormalized_probs = label_probs / label_probs.sum(dim=-1, keepdim=True)
            for suffix, label_prob in (("", label_probs), ("_renorm", renormalized_probs)):
                var, bias = utils.compute_brier_decomposition(label_prob[:, -1:], valid_truth.tolist())
                mlflow.log_metric(f"brier_score{suffix}", var + bias)
                mlflow.log_metric(f"brier_score_var{suffix}", var)
                mlflow.log_metric(f"brier_score_bias{suffix}", bias)

        # Sample traces, stratified on correctness
        results = pd.DataFrame(dict(completed_prompt=completed_prompt, ground_truth=ground_truth, prediction=predictions))
        results["has_answer"] = answered
        results["label_probs"] = [None if p is None else p.tolist() for p in probs]
        results["renormalized_label_probs"] = [None if p is None else (p / p.sum()).tolist() for p in probs]
        results["correct"] = results["has_answer"] & (results["ground_truth"] == results["prediction"])
        n_traces = config.get("validation", {}).get("num_traces", 5)
        traces = results.groupby("correct", group_keys=False).apply(
            lambda g: g.sample(min(n_traces, len(g)), random_state=config.get("seed", 0))
        )
        mlflow.log_table(traces, "validation_traces.json")

        # full validation output, so renormalization etc. can be tested later
        mlflow.log_table(results, "validation_results.json")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("bias variance run")

    parser.add_argument("--task_config", default="run1_sentiment.yaml", required=False)
    parser.add_argument("--seed", default=0, required=False)
    parser.add_argument("--train_split", default=0.7, required=False)
    parser.add_argument("--prompt_dir", default="induced_prompts", required=False)

    args = parser.parse_args()

    # load configs
    task_config = utils.load_config(args.task_config)

    # loading dataset 
    seed = task_config.get("seed", args.seed)
    dataset_config = task_config.get("dataset", {})
    dataset_name = dataset_config.get("name")
    train_split = dataset_config.get("train_split", args.train_split)

    hf_datasets = hf_utils.load_hf_dataset(dataset_name, train_split, seed) 

    # get prompt
    prompts = utils.load_prompts(args.prompt_dir)

    # HF generator
    hf_generator = hf_utils.HFGenerator(task_config["model"], task_config["generation"])

    # run every induced prompt on the validation split
    mlflow_config = task_config.get("mlflow", {})
    if mlflow_config.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config.get("experiment", "llm-bias-variance"))

    for num_k, induced_prompts in prompts.items():
        for induced_prompt in induced_prompts:
            run_prompt(induced_prompt, num_k, hf_generator, hf_datasets["val"], task_config)
