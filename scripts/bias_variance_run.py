import argparse
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from shared import utils, hf_utils

ANSWER_HEADER = "### output"
ANSWER_HEADERS = [ANSWER_HEADER, "</think>"]

def evaluate_prompt(
    induced_prompt: str,
    hf_generator: hf_utils.HFGenerator,
    hf_dataset: Dataset,
    config: dict
) -> dict:
    # Validation
    val_batch_dataset = hf_dataset.batch(
        batch_size=config.get("validation", {}).get("batch_size", 32),
        drop_last_batch=False,
    )
    labels = config["generation"]["labels"]
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

            # only samples with a label token right after a header have a prediction and label probabilities
            answers = hf_generator.get_answer_probs(ANSWER_HEADERS, output_dict)

            # merge
            ground_truth.extend(batch['label'])
            completed_prompt.extend(full_text)
            answered.extend(answer is not None for answer in answers)
            predictions.extend(
                None if answer is None else utils.coerce_numeric(labels[answer[1]])
                for answer in answers
            )
            probs.extend(None if answer is None else answer[2] for answer in answers)

    return dict(
        completed_prompt=completed_prompt,
        ground_truth=ground_truth,
        predictions=predictions,
        answered=np.array(answered),
        probs=probs
    )

def run_prompt(
    induced_prompts: list[str],
    num_k: int,
    hf_generator: hf_utils.HFGenerator,
    hf_dataset: Dataset,
    config: dict
):
    with mlflow.start_run():

        # tokenize and generate
        mlflow.log_params(dict(num_samples=num_k, num_runs=len(induced_prompts), **config["model"], **config.get("generation", {})))

        # log prompts
        for i, induced_prompt in enumerate(induced_prompts):
            mlflow.log_text(induced_prompt, f"induced_prompt_{i}.txt")

        runs = [evaluate_prompt(induced_prompt, hf_generator, hf_dataset, config) for induced_prompt in induced_prompts]

        # (samples, runs): same samples in every run, so the rows line up
        answered = np.stack([run["answered"] for run in runs], axis=1)
        ground_truth = np.array(runs[0]["ground_truth"])
        mlflow.log_metric("num_val_samples", len(ground_truth))
        mlflow.log_metric("failure_rate", float(1 - answered.mean()))

        # accuracy per run, on the samples that run answered
        accuracies = []
        for i, run in enumerate(runs):
            mlflow.log_metric(f"failure_rate_run{i}", float(1 - answered[:, i].mean()))
            if answered[:, i].any():
                valid_predictions = np.array([p for p in run["predictions"] if p is not None])
                accuracies.append(float((ground_truth[answered[:, i]] == valid_predictions).mean()))
                mlflow.log_metric(f"accuracy_run{i}", accuracies[-1])
        if accuracies:
            mlflow.log_metric("accuracy", float(np.mean(accuracies)))

        # the decomposition needs a full row of probabilities, so only samples answered in every run
        complete = answered.all(axis=1)
        mlflow.log_metric("num_complete_samples", int(complete.sum()))
        mlflow.log_metric("over_all_failure_rate", float(1- complete.mean()))
        if complete.any():
            rows = np.flatnonzero(complete)
            label_probs = torch.stack([torch.stack([run["probs"][row] for run in runs]) for row in rows])
            renormalized_probs = label_probs / label_probs.sum(dim=-1, keepdim=True)

            for suffix, label_prob in (("", label_probs), ("_renorm", renormalized_probs)):
                var, bias = utils.compute_brier_decomposition(label_prob[:, :, -1], ground_truth[complete].tolist())
                mlflow.log_metric(f"brier_score{suffix}", var + bias)
                mlflow.log_metric(f"brier_score_var{suffix}", var)
                mlflow.log_metric(f"brier_score_bias{suffix}", bias)

        # Sample traces, stratified on run and correctness
        results = pd.concat([
            pd.DataFrame(dict(
                run=i,
                completed_prompt=run["completed_prompt"],
                ground_truth=run["ground_truth"],
                prediction=run["predictions"],
                has_answer=run["answered"],
                label_probs=[None if p is None else p.tolist() for p in run["probs"]],
                renormalized_label_probs=[None if p is None else (p / p.sum()).tolist() for p in run["probs"]],
            ))
            for i, run in enumerate(runs)
        ], ignore_index=True)
        results["correct"] = results["has_answer"] & (results["ground_truth"] == results["prediction"])
        n_traces = config.get("validation", {}).get("num_traces", 5)
        traces = results.groupby(["run", "correct"], group_keys=False).apply(
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
        run_prompt(induced_prompts, num_k, hf_generator, hf_datasets["val"], task_config)
