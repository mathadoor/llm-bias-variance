import yaml
import argparse
import numpy as np 
from datasets import Dataset
from shared import utils, hf_utils


def create_runs(
    train_dataset: Dataset,
    train_sizes: list[int],
    num_runs_per_size: int,
    seed: int
):
    rng = np.random.default_rng(seed)
    dataset_runs = []

    for train_size in train_sizes:
        datasets = rng.choice(train_dataset, size=(train_size, num_runs_per_size))
        dataset_runs.append({
            "train_size": train_size,
            "dataset": datasets
        })

    return dataset_runs

if __name__ == "__main__":
    # load config 
    parser = argparse.ArgumentParser(
            prog="induction_prompt_extraction"
        )

    parser.add_argument("--task_config", default="run1_sentiment.yaml", required=False)
    parser.add_argument("--induction_config", default="induction_model.yaml", required=False)
    parser.add_argument("--seed", default=0, required=False)
    parser.add_argument("--train_split", default=0.7, required=False)

    args = parser.parse_args()

    # load configs
    task_config = utils.load_config(args.task_config)
    induction_config = utils.load_config(args.induction_config)

    # loading dataset 
    seed = task_config.get("seed", args.seed)
    dataset_config = task_config.get("dataset", {})
    dataset_name = dataset_config.get("name")
    train_split = dataset_config.get("train_split", args.train_split)

    hf_datasets = hf_utils.load_hf_dataset(dataset_name, train_split, seed) 

    run_datasets = create_runs(
        train_dataset=hf_datasets["train"],
        train_sizes=task_config["train"]["num_samples"],
        num_runs_per_size=task_config["train"]["num_runs_per_size"],
        seed=seed
    ) 
