import yaml
import argparse
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np 
from datasets import Dataset
from shared import utils, hf_utils
import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed

class InductionPromptSignature(dspy.Signature):
    """
    You are given a list of inputs and outputs pertaining to the task of classifying the sentiment of a movie review. 
    Your task is to write instructions for an LLM to correctly perform zero-shot classification of any movie review based on the 
    patterns in these samples. Specify these patterns or signals that correlate with each classification separately. Add any additional guidelines
    to avoid pitfalls. Keep the instructions strictly contextualized to the reviews shown do not add any pattern you do not infer in the reviews shown.
    """
    selected_samples: list[dict] = dspy.InputField(desc="list of sample dictionary with sample and label. 1 for negative and 0 for positive")
    output_prompt: str = dspy.OutputField(desc="Output prompt instruction for an LLM to extract sentiment of a movie review as 0 or 1.")

def get_dspy_lm(
    config: dict      
):
    return dspy.LM(
        config.get("name"),
        **config.get("params", {})
    )

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

def generate_induction_prompts(
    runs: dict,
    lm: dspy.LM
):
    module = dspy.ChainOfThought(InductionPromptSignature)

    def run(samples):
        with dspy.context(lm=lm):
            return module(selected_samples=samples)

    futures = {} 
    with ThreadPoolExecutor(max_workers=32) as executor:
        for train_size_run in runs:
            train_size = train_size_run["train_size"]
            futures.update({
                executor.submit(run, list(dataset)): (train_size, list(dataset))
                for dataset in train_size_run["dataset"].T
            })

    results = []
    for future in as_completed(futures):
        props = futures[future]
        result = future.result()
        results.append({
            "train_size": props[0],
            "dataset": props[1],
            "results": result
        })

    return results

def format_samples(samples: list[dict]) -> str:
    blocks = []
    for i, sample in enumerate(samples, start=1):
        lines = [f"--- sample {i} ---"]
        lines += [textwrap.fill(f"{key}: {value}", width=100) for key, value in sample.items()]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"

def save_prompts(
    results: list[dict],
    output_dir: Path
) -> Path:
    # <output_dir>/<date_time>/<train_size>/<run_index>/{prompts,selected_samples}.txt
    root = Path(output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root.mkdir(parents=True, exist_ok=True)

    # run index is just a counter per train size, any run may get any index
    run_counts = Counter()
    for result in results:
        run_index = run_counts[result["train_size"]]
        run_counts[result["train_size"]] += 1
        run_dir = root / str(result["train_size"]) / str(run_index)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "prompts.txt").write_text(result["results"].output_prompt + "\n", encoding="utf-8")
        (run_dir / "selected_samples.txt").write_text(format_samples(result["dataset"]), encoding="utf-8")

    return root

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

    lm = get_dspy_lm(induction_config)
    prompts = generate_induction_prompts(run_datasets, lm)

    # save prompts
    output_path = save_prompts(prompts, Path(__file__).parent.parent / "output")
    print(f"Saved {len(prompts)} prompts to {output_path}")
