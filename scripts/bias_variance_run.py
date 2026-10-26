import argparse
from shared import utils, hf_utils

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
