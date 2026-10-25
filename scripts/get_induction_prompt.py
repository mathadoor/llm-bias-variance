import yaml
import argparse
from shared import utils, hf_utils


if __name__ == "__main__":
    # load config 
    parser = argparse.ArgumentParser(
            prog="induction_prompt_extraction"
        )

    parser.add_argument("--task_config", default="run1_sentiment.yaml", required=False)
    parser.add_argument("--induction_config", default="induction_model.yaml", required=False)

    args = parser.parse_args()

    # load configs
    task_config = utils.load_config(args.task_config)
    induction_config = utils.load_config(args.induction_config)

    # loading dataset 
    
