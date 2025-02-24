import os
import json
import argparse

from src.utils import normalize_name
from src.datasets import ViMCQDataset
from src.llms import (
    VllmModel, 
    VllmModelForMultipleChoice,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--endpoint_ip",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--served_model_name",
        type=str,
        default=None,
        help="Only neccesary when serving with vllm"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="absolute path to your model directory"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "tgi"],
        help="`tgi` or `vllm`"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="path of the dataset you want to evaluate"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vmlu",
        choices=["vmlu", "vi_mmlu"],
        help="name of the dataset you want to evaluate"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="the path to the output file"
    )
    parser.add_argument( 
        "--overwrite_output_file",
        action='store_true',
        help="Do you wanna overwrite the output file?"
    )
    
    args = parser.parse_args()
    
    return args
       

if __name__ == "__main__":
    
    args = parse_args()
    # load dataset
    if args.dataset_name in ["vmlu", "vi_mmlu"]:
        DATASET_CLASS = ViMCQDataset
    else:
        raise ValueError(
            "`dataset_name` must take one of the following values: "
            "vmlu, vi_mmlu"
        )
    if args.dataset_path:
        dataset = DATASET_CLASS(args.dataset_path)
    else:
        default_dataset_path = f"./data/{args.dataset_name}/test.jsonl"
        dataset = DATASET_CLASS(default_dataset_path)

    if not args.output_path:
        output_folder = os.path.abspath("./output/") + "/" + normalize_name(args.served_model_name)
        output_path = output_folder + f"/{args.dataset_name}_{args.engine}.jsonl"
    else:
        output_path = args.output_path
        output_folder = "/".join(output_path.split("/")[:-1])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.engine == "vllm":
        if args.dataset_name in ["vmlu", "vi_mmlu"]:
            MODEL_CLASS = VllmModelForMultipleChoice
        else:
            MODEL_CLASS =  VllmModel
    elif args.engine == "tgi":
        if args.dataset_name in ["vmlu", "vi_mmlu"]:
            MODEL_CLASS = TgiModel
        else:
            MODEL_CLASS = TgiModel
    else:
        raise ValueError(
            "`engine` must take one of the following values: "
            "tgi, vllm"
        )
    model = MODEL_CLASS(
        endpoint_ip = args.endpoint_ip,
        served_model_name = args.served_model_name,
        tokenizer_path=args.tokenizer_path,
    )
    infered_dataset = []
    if os.path.isfile(output_path):
        infered_dataset = read_jsonl(output_path)
    for batch in dataset[len(infered_dataset):].batch_iter(batch_size=128):
        sys_answers, final_choices = model(
            batch.to_user_prompts(),
            # n=10,
            # best_of=10,
            # temperature=0.6
        )
        with open(output_path, "a") as f:
            for sample, answer_candidates, final_choice_candidates in zip(batch, sys_answers, final_choices):
                f.write(json.dumps({
                    "explanation": answer_candidates,
                    "final_choice": final_choice_candidates,
                    **sample
                }) + "\n")
