import os
import json
import argparse

from src.utils import normalize_name, read_jsonl
from src.datasets import VmluDataset
from src.llms import (
    VllmModel, 
    VllmModelForMultipleChoice,
    TgiModel
)

os.environ['http_proxy'] = ""
os.environ['https_proxy'] = ""


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
        choices=["vmlu", "vi_mmlu", "mmlu"],
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
        help="Do you wanna overwrite the output file or not"
    )
    
    args = parser.parse_args()
    
    return args


# print("="*56)
       

if __name__ == "__main__":
    
    args = parse_args()
    # load dataset
    if args.dataset_name=="vmlu":
        DATASET_CLASS = VmluDataset
    elif args.dataset_name=="vi_mmlu":
        pass
    elif args.dataset_name=="mmlu":
        pass
    else:
        raise ValueError(
            "`dataset_name` must take one of the following values: "
            "vmlu, vi_mmlu, mmlu"
        )
    if args.dataset_path:
        dataset = DATASET_CLASS(filename=args.dataset_path)
    else:
        default_dataset_path = f"./data/{args.dataset_name}/test.jsonl"
        dataset = DATASET_CLASS(filepath=default_dataset_path)

    if not args.output_path:
        output_folder = os.path.abspath("./output/")
        if not os.path.exists(output_folder + "/" + normalize_name(args.served_model_name)):
            os.makedirs(output_folder + "/" + normalize_name(args.served_model_name))
        output_path = output_folder + "/" + normalize_name(args.served_model_name) + f"/{args.dataset_name}_{args.engine}.jsonl"
    else:
        output_path = args.output_path
    if args.overwrite_output_file:
        f = open(output_path, "w")
        f.close()

    if args.engine == "vllm":
        if args.dataset_name in ["vmlu", "vi_mmlu", "mmlu"]:
            MODEL_CLASS = VllmModelForMultipleChoice
        else:
            MODEL_CLASS =  VllmModel
    else:
        if args.dataset_name in ["vmlu", "vi_mmlu", "mmlu"]:
            MODEL_CLASS = TgiModel
        else:
            MODEL_CLASS = TgiModel
    model = MODEL_CLASS(
        endpoint_ip = args.endpoint_ip,
        served_model_name = args.served_model_name,
        tokenizer_path=args.tokenizer_path,
        eos_token = args.eos_token,
        system_prompt = SYS_PROMPT_DICT[args.system_prompt_type], 
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
