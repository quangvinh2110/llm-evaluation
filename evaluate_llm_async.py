import os
import json
import argparse

from src.utils import normalize_name
from src.datasets import VmluDataset
from src.llms import (
    VllmModel, 
    VllmModelForMultipleChoice,
    TgiModel
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
        "--eos_token",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--system_prompt_type",
        type=str,
        default="empty",
        help="",
        choices=["empty", "vistral", "vinbigdata", "mixsura", "vinallama", "seallm", "llama3"],
    )
    # parser.add_argument( 
    #     "--compute_final_score",
    #     action='store_true',
    #     help="Do you wanna compute the final score or you just need answers from your LLMs"
    # )
    
    args = parser.parse_args()
    
    return args



SYS_PROMPT_DICT = {
    "empty": "",
    "vistral": "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.",
    "vinbigdata": "A chat between a curious user and an artificial intelligence assistant. The assistant think carefully, then intergrate step-by-step reasoning to answer the question.",
    "mixsura": "Bạn là một trợ lý thông minh. Hãy thực hiện các yêu cầu hoặc trả lời câu hỏi từ người dùng bằng tiếng Việt.",
    "vinallama": "Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.",
    "seallm": "You are a helpful assistant.",
}


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
        dataset = DATASET_CLASS(args.dataset_path)
    else:
        default_dataset_path = f"./data/{args.dataset_name}/test.jsonl"
        dataset = DATASET_CLASS(default_dataset_path)

    if not args.output_path:
        output_folder = os.path.abspath("./output/")
        if not os.path.exists(output_folder + "/" + normalize_name(args.served_model_name)):
            os.makedirs(output_folder + "/" + normalize_name(args.served_model_name))
        output_path = output_folder + "/" + normalize_name(args.served_model_name) + f"/{args.dataset_name}_{args.engine}.jsonl"
    else:
        output_path = args.output_path

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
    sys_answers, final_choices = model(dataset.to_user_prompts())
    # print(dataset[0])
    with open(output_path, "w") as f:
        for sample, answer_candidates, final_choice_candidates in zip(dataset, sys_answers, final_choices):
            f.write(json.dumps({
                "explanation": answer_candidates,
                "final_choice": final_choice_candidates,
                **sample
            }) + "\n")
