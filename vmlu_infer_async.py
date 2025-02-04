import os
import json
import asyncio
import argparse

from tqdm.asyncio import tqdm

from datasets import load_from_disk, Dataset

from src.utils import normalize_name
from src.llms import VllmModel, TgiModel


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
        "--model_name",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "tgi"],
        help="`tgi` or `vllm`"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=4096,
        help=""
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=3072,
        help=""
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
    
    args = parser.parse_args()
    
    return args


print("="*56)
print("="*20+" VMLU VAL INFER TGI "+"="*20)
print("="*56)


SYS_PROMPT_DICT = {
    "empty": "",
    "vistral": "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.",
    "vinbigdata": "A chat between a curious user and an artificial intelligence assistant. The assistant think carefully, then intergrate step-by-step reasoning to answer the question.",
    "mixsura": "Bạn là một trợ lý thông minh. Hãy thực hiện các yêu cầu hoặc trả lời câu hỏi từ người dùng bằng tiếng Việt.",
    "vinallama": "Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.",
    "seallm": "You are a helpful assistant.",
}


# Hãy trả lời câu hỏi trắc nghiệm sau và đưa câu trả lời cuối cùng của bạn vào trong \boxed{{}}: # for 70B models
# Hãy trả lời câu hỏi trắc nghiệm sau:
# Đưa ra câu trả lời cuối cùng dưói dạng JSON. VD: {{"final_choice": "A"}}
USER_PROMPT_TEMPLATE = r"""
Hãy trả lời câu hỏi trắc nghiệm sau. Đưa ra câu trả lời cuối cùng dưói dạng JSON. VD: {{"final_choice": A}}
{question}
{choices}
Lưu ý: Câu hỏi chỉ có một đáp án duy nhất.
""".strip()

COT_PROMPT = """
Trước hết hãy phân tích câu hỏi một cách cẩn thận và suy luận từng bước một.
""".strip()


print("="*56)


class VllmModelForMMLU(VllmModel):
    def extract_sample(self, sample):
        question = sample["question"]
        choices = sample["choices"]
        return {
            "question": question, 
            "choices": "\n".join(choices), 
        }
    
    
class TgiModelForMMLU(TgiModel):
    def extract_sample(self, sample):
        question = sample["question"]
        choices = sample["choices"]
        return {
            "question": question, 
            "choices": "\n".join(choices), 
        }
       

if __name__ == "__main__":
    
    args = parse_args()
    data_hub = "/workspace/home/NLP_CORE/evaluation/benchmark_eval_v3/data"
    data_name = "vmlu_valid_full"
    data_path = data_hub + f"/{data_name}"
    dataset = load_from_disk(data_path)
    output_folder = "/workspace/home/NLP_CORE/evaluation/benchmark_eval_v3/output/"
    # output_folder = "/workspace/NLP_CORE/evaluation/benchmark_eval/output/"
    output_file = output_folder + normalize_name(args.model_name) + f"/{data_name}_tgi.jsonl"
    if not os.path.exists(output_folder + normalize_name(args.model_name)):
        os.makedirs(output_folder + normalize_name(args.model_name))
    if args.engine == "vllm":
        modelformmlu = VllmModelForMMLU
    else:
        modelformmlu = TgiModelForMMLU
    model = modelformmlu(
        endpoint_ip = args.endpoint_ip,
        model_name = args.model_name,
        eos_token = args.eos_token,
        system_prompt = SYS_PROMPT_DICT[args.system_prompt_type], 
        user_prompt_template = USER_PROMPT_TEMPLATE, 
        assistant_prompt_prefix = COT_PROMPT
    )
    asyncio.run(model.generate_async(
        dataset=dataset,
        output_field="cot_answer",
        output_file=output_file
    ))
