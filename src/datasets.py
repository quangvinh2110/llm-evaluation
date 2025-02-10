import json

class Dataset:
    def __init__(self, filepath: str):
        self.data = []
        if filepath.endswith(".jsonl"):
            with open(filepath) as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif filepath.endswith(".json"):
            with open(filepath) as f:
                self.data = json.load(f)
        elif filepath.endswith(".parquet"):
            import pandas as pd
            self.data = pd.read_parquet(filepath).to_dict('records')
        elif filepath.endswith(".csv"):
            import pandas as pd
            data = pd.read_csv(filepath).to_dict('records')  

    @classmethod
    def process(self, sample):
        pass


    def to_user_prompts(self):
        return [self.process(s) for s in self.data]
    

    def __iter__(self):
        for sample in self.data:
            yield sample


class VmluDataset(Dataset):

    USER_PROMPT_TEMPLATE = r"""
Hãy suy luận từng bước để trả lời câu hỏi trắc nghiệm sau. Đưa ra câu trả lời cuối cùng dưói dạng JSON. VD: {{"final_choice": "A"}}
{question}
{choices}
Lưu ý: Câu hỏi chỉ có một đáp án duy nhất.
""".strip()
    
    def process(self, sample):
        return self.USER_PROMPT_TEMPLATE.format(
            question=sample["question"],
            choices="\n".join(sample["choices"])
        )
