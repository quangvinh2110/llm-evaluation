from collections.abc import Iterable
from abc import ABC, abstractmethod
from .utils import read_jsonl, read_json


class BaseBatch(ABC):
    def __init__(self, data: Iterable):
        self.data = data

    @abstractmethod
    def process(self, sample):
        pass


    def to_user_prompts(self):
        # print(self.data[0:1000])
        return [self.process(s) for s in self.data]

    
    def __iter__(self):
        for sample in self.data:
            yield sample

    
    def __len__(self):
        return len(self.data)
        

    def __repr__(self) -> str:
        return str(self.data)
    

    def __str__(self) -> str:
        return str(self.data)
    

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return type(self)([self.data[ii] for ii in range(*key.indices(len(self)))])
        elif isinstance(key, int):
            if key < 0: # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.data[key] # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")



class VmluBatch(BaseBatch):

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


class VmluDataset(VmluBatch):

    def __init__(self, data: Iterable = None, filepath: str = None):
        if not filepath and not data:
            raise ValueError()
        if data:
            self.data = data
            return
        if filepath.endswith(".jsonl"):
            data = read_jsonl(filepath)
        elif filepath.endswith(".json"):
            data = read_json(filepath)
        elif filepath.endswith(".parquet"):
            import pandas as pd
            data = pd.read_parquet(filepath).to_dict('records')
        elif filepath.endswith(".csv"):
            import pandas as pd
            data = pd.read_csv(filepath).to_dict('records')  
        else:
            raise ValueError()
        
        self.data = data
    

    def batch_iter(self, batch_size: int):
        for i in range(0, len(self.data), batch_size):
            yield VmluBatch(self.data[i : i + batch_size])



