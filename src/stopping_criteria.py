import torch
from transformers import StoppingCriteria, AutoTokenizer


class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, target_sequence: str, tokenizer: "AutoTokenizer"):
        self.target_sequence = target_sequence
        self.tokenizer = tokenizer
        

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0].cpu()[-10:])
        return self.target_sequence in generated_text
    
    
    def __len__(self):
        return 1

    
    def __iter__(self):
        yield self
