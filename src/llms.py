import json
import os
import aiohttp
import asyncio
import traceback
from collections.abc import Iterable

from transformers import AutoTokenizer

from tqdm.asyncio import tqdm

from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    model_hub = "/workspace/home/NLP_CORE/HUB_LLM"
    
    def __init__(
        self,
        endpoint_ip: str = "",
        model_name: str = "",
        eos_token: str = None,
        system_prompt: str = "",
        user_prompt_template: str = "",
        assistant_prompt_prefix: str = "",
    ):
        self.endpoint_ip = endpoint_ip
        self.model_name = model_name
        self.tokenizer = self.get_tokenizer()
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"
        if eos_token:
            self.tokenizer.eos_token = eos_token
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.assistant_prompt_prefix = assistant_prompt_prefix
        
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            os.path.join(self.model_hub, self.model_name),
            trust_remote_code=True
        )
    
    @abstractmethod
    def extract_sample(self, sample) -> dict:
        pass
    
    @abstractmethod
    def format_request_payload(self, prompt: str, max_len: int) -> dict:
        pass
    
    async def get_answer(
            self, 
            session, 
            sample, 
            max_len=1024):
        
        user_prompt = self.user_prompt_template.format(
            **self.extract_sample(sample)
        )
        temp = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        prompt = self.tokenizer.apply_chat_template(
            temp + [{"role": "user", "content": user_prompt}], 
            tokenize=False, add_generation_prompt=True
        ) + self.assistant_prompt_prefix

        headers = {
            "Content-Type": "application/json",
        }
        data = self.format_request_payload(prompt, max_len)
        try:
            async with session.post(self.endpoint_ip, headers=headers, json=data, timeout=600000) as resp:
                try:
                    resp = await resp.json()
                    # print(resp)
                    return prompt, resp["choices"][0]["text"]
                except:
                    resp = await resp.text()
                    return prompt, resp
        except:
            return prompt, "Failed: " + str(traceback.format_exc())
    
    async def generate_async(
        self, 
        dataset: Iterable,
        max_len: int = 1024,
        output_file: str = "tmp.jsonl",
        output_field: str = "output",
    ):
        async with asyncio.BoundedSemaphore(8):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for sample in dataset:
                    tasks.append(asyncio.ensure_future(
                        self.get_answer(session, sample, max_len)
                    ))
                results = await tqdm.gather(*tasks)
        with open(output_file, "w") as f:
            for sample, (input_prompt, output) in zip(dataset, results):
                sample[output_field] = self.assistant_prompt_prefix+output
                sample["input_prompt"] = input_prompt
                f.write(json.dumps(sample, ensure_ascii=False)+"\n")

        return 


class VllmModel(BaseModel):

    def format_request_payload(self, prompt: str, max_len: int) -> dict:
        return {
            "model": self.model_name,
            "prompt": prompt, 
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "max_tokens": max_len,
            "repetition_penalty": 1.0,
            "temperature": 0,
            "top_p": 0.9,
            "top_k": -1,
            "stop": [self.tokenizer.eos_token]
        }
    
    
class TgiModel(BaseModel):

    def get_tokenizer(self):
        if not self.model_name:
            tgi_info = requests.get(self.endpoint_ip.strip("/")+"/info").json()
            self.model_name = tgi_info["model_id"].split("/")[-1]
        return AutoTokenizer.from_pretrained(
            os.path.join(self.model_hub, self.model_name),
            trust_remote_code=True
        )
    
    def format_request_payload(self, prompt: str, max_len: int) -> dict:
        return {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': 1024,
                'repetition_penalty': 1.0,
                'do_sample': False,
                'use_cache': True,
                'stop': [self.tokenizer.eos_token],
                'temperature': 0,
                'top_p': 1,
                'top_k': -1
            },
        }
