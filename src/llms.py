import aiohttp
import asyncio
import traceback
from collections.abc import Iterable

from transformers import AutoTokenizer

from tqdm.asyncio import tqdm

from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    def __init__(
        self,
        endpoint_ip: str = "",
        served_model_name: str = "",
        model_path: str = "",
        eos_token: str = None,
        system_prompt: str = "",
    ):
        self.endpoint_ip = endpoint_ip
        self.served_model_name = served_model_name
        self.model_path = model_path
        self.tokenizer = self.get_tokenizer()
        self.tokenizer.add_eos_token = False
        self.tokenizer.padding_side = "left"
        if eos_token:
            self.tokenizer.eos_token = eos_token
        self.system_prompt = system_prompt
        

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    
    @abstractmethod
    def format_request_payload(self, prompt: str, **generation_kwargs) -> dict:
        pass


    async def request(self, session, data):
        headers = {"Content-Type": "application/json"}
        try:
            async with session.post(self.endpoint_ip, headers=headers, json=data, timeout=600000) as resp:
                try:
                    resp = await resp.json()
                    return [answer["text"] for answer in resp["choices"]]
                except:
                    resp = await resp.text()
                    return [resp]
        except:
            return ["Failed: " + str(traceback.format_exc())]
    

    async def generate_answer(
        self, 
        session, 
        user_query, 
        **generation_kwargs
    ):
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        prompt = self.tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": user_query}], 
            tokenize=False, add_generation_prompt=True
        )
        data = self.format_request_payload(prompt, **generation_kwargs)
        resp = await self.request(session=session, data=data)
        return resp
    

    async def batch_generate_answers(
        self, 
        user_queries: Iterable,
        **generation_kwargs
    ):
        async with asyncio.BoundedSemaphore(8):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for query in user_queries:
                    tasks.append(asyncio.ensure_future(
                        self.generate_answer(session, query, **generation_kwargs)
                    ))
                answers = await tqdm.gather(*tasks)
        
        return answers
        

    def __call__(self, user_queries: Iterable[str], generation_config: dict = {}):
        return asyncio.run(self.batch_generate_answers(
            user_queries=user_queries,
            **generation_config
        ))
    

class BaseModelForMultipleChoice(BaseModel):
    
    async def extract_final_choice(
        self,
        session,
        user_query,
        sys_answer,
    ):
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        prompt = self.tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": user_query}], 
            tokenize=False, add_generation_prompt=True
        ) + sys_answer + "\nFinal answer: "
        data = self.format_request_payload(
            prompt, 
            max_tokens=1,
            allowed_token_ids=[
                self.tokenizer.convert_tokens_to_ids("A"),
                self.tokenizer.convert_tokens_to_ids("B"),
                self.tokenizer.convert_tokens_to_ids("C"),
                self.tokenizer.convert_tokens_to_ids("D"),
                self.tokenizer.convert_tokens_to_ids("E"),
            ]
        )
        resp = await self.request(session=session, data=data)
        return resp[0]
    

    async def batch_extract_final_choices(
        self,
        user_queries,
        sys_answers,
    ):
        async with asyncio.BoundedSemaphore(8):
            session_timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                tasks = []
                for query, answer_candidates in zip(user_queries, sys_answers):
                    for answer in answer_candidates:
                        tasks.append(asyncio.ensure_future(
                            self.extract_final_choice(
                                session, query, answer
                            )
                        ))
                final_choices = await tqdm.gather(*tasks)
        reformatted_final_choices = []
        for query, answer_candidates in zip(user_queries, sys_answers):
            reformatted_final_choices.append([])
            for answer in answer_candidates:
                reformatted_final_choices[-1].append(final_choices.pop(0))

        return reformatted_final_choices


    def __call__(self, user_queries: Iterable[str], generation_config: dict = {}):
        sys_answers = asyncio.run(self.batch_generate_answers(
            user_queries=user_queries,
            **generation_config
        ))
        final_choices = asyncio.run(self.batch_extract_final_choices(
            user_queries=user_queries,
            sys_answers=sys_answers
        ))
        return sys_answers, final_choices
    


class VllmModel(BaseModel):

    def __init__(
        self,
        endpoint_ip: str = "",
        served_model_name: str = "",
        model_path: str = "",
        eos_token: str = None,
        system_prompt: str = "",
    ):
        super().__init__(
            endpoint_ip.strip("/") + "/v1/completions", 
            served_model_name, 
            model_path, 
            eos_token, 
            system_prompt, 
        )

    def format_request_payload(self, prompt: str, **generation_kwargs) -> dict:
        return {
            "model": self.served_model_name,
            "prompt": prompt, 
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "max_tokens": 1024,
            "repetition_penalty": 1.0,
            "temperature": 0,
            "top_p": 0.9,
            "top_k": -1,
            "stop": [self.tokenizer.eos_token],
            **generation_kwargs
        }
    


class VllmModelForMultipleChoice(BaseModelForMultipleChoice):

    def __init__(
        self,
        endpoint_ip: str = "",
        served_model_name: str = "",
        model_path: str = "",
        eos_token: str = None,
        system_prompt: str = "",
    ):
        super().__init__(
            endpoint_ip.strip("/") + "/v1/completions", 
            served_model_name, 
            model_path, 
            eos_token, 
            system_prompt,
        )


    def format_request_payload(self, prompt: str, **generation_kwargs) -> dict:
        return {
            "model": self.served_model_name,
            "prompt": prompt, 
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "max_tokens": 1024,
            "repetition_penalty": 1.0,
            "temperature": 0,
            "top_p": 0.9,
            "top_k": -1,
            "stop": [self.tokenizer.eos_token],
            **generation_kwargs
        }
    
    
class TgiModel(BaseModel):

    # def get_tokenizer(self):
    #     import requests

    #     if not self.model_name:
    #         tgi_info = requests.get(self.endpoint_ip.strip("/")+"/info").json()
    #         self.model_name = tgi_info["model_id"].split("/")[-1]
    #     return AutoTokenizer.from_pretrained(
    #         os.path.join(self.model_hub, self.model_name),
    #         trust_remote_code=True
    #     )
    
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
