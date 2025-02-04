import unicodedata
import string
import json
import re
from string import punctuation, whitespace

import torch

# PREDICTION_PATTERN = re.compile(r"""\\"final_choice":{(.*?)}""")
# PREDICTION_PATTERN = re.compile(r"\{\"final_choice\": \"(.*?)\"\n}")
# PREDICTION_PATTERN1 = re.compile("""\{"final_choice": (.*?)\}""")
PREDICTION_PATTERN = re.compile("""\"final_choice\": (.*?)""")
def read_jsonl(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_json(path: str):
    data = None
    with open(path, "r") as f:
        data = json.loads(f.read())
    return data

def read_json_specific(path: str):
    data = None
    with open(path, "r") as f:
        data = f.read().split("\n")[:-1]
        data = [json.loads(x) for x in data]
    return data
    
SPLITTER_PATTERN = re.compile(r"\S+|\s+")
def splitter(num_words, doc):
    words_n_space = SPLITTER_PATTERN.findall(doc)
    return ["".join(words_n_space[i:i+num_words*2]) for i in range(0, len(words_n_space), num_words*2)]

    print(PREDICTION_PATTERN1.findall(prompt))
    if PREDICTION_PATTERN1.findall(prompt) != []:
        predictions = PREDICTION_PATTERN1.findall(prompt)
    else:
        predictions = PREDICTION_PATTERN2.findall(prompt)
    # print(predictions)
def normalize_name(text: str):
    replace_str = "/"+whitespace
    return text.translate((str.maketrans(replace_str, '_'*len(replace_str))))


def remove_punct(text: str):
    return text.translate(
        str.maketrans('', '', string.punctuation)
    )

def bm25_preprocess(text: str):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = remove_punct(text)
    return text.split()


def bm25_get_top_n(bm25, sample, corpus, n=1):
    tokenized_query = bm25_preprocess(sample["question"])
    top_2n = bm25.get_top_n(tokenized_query, corpus, n=2*n+4)
    count = 1
    for i in range(count):
        sample[f"top_{i+1}"] = ""
    for doc in top_2n:
        if doc in sample["chunks"]:
            sample[f"top_{count}"] = doc
            count+=1
        if count > n:
            break


def select_answer_for_multiple_choices(model, tokenizer, prompt, choices):
    predictions = PREDICTION_PATTERN.findall(prompt)
    print(predictions)
    # if len(predictions) > 1:
    #     # print(ord(predictions[-1][0].upper())-65)
    #     return ord(predictions[-1][0].upper())-65
    # else:
    #     return -65
    # print(len(predictions[-1]))
    # if len(predictions) >= 1:
    #     # print(ord(predictions[-1][0].upper())-65)
    #     if len(predictions[-1]) == 1:
    #         return ord(predictions[-1][0].upper())-65
    #     else:
    #         return ord(predictions[-1][1].upper())-65
    # else:
    #     return -65
    
    if len(predictions) >= 1:
        # print(ord(predictions[0].upper())-65)
        if len(predictions) == 1:
            return ord(predictions[0].upper())-65
        else:
            return ord(predictions[1].upper())-65
    else:
        return -65
        
    logits = compute_logit_for_choices(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt, 
        choices=choices
    )
    max_logit = max(logits)
    max_logit_id = logits.index(max_logit)
    print("Logits: " + str(logits))
    return max_logit_id
    
    
def compute_logit_for_choices(prompt, choices, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    choices_token_id = [tokenizer.convert_tokens_to_ids(choice.strip()[0]) for choice in choices]
    with torch.no_grad():
        logits = model(input_ids).logits
    return [logits[:, -1, choice_token_id].item() for choice_token_id in choices_token_id]
