import pandas as pd
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm.notebook import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Compute avg char per tok")
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="File path ",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="auto",
        help="Pretrained tokenizer name",
    ),
    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="The number of processes to use for the preprocessing.",
    )
    args = parser.parse_args()
    return args

args = parse_args()
print(args.tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

def map_row(row):
    return len(row), len(tokenizer.tokenize(row))

def main():
    args = parse_args()
    
    with open(args.file_path,'r') as f:
        data = json.load(f)

    # Use multiprocessing to parallelize the tokenization and mapping process
    with Pool(args.num_workers) as pool:
        word_mappings = list(tqdm(pool.imap(map_row, data['data']), total=len(data['data'])))
    length = sum([x[0] for x in word_mappings])
    tok = sum([x[1] for x in word_mappings])
    print("Avg char per token : ",length/tok)

if __name__ == "__main__":
    main()
