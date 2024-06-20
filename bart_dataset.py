import os
import argparse
from typing import List

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils import load_yaml, batchify, pad_to


def _generate_data(source_lines: List[str], target_lines: List[str], tokenizer: BertTokenizer):
    for src_line, tgt_line in tqdm(zip(source_lines, target_lines), "building dataset", total=len(source_lines)):
        src_tokens = ['[CLS]'] + tokenizer.tokenize(src_line) + ['[SEP]']
        tgt_tokens = ["[CLS]"] + tokenizer.tokenize(tgt_line) + ["[SEP]"]
        src_token_ids = tokenizer.convert_tokens_to_ids(src_tokens)
        tgt_token_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)
        yield src_token_ids, tgt_token_ids


def _make_dataset(source_lines: List[str], target_lines: List[str], tokenizer: BertTokenizer, tgt_seq_len, src_seq_len, **kwargs):
    for src_token_ids, tgt_ids in _generate_data(source_lines, target_lines, tokenizer):
        tgt_ids = pad_to(tgt_seq_len, tgt_ids, tokenizer.pad_token_id)
        for (src_ids, ), attn_mask in batchify(src_token_ids, max_len=src_seq_len, pad=tokenizer.pad_token_id, **kwargs):
            yield src_ids, tgt_ids, attn_mask


def make_dataset(data_path, tokenizer_path, **kwargs) -> TensorDataset:
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    with open(data_path, "r") as f:
        parts = [line.strip().split("\t") for line in f]
        src_lines, tgt_lines = zip(*parts)
    src_tokens, tgt_tokens, attn_mask = zip(*_make_dataset(src_lines, tgt_lines, tokenizer, **kwargs))
    src_tokens = torch.LongTensor(src_tokens)
    tgt_tokens = torch.LongTensor(tgt_tokens)
    attn_mask = torch.LongTensor(attn_mask)
    return TensorDataset(src_tokens, tgt_tokens, attn_mask)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    dataset = make_dataset(**load_yaml(args.data_config))
    print(dataset[0])

