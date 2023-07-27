import random
import torch
import os
from torch.utils.data import Dataset

from torch.distributed import get_rank, get_world_size
from tqdm import tqdm
import json


class PromptDataset(Dataset):
    def __init__(self, tokenizer, data_path):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.eos_token_id
        self.max_prompt_length = 512

        with open(data_path, "r") as file:
            self.raw = json.load(file)
        
        self.data = [self.process_raw_sample(raw_sample) for raw_sample in self.raw]

    def process_raw_sample(self, raw_sample):
        name = {
            "human": "Human",
            "gpt": "chatgpt2-base"
        }
        text = "".join([name[conv["from"]] + ': ' + conv["value"] + '\n' for conv in raw_sample["items"]])
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        input_ids = self.tokenizer.encode(self.data[index])
        input_ids = input_ids[:self.max_prompt_length]
        input_ids[-1] = self.tokenizer.eos_token_id
        return input_ids

    def collate(self, samples):
        bs = len(samples)

        max_prompt_length = self.max_prompt_length

        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }

        for i, (input_ids) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(input_ids):] = torch.tensor(input_ids, dtype=torch.long)
            model_batch["attention_mask"][i][-len(input_ids):] = 1

        return model_batch
