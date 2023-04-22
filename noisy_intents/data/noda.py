from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class NoDA(Dataset):
    _DATASET_URL = "https://drive.google.com/uc?export=download&id=15xhoZLpXuu1sA0ZieSQN5_A0-VrXaZy8"

    def __init__(
        self, tokenizer: PreTrainedTokenizer, max_len: int | None = None, raw_data: pd.DataFrame | None = None
    ):
        if raw_data is None:
            self.data = pd.read_csv(self._DATASET_URL)
        else:
            self.data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        txt = self.data.iloc[index]["text"]
        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
            padding="max_length",
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.data.iloc[index]["label"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_csv(cls, path: Path | str, tokenizer, max_len=None):
        """
        Load dataset from the hugging face hub.
        """
        return cls(pd.read_csv(path), tokenizer, max_len)
