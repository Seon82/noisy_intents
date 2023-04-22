import torch
from datasets import Dataset as HGDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DydaDA(Dataset):
    def __init__(self, raw_data: HGDataset, tokenizer: PreTrainedTokenizer, max_len: int | None = None):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        txt = self.data[index]["Utterance"]
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
            "targets": torch.tensor(self.data[index]["Label"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_hugging_face(cls, split, tokenizer, max_len=None):
        """
        Load dataset from the hugging face hub.
        """
        return cls(load_dataset("silicone", "dyda_da", split=split), tokenizer, max_len)
