from typing import Dict
import pandas as pd

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class POIDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.data = pd.read_csv(config["data"]["path"])
        # self.points = data_df[["feature_easting", "feature_northing"]].values
        # self.descriptions = data_df["description"].values
        if config.model.text_encoder == "sentence_transformers":
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        elif config.model.text_encoder == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        else:
            pass
        print(111)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[torch.Tensor, str]:
        sample = {}
        point = torch.tensor(self.data.iloc[idx][["feature_easting", "feature_northing"]].values)
        text = self.data.iloc[idx]["description"]
        sample["point"] = point
        sample["text"] = text
        # point_normed = torch.tensor(self.data.iloc[idx][["x_normed", "y_normed"]].values)
        # sample["point_normed"] = point_normed

        return sample


class POIDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = config["data"]["path"]
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]

        if config.model.text_encoder == "sentence_transformers":
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        elif config.model.text_encoder == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        else:
            pass

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

    def _tokenize_and_encode(self, batch): 
        return self.tokenizer(batch['description'], truncation=True)

    def prepare_data(self) -> None:
        df_data = pd.read_csv(self.config["data"]["path"])
        dataset = Dataset.from_pandas(df_data)
        self.tokenized_data = dataset.map(self._tokenize_and_encode, batched=True)
        # need to remove the description column, otherwise will get error later
        self.tokenized_data = self.tokenized_data.remove_columns(["description", "pointx_class", "name", "ref_no"])
        print(222)

    def setup(self, stage=None):
        # train_test_split. DatasetDict({"train":, "test":})
        self.tokenized_data = self.tokenized_data.train_test_split(test_size=0.1)

    # def setup(self, stage=None):
    #     dataset = POIDataset(self.config["data"]["path"])

    #     N_val = int(len(dataset) * 0.2)
    #     N_train = len(dataset) - N_val
    #     self.train_dataset, self.val_dataset = torch.utils.data.random_split(
    #         dataset, [N_train, N_val]
    #     )

    def train_dataloader(self):

        train_loader = DataLoader(self.tokenized_data["train"], 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn= DataCollatorWithPadding(self.tokenizer, 
                                                              padding=True)
                          )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.tokenized_data["test"], 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn= DataCollatorWithPadding(self.tokenizer, 
                                                              padding=True)
                          )
        return val_loader



