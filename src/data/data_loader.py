"""
Data Loader Module
==================
"""

import os
import json
import random
from typing import Optional, Tuple, Dict, List, Union, Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import Tensor
import numpy as np


@dataclass
class DataConfig:
    max_length: int = 512
    min_length: int = 32
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    vocab_size: int = 50000


class SimpleTokenizer:

    def __init__(
            self,
            vocab_size: int = 50000,
            max_length: int = 512,
            pad_token_id: int = 0,
            unk_token_id: int = 1,
            bos_token_id: int = 2,
            eos_token_id: int = 3,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self._char_to_id = {}
        self._id_to_char = {}
        self._build_vocab()

    def _build_vocab(self):
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

        printable_chars = [chr(i) for i in range(32, 127)]

        extra_chars = ['\n', '\t', '\r']

        all_tokens = special_tokens + printable_chars + extra_chars

        for idx, token in enumerate(all_tokens):
            self._char_to_id[token] = idx
            self._id_to_char[idx] = token

    def _tokenize(self, text: str) -> List[int]:
        token_ids = [self.bos_token_id]

        for char in text:
            if char in self._char_to_id:
                token_ids.append(self._char_to_id[char])
            else:
                token_id = (hash(char) % (self.vocab_size - 100)) + 100
                token_ids.append(token_id)

        token_ids.append(self.eos_token_id)
        return token_ids

    def __call__(
            self,
            text: Union[str, List[str]],
            max_length: Optional[int] = None,
            padding: str = "max_length",
            truncation: bool = True,
            return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self.max_length

        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        batch_input_ids = []
        batch_attention_mask = []

        for t in texts:
            token_ids = self._tokenize(t)

            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            attention_mask = [1] * len(token_ids)

            if padding == "max_length":
                pad_length = max_length - len(token_ids)
                if pad_length > 0:
                    token_ids = token_ids + [self.pad_token_id] * pad_length
                    attention_mask = attention_mask + [0] * pad_length

            batch_input_ids.append(token_ids)
            batch_attention_mask.append(attention_mask)

        if return_tensors == "pt":
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        else:
            input_ids = batch_input_ids
            attention_mask = batch_attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for tid in token_ids:
            if tid in self._id_to_char:
                char = self._id_to_char[tid]
                if char not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    chars.append(char)
        return ''.join(chars)


class TextDataset(Dataset):

    def __init__(
            self,
            texts: List[str],
            labels: Optional[List[int]] = None,
            tokenizer=None,
            max_length: int = 512,
            granularity: str = "document",
            vocab_size: int = 50000,
    ):
        """
        Args:
            texts
            labels
            tokenizer
            max_length
            granularity
            vocab_size
        """
        self.texts = texts
        self.labels = labels if labels is not None else [None] * len(texts)
        self.max_length = max_length
        self.granularity = granularity
        self.vocab_size = vocab_size

        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(
                vocab_size=vocab_size,
                max_length=max_length,
            )
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, int]]:
        text = self.texts[idx]
        label = self.labels[idx]

        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        except Exception as e:
            input_ids = self._simple_encode(text)
            attention_mask = (input_ids != 0).long()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text,
            "idx": idx,
        }

        if label is not None:
            item["label"] = label

        return item


        tokens = text.lower().split()[:self.max_length - 2]
        input_ids = torch.zeros(self.max_length, dtype=torch.long)

        # BOS token
        input_ids[0] = 2

        for i, token in enumerate(tokens):
            input_ids[i + 1] = (hash(token) % (self.vocab_size - 100)) + 100

        # EOS token
        if len(tokens) + 1 < self.max_length:
            input_ids[len(tokens) + 1] = 3

        return input_ids


class MembershipDataset(Dataset):

    def __init__(
            self,
            member_texts: List[str],
            non_member_texts: List[str],
            tokenizer=None,
            max_length: int = 512,
            balance: bool = True,
            vocab_size: int = 50000,
    ):
        self.max_length = max_length
        self.vocab_size = vocab_size

        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(
                vocab_size=vocab_size,
                max_length=max_length,
            )
        else:
            self.tokenizer = tokenizer

        if balance:
            min_size = min(len(member_texts), len(non_member_texts))
            member_texts = random.sample(member_texts, min_size)
            non_member_texts = random.sample(non_member_texts, min_size)

        self.texts = member_texts + non_member_texts
        self.labels = [1] * len(member_texts) + [0] * len(non_member_texts)

        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)
        self.texts = list(self.texts)
        self.labels = list(self.labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        text = self.texts[idx]
        label = self.labels[idx]

        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        except Exception as e:
            # Fallback: 使用简单的hash编码
            input_ids = self._simple_encode(text)
            attention_mask = (input_ids != 0).long()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "text": text,
            "idx": idx,
        }

        return item

    def _simple_encode(self, text: str) -> Tensor:
        tokens = text.lower().split()[:self.max_length - 2]
        input_ids = torch.zeros(self.max_length, dtype=torch.long)

        # BOS token
        input_ids[0] = 2

        for i, token in enumerate(tokens):
            input_ids[i + 1] = (hash(token) % (self.vocab_size - 100)) + 100

        # EOS token
        if len(tokens) + 1 < self.max_length:
            input_ids[len(tokens) + 1] = 3

        return input_ids

    def get_member_ratio(self) -> float:
        return sum(self.labels) / len(self.labels)


class DomainDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            domain_id: int,
            tokenizer=None,
            max_length: int = 512,
            vocab_size: int = 50000,
    ):
        self.texts = texts
        self.domain_id = domain_id
        self.max_length = max_length
        self.vocab_size = vocab_size

        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(
                vocab_size=vocab_size,
                max_length=max_length,
            )
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        text = self.texts[idx]

        try:
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        except Exception as e:
            input_ids = self._simple_encode(text)
            attention_mask = (input_ids != 0).long()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "domain_id": self.domain_id,
            "text": text,
            "idx": idx,
        }

        return item

    def _simple_encode(self, text: str) -> Tensor:
        tokens = text.lower().split()[:self.max_length - 2]
        input_ids = torch.zeros(self.max_length, dtype=torch.long)

        # BOS token
        input_ids[0] = 2

        for i, token in enumerate(tokens):
            input_ids[i + 1] = (hash(token) % (self.vocab_size - 100)) + 100

        # EOS token
        if len(tokens) + 1 < self.max_length:
            input_ids[len(tokens) + 1] = 3

        return input_ids


def collate_fn(batch: List[Dict]) -> Dict[str, Tensor]:
    keys = batch[0].keys()

    result = {}

    for key in keys:
        values = [item[key] for item in batch]

        if isinstance(values[0], (str, int, float)):
            result[key] = values
        elif isinstance(values[0], Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values

    return result


class DataLoaderFactory:
    def __init__(self, config: DataConfig):
        self.config = config
        self.tokenizer = SimpleTokenizer(
            vocab_size=getattr(config, 'vocab_size', 50000),
            max_length=config.max_length,
        )

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def create_text_dataloader(
            self,
            texts: List[str],
            labels: Optional[List[int]] = None,
            shuffle: bool = True,
    ) -> DataLoader:
        dataset = TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def create_membership_dataloader(
            self,
            member_texts: List[str],
            non_member_texts: List[str],
            shuffle: bool = True,
            balance: bool = True,
    ) -> DataLoader:
        dataset = MembershipDataset(
            member_texts=member_texts,
            non_member_texts=non_member_texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            balance=balance,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def create_domain_dataloaders(
            self,
            domain_texts: Dict[int, List[str]],
            shuffle: bool = True,
    ) -> Dict[int, DataLoader]:
        dataloaders = {}

        for domain_id, texts in domain_texts.items():
            dataset = DomainDataset(
                texts=texts,
                domain_id=domain_id,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )

            dataloaders[domain_id] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=True if torch.cuda.is_available() else False,
            )

        return dataloaders


class HuggingFaceDataLoader:

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def load_dataset(
            self,
            dataset_name: str,
            subset: Optional[str] = None,
            split: str = "train",
            text_column: str = "text",
            max_samples: Optional[int] = None,
    ) -> List[str]:
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                cache_dir=self.cache_dir,
            )

            if max_samples is not None:
                dataset = dataset.select(range(min(max_samples, len(dataset))))

            texts = dataset[text_column]
            return list(texts)

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return []

    def load_pile_subset(
            self,
            subset_name: str,
            max_samples: int = 10000,
    ) -> List[str]:
        texts = self.load_dataset(
            "EleutherAI/pile",
            split="train",
            text_column="text",
            max_samples=max_samples * 10,
        )

        return texts[:max_samples]

    def load_wikitext(
            self,
            subset: str = "wikitext-103-v1",
            split: str = "train",
            max_samples: Optional[int] = None,
    ) -> List[str]:
        return self.load_dataset(
            "wikitext",
            subset=subset,
            split=split,
            text_column="text",
            max_samples=max_samples,
        )


class TextGranularityProcessor:
    def __init__(self, min_length: int = 32):
        self.min_length = min_length

    def process_document(self, text: str) -> List[str]:
        if len(text) >= self.min_length:
            return [text]
        return []

    def process_paragraph(self, text: str) -> List[str]:
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if len(p.strip()) >= self.min_length]

    def process_sentence(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_length]

    def process(self, texts: List[str], granularity: str) -> List[str]:
        processed = []

        for text in texts:
            if granularity == "document":
                processed.extend(self.process_document(text))
            elif granularity == "paragraph":
                processed.extend(self.process_paragraph(text))
            elif granularity == "sentence":
                processed.extend(self.process_sentence(text))
            else:
                raise ValueError(f"Unknown granularity: {granularity}")

        return processed


class AttackDatasetBuilder:
    def __init__(
            self,
            tokenizer=None,
            max_length: int = 512,
            num_samples_per_granularity: int = 20000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples_per_granularity
        self.granularity_processor = TextGranularityProcessor()

    def build_from_open_source_model(
            self,
            train_texts: List[str],
            test_texts: List[str],
            granularity: str = "document",
    ) -> MembershipDataset:

        member_texts = self.granularity_processor.process(train_texts, granularity)
        non_member_texts = self.granularity_processor.process(test_texts, granularity)


        num_each = self.num_samples // 2

        if len(member_texts) > num_each:
            member_texts = random.sample(member_texts, num_each)
        if len(non_member_texts) > num_each:
            non_member_texts = random.sample(non_member_texts, num_each)

        return MembershipDataset(
            member_texts=member_texts,
            non_member_texts=non_member_texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def build_from_timestamp(
            self,
            all_texts: List[str],
            timestamps: List[int],
            cutoff_date: int,
            granularity: str = "document",
    ) -> MembershipDataset:

        member_texts = []
        non_member_texts = []

        for text, ts in zip(all_texts, timestamps):
            if ts < cutoff_date:
                member_texts.append(text)
            else:
                non_member_texts.append(text)


        member_texts = self.granularity_processor.process(member_texts, granularity)
        non_member_texts = self.granularity_processor.process(non_member_texts, granularity)

        return MembershipDataset(
            member_texts=member_texts,
            non_member_texts=non_member_texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def build_multi_granularity(
            self,
            train_texts: List[str],
            test_texts: List[str],
    ) -> Dict[str, MembershipDataset]:

        datasets = {}

        for granularity in ["document", "paragraph", "sentence"]:
            datasets[granularity] = self.build_from_open_source_model(
                train_texts, test_texts, granularity
            )

        return datasets


if __name__ == "__main__":
    print("Testing Data Loader Module...")

    texts = [
                "This is a sample document about machine learning and artificial intelligence.",
                "Another document discussing natural language processing techniques.",
                "Deep learning has revolutionized many fields including computer vision.",
                "Transformers are the foundation of modern large language models.",
            ] * 100

    labels = [1, 0, 1, 0] * 100

    print("Testing TextDataset...")
    dataset = TextDataset(texts, labels, max_length=128)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")

    print("\nTesting MembershipDataset...")
    member_texts = texts[:200]
    non_member_texts = texts[200:]
    membership_dataset = MembershipDataset(
        member_texts, non_member_texts, max_length=128
    )
    print(f"Membership dataset size: {len(membership_dataset)}")
    print(f"Member ratio: {membership_dataset.get_member_ratio():.2f}")

    print("\nTesting DataLoaderFactory...")
    config = DataConfig(batch_size=16)
    factory = DataLoaderFactory(config)
    dataloader = factory.create_text_dataloader(texts, labels)
    print(f"Number of batches: {len(dataloader)}")

    print("\nTesting TextGranularityProcessor...")
    processor = TextGranularityProcessor(min_length=10)

    test_text = """This is the first paragraph. It has multiple sentences.

This is the second paragraph. It also has sentences.

And this is the third paragraph."""

    doc_results = processor.process([test_text], "document")
    para_results = processor.process([test_text], "paragraph")
    sent_results = processor.process([test_text], "sentence")

    print(f"Document count: {len(doc_results)}")
    print(f"Paragraph count: {len(para_results)}")
    print(f"Sentence count: {len(sent_results)}")

    print("\nTesting AttackDatasetBuilder...")
    builder = AttackDatasetBuilder(num_samples_per_granularity=100)

    train_texts = texts[:200]
    test_texts = texts[200:]

    multi_datasets = builder.build_multi_granularity(train_texts, test_texts)
    for granularity, dataset in multi_datasets.items():
        print(f"{granularity}: {len(dataset)} samples")

    print("\nAll tests passed!")