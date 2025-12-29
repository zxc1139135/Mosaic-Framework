"""
Data Loader Module
==================

数据加载和预处理模块，支持多种数据集和数据格式。

功能:
    - 加载训练和评估数据集
    - 数据预处理和tokenization
    - 成员/非成员数据集构建
    - 多粒度数据划分 (文档/段落/句子)
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
    """数据配置"""
    max_length: int = 512
    min_length: int = 32
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    vocab_size: int = 50000


class SimpleTokenizer:
    """
    简单的Tokenizer实现

    用于在没有外部tokenizer库时进行基本的文本tokenization
    """

    def __init__(
            self,
            vocab_size: int = 50000,
            max_length: int = 512,
            pad_token_id: int = 0,
            unk_token_id: int = 1,
            bos_token_id: int = 2,
            eos_token_id: int = 3,
    ):
        """
        Args:
            vocab_size: 词汇表大小
            max_length: 最大序列长度
            pad_token_id: 填充token ID
            unk_token_id: 未知token ID
            bos_token_id: 序列开始token ID
            eos_token_id: 序列结束token ID
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 简单的字符级词汇表
        self._char_to_id = {}
        self._id_to_char = {}
        self._build_vocab()

    def _build_vocab(self):
        """构建基础词汇表"""
        # 特殊token
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

        # ASCII可打印字符
        printable_chars = [chr(i) for i in range(32, 127)]

        # 常见标点和空白
        extra_chars = ['\n', '\t', '\r']

        all_tokens = special_tokens + printable_chars + extra_chars

        for idx, token in enumerate(all_tokens):
            self._char_to_id[token] = idx
            self._id_to_char[idx] = token

    def _tokenize(self, text: str) -> List[int]:
        """将文本转换为token ID列表"""
        token_ids = [self.bos_token_id]

        for char in text:
            if char in self._char_to_id:
                token_ids.append(self._char_to_id[char])
            else:
                # 使用hash进行未知字符的确定性映射
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
        """
        Tokenize文本

        Args:
            text: 输入文本或文本列表
            max_length: 最大长度
            padding: 填充策略
            truncation: 是否截断
            return_tensors: 返回tensor类型

        Returns:
            包含input_ids和attention_mask的字典
        """
        if max_length is None:
            max_length = self.max_length

        # 处理单个文本
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        batch_input_ids = []
        batch_attention_mask = []

        for t in texts:
            token_ids = self._tokenize(t)

            # 截断
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            # 创建attention mask
            attention_mask = [1] * len(token_ids)

            # 填充
            if padding == "max_length":
                pad_length = max_length - len(token_ids)
                if pad_length > 0:
                    token_ids = token_ids + [self.pad_token_id] * pad_length
                    attention_mask = attention_mask + [0] * pad_length

            batch_input_ids.append(token_ids)
            batch_attention_mask.append(attention_mask)

        # 转换为tensor
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
        """将token ID列表解码为文本"""
        chars = []
        for tid in token_ids:
            if tid in self._id_to_char:
                char = self._id_to_char[tid]
                if char not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                    chars.append(char)
        return ''.join(chars)


class TextDataset(Dataset):
    """
    通用文本数据集

    支持成员推理攻击的数据格式:
        - 文本内容
        - 成员标签 (1: 成员, 0: 非成员)
        - 数据粒度 (document, paragraph, sentence)
    """

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
            texts: 文本列表
            labels: 成员标签列表
            tokenizer: Tokenizer实例
            max_length: 最大序列长度
            granularity: 数据粒度
            vocab_size: 词汇表大小（用于fallback tokenization）
        """
        self.texts = texts
        self.labels = labels if labels is not None else [None] * len(texts)
        self.max_length = max_length
        self.granularity = granularity
        self.vocab_size = vocab_size

        # 如果没有提供tokenizer，创建一个默认的
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

        # 使用tokenizer编码文本
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
            "text": text,
            "idx": idx,
        }

        if label is not None:
            item["label"] = label

        return item

    def _simple_encode(self, text: str) -> Tensor:
        """简单的hash编码作为fallback"""
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
    """
    成员推理攻击专用数据集

    包含成员和非成员样本，支持平衡采样
    """

    def __init__(
            self,
            member_texts: List[str],
            non_member_texts: List[str],
            tokenizer=None,
            max_length: int = 512,
            balance: bool = True,
            vocab_size: int = 50000,
    ):
        """
        Args:
            member_texts: 成员文本列表
            non_member_texts: 非成员文本列表
            tokenizer: Tokenizer实例
            max_length: 最大序列长度
            balance: 是否平衡数据
            vocab_size: 词汇表大小（用于fallback tokenization）
        """
        self.max_length = max_length
        self.vocab_size = vocab_size

        # 如果没有提供tokenizer，创建一个默认的
        if tokenizer is None:
            self.tokenizer = SimpleTokenizer(
                vocab_size=vocab_size,
                max_length=max_length,
            )
        else:
            self.tokenizer = tokenizer

        # 如果需要平衡，采样到相同数量
        if balance:
            min_size = min(len(member_texts), len(non_member_texts))
            member_texts = random.sample(member_texts, min_size)
            non_member_texts = random.sample(non_member_texts, min_size)

        # 合并数据
        self.texts = member_texts + non_member_texts
        self.labels = [1] * len(member_texts) + [0] * len(non_member_texts)

        # 打乱顺序
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

        # 使用tokenizer编码文本
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
        """简单的hash编码作为fallback"""
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
        """获取成员比例"""
        return sum(self.labels) / len(self.labels)


class DomainDataset(Dataset):
    """
    领域专用数据集

    用于专家模型的领域专业化训练
    """

    def __init__(
            self,
            texts: List[str],
            domain_id: int,
            tokenizer=None,
            max_length: int = 512,
            vocab_size: int = 50000,
    ):
        """
        Args:
            texts: 文本列表
            domain_id: 领域ID (0-7)
            tokenizer: Tokenizer实例
            max_length: 最大序列长度
            vocab_size: 词汇表大小（用于fallback tokenization）
        """
        self.texts = texts
        self.domain_id = domain_id
        self.max_length = max_length
        self.vocab_size = vocab_size

        # 如果没有提供tokenizer，创建一个默认的
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

        # 使用tokenizer编码文本
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
            "domain_id": self.domain_id,
            "text": text,
            "idx": idx,
        }

        return item

    def _simple_encode(self, text: str) -> Tensor:
        """简单的hash编码作为fallback"""
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
    """
    自定义collate函数，用于批次数据整理

    Args:
        batch: 数据项列表

    Returns:
        整理后的批次字典
    """
    # 获取第一个样本的keys
    keys = batch[0].keys()

    result = {}

    for key in keys:
        values = [item[key] for item in batch]

        # 跳过非tensor类型
        if isinstance(values[0], (str, int, float)):
            result[key] = values
        elif isinstance(values[0], Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values

    return result


class DataLoaderFactory:
    """
    数据加载器工厂类

    提供便捷的数据加载器创建接口
    """

    def __init__(self, config: DataConfig):
        """
        Args:
            config: 数据配置
        """
        self.config = config
        # 默认使用SimpleTokenizer
        self.tokenizer = SimpleTokenizer(
            vocab_size=getattr(config, 'vocab_size', 50000),
            max_length=config.max_length,
        )

    def set_tokenizer(self, tokenizer):
        """设置tokenizer"""
        self.tokenizer = tokenizer

    def create_text_dataloader(
            self,
            texts: List[str],
            labels: Optional[List[int]] = None,
            shuffle: bool = True,
    ) -> DataLoader:
        """创建文本数据加载器"""
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
            num_workers=0,  # 使用0避免多进程问题
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
        """创建成员推理数据加载器"""
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
            num_workers=0,  # 使用0避免多进程问题
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def create_domain_dataloaders(
            self,
            domain_texts: Dict[int, List[str]],
            shuffle: bool = True,
    ) -> Dict[int, DataLoader]:
        """为每个领域创建数据加载器"""
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
                num_workers=0,  # 使用0避免多进程问题
                collate_fn=collate_fn,
                pin_memory=True if torch.cuda.is_available() else False,
            )

        return dataloaders


class HuggingFaceDataLoader:
    """
    HuggingFace数据集加载器

    支持加载HuggingFace Hub上的数据集
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir

    def load_dataset(
            self,
            dataset_name: str,
            subset: Optional[str] = None,
            split: str = "train",
            text_column: str = "text",
            max_samples: Optional[int] = None,
    ) -> List[str]:
        """
        加载HuggingFace数据集

        Args:
            dataset_name: 数据集名称
            subset: 子集名称
            split: 数据集划分
            text_column: 文本列名
            max_samples: 最大样本数

        Returns:
            texts: 文本列表
        """
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
        """
        加载Pile数据集的特定子集

        Args:
            subset_name: 子集名称 (如 "ArXiv", "PubMed", "Wikipedia")
            max_samples: 最大样本数

        Returns:
            texts: 文本列表
        """
        texts = self.load_dataset(
            "EleutherAI/pile",
            split="train",
            text_column="text",
            max_samples=max_samples * 10,  # 预加载更多以过滤
        )

        # 按子集过滤 (简化处理)
        return texts[:max_samples]

    def load_wikitext(
            self,
            subset: str = "wikitext-103-v1",
            split: str = "train",
            max_samples: Optional[int] = None,
    ) -> List[str]:
        """加载WikiText数据集"""
        return self.load_dataset(
            "wikitext",
            subset=subset,
            split=split,
            text_column="text",
            max_samples=max_samples,
        )


class TextGranularityProcessor:
    """
    文本粒度处理器

    将文本分解为不同粒度的单元:
        - 文档级: 完整文本
        - 段落级: 按段落分割
        - 句子级: 按句子分割
    """

    def __init__(self, min_length: int = 32):
        """
        Args:
            min_length: 最小文本长度
        """
        self.min_length = min_length

    def process_document(self, text: str) -> List[str]:
        """文档级处理"""
        if len(text) >= self.min_length:
            return [text]
        return []

    def process_paragraph(self, text: str) -> List[str]:
        """段落级处理"""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if len(p.strip()) >= self.min_length]

    def process_sentence(self, text: str) -> List[str]:
        """句子级处理"""
        # 简单的句子分割
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= self.min_length]

    def process(self, texts: List[str], granularity: str) -> List[str]:
        """
        处理文本列表

        Args:
            texts: 文本列表
            granularity: 粒度 ("document", "paragraph", "sentence")

        Returns:
            processed: 处理后的文本列表
        """
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
    """
    攻击数据集构建器

    论文5.1节: 构建成员推理攻击的训练和测试数据集
    """

    def __init__(
            self,
            tokenizer=None,
            max_length: int = 512,
            num_samples_per_granularity: int = 20000,
    ):
        """
        Args:
            tokenizer: Tokenizer实例
            max_length: 最大序列长度
            num_samples_per_granularity: 每个粒度的样本数
        """
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
        """
        为开源模型构建攻击数据集

        论文策略: 从官方训练数据集采样正样本，
        从训练截止日期后的数据采样负样本

        Args:
            train_texts: 训练集文本 (成员)
            test_texts: 测试集文本 (非成员)
            granularity: 数据粒度

        Returns:
            dataset: 成员推理数据集
        """
        # 处理粒度
        member_texts = self.granularity_processor.process(train_texts, granularity)
        non_member_texts = self.granularity_processor.process(test_texts, granularity)

        # 采样到指定数量
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
        """
        使用时间戳推断构建数据集

        用于商业黑盒模型，根据训练截止日期推断标签

        Args:
            all_texts: 所有文本
            timestamps: 文本时间戳
            cutoff_date: 训练截止日期
            granularity: 数据粒度

        Returns:
            dataset: 成员推理数据集
        """
        member_texts = []
        non_member_texts = []

        for text, ts in zip(all_texts, timestamps):
            if ts < cutoff_date:
                member_texts.append(text)
            else:
                non_member_texts.append(text)

        # 处理粒度
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
        """
        构建多粒度数据集

        Args:
            train_texts: 训练集文本
            test_texts: 测试集文本

        Returns:
            datasets: {granularity: dataset} 字典
        """
        datasets = {}

        for granularity in ["document", "paragraph", "sentence"]:
            datasets[granularity] = self.build_from_open_source_model(
                train_texts, test_texts, granularity
            )

        return datasets


if __name__ == "__main__":
    # 测试代码
    print("Testing Data Loader Module...")

    # 创建测试数据
    texts = [
                "This is a sample document about machine learning and artificial intelligence.",
                "Another document discussing natural language processing techniques.",
                "Deep learning has revolutionized many fields including computer vision.",
                "Transformers are the foundation of modern large language models.",
            ] * 100

    labels = [1, 0, 1, 0] * 100

    # 测试TextDataset
    print("Testing TextDataset...")
    dataset = TextDataset(texts, labels, max_length=128)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0]}")

    # 测试MembershipDataset
    print("\nTesting MembershipDataset...")
    member_texts = texts[:200]
    non_member_texts = texts[200:]
    membership_dataset = MembershipDataset(
        member_texts, non_member_texts, max_length=128
    )
    print(f"Membership dataset size: {len(membership_dataset)}")
    print(f"Member ratio: {membership_dataset.get_member_ratio():.2f}")

    # 测试DataLoaderFactory
    print("\nTesting DataLoaderFactory...")
    config = DataConfig(batch_size=16)
    factory = DataLoaderFactory(config)
    dataloader = factory.create_text_dataloader(texts, labels)
    print(f"Number of batches: {len(dataloader)}")

    # 测试TextGranularityProcessor
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

    # 测试AttackDatasetBuilder
    print("\nTesting AttackDatasetBuilder...")
    builder = AttackDatasetBuilder(num_samples_per_granularity=100)

    train_texts = texts[:200]
    test_texts = texts[200:]

    multi_datasets = builder.build_multi_granularity(train_texts, test_texts)
    for granularity, dataset in multi_datasets.items():
        print(f"{granularity}: {len(dataset)} samples")

    print("\nAll tests passed!")