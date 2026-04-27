"""
Domain-specialized reference model training via asymmetric distillation.
"""

import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import TextDataset
from token_alignment import token_aligned_values_from_char_spans

logger = logging.getLogger(__name__)


def compute_hard_loss(logits, labels, mask):
    s_logits = logits[:, :-1, :].contiguous()
    s_labels = labels[:, 1:].contiguous()
    s_mask = mask[:, 1:].contiguous().float()

    batch, seq_len, vocab = s_logits.shape
    loss = nn.CrossEntropyLoss(reduction="none")(s_logits.view(-1, vocab), s_labels.view(-1))
    loss = (loss.view(batch, seq_len) * s_mask).sum(dim=1) / s_mask.sum(dim=1).clamp(min=1)
    return loss.mean()


def compute_imitation_loss(logits, labels, mask, aligned_target_probs):
    s_logits = logits[:, :-1, :].contiguous()
    s_labels = labels[:, 1:].contiguous()
    s_mask = mask[:, 1:].contiguous().float()
    s_probs = aligned_target_probs[:, : s_labels.size(1)].contiguous()

    batch, seq_len, vocab = s_logits.shape
    loss = nn.CrossEntropyLoss(reduction="none")(s_logits.view(-1, vocab), s_labels.view(-1))
    weighted = (loss.view(batch, seq_len) * s_probs * s_mask).sum(dim=1) / s_mask.sum(dim=1).clamp(min=1)
    return weighted.mean()


def compute_asymmetric_loss(logits, labels, mask, aligned_target_probs, epsilon, is_member):
    l_hard = compute_hard_loss(logits, labels, mask)
    l_imit = compute_imitation_loss(logits, labels, mask, aligned_target_probs)
    return (1 - epsilon) * l_hard + epsilon * l_imit if is_member else epsilon * l_hard + (1 - epsilon) * l_imit


class _RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, global_indices):
        self.base = base_dataset
        self.global_indices = global_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["idx"] = self.global_indices[idx]
        return item


class ReferenceModelTrainer:
    def __init__(
        self,
        ref_model_name,
        epsilon=0.3,
        stage1_epochs=15,
        stage2_epochs=10,
        lr=5e-4,
        weight_decay=0.01,
        batch_size=32,
        device="cuda",
        use_fp16=True,
        cache_dir=None,
    ):
        self.ref_model_name = ref_model_name
        self.epsilon = epsilon
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.cache_dir = cache_dir

    def load_base_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.ref_model_name, cache_dir=self.cache_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_base_artifacts(self):
        tokenizer = self.load_base_tokenizer()
        model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
        )
        return model.to(self.device), tokenizer

    def _build_aligned_target_probs(self, batch, target_metadata_dict):
        aligned = []
        offset_mapping = batch["offset_mapping"][:, 1:, :].cpu().numpy()
        shifted_mask = batch["attention_mask"][:, 1:].cpu().numpy().astype(bool)
        texts = batch["text"]
        indices = batch["idx"]
        max_len = offset_mapping.shape[1]

        for row_idx, idx_val in enumerate(indices):
            meta = target_metadata_dict[int(idx_val.item())]
            ref_offsets = offset_mapping[row_idx][shifted_mask[row_idx]].tolist()
            weights = token_aligned_values_from_char_spans(
                text_length=len(texts[row_idx]),
                source_offsets=meta["offsets"],
                source_values=meta["token_probs"],
                target_offsets=ref_offsets,
                default_value=0.0,
            )
            padded = np.zeros(max_len, dtype=np.float32)
            padded[: len(weights)] = weights
            aligned.append(padded)
        return torch.tensor(aligned, dtype=torch.float32, device=self.device)

    def _train_stage(self, model, dataloader, target_metadata_dict, epochs, is_member, stage_name):
        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if self.use_fp16 else None
        model.train()

        for epoch in range(epochs):
            total_loss, n_batches = 0.0, 0
            for batch in dataloader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                aligned_target_probs = self._build_aligned_target_probs(batch, target_metadata_dict)

                optimizer.zero_grad()
                if self.use_fp16:
                    with torch.amp.autocast("cuda"):
                        out = model(input_ids=ids, attention_mask=mask)
                        loss = compute_asymmetric_loss(out.logits, ids, mask, aligned_target_probs, self.epsilon, is_member)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(input_ids=ids, attention_mask=mask)
                    loss = compute_asymmetric_loss(out.logits, ids, mask, aligned_target_probs, self.epsilon, is_member)
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            avg = total_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("  %s Epoch %d/%d Loss: %.4f", stage_name, epoch + 1, epochs, avg)
        return model

    def train_reference_model(self, domain_id, member_loader, nonmember_loader, target_metadata_dict, save_dir=None):
        logger.info("=== Training reference model for domain %d ===", domain_id)
        model, tokenizer = self.load_base_artifacts()

        logger.info("  Stage 1: Non-member alignment (%d epochs)", self.stage1_epochs)
        model = self._train_stage(
            model,
            nonmember_loader,
            target_metadata_dict,
            self.stage1_epochs,
            is_member=False,
            stage_name=f"[D{domain_id}][S1]",
        )

        logger.info("  Stage 2: Member memorization (%d epochs)", self.stage2_epochs)
        model = self._train_stage(
            model,
            member_loader,
            target_metadata_dict,
            self.stage2_epochs,
            is_member=True,
            stage_name=f"[D{domain_id}][S2]",
        )

        if save_dir:
            path = os.path.join(save_dir, f"ref_model_domain_{domain_id}")
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
            logger.info("  Saved to %s", path)

        model.eval()
        return {"model": model, "tokenizer": tokenizer, "name": self.ref_model_name}


def train_all_reference_models(ref_model_name, domains, texts, labels, target_metadata, config):
    target_metadata_dict = {i: p for i, p in enumerate(target_metadata)}

    probe_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        cache_dir=config.data.cache_dir,
        torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
    )
    num_params = int(sum(p.numel() for p in probe_model.parameters()))
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer = ReferenceModelTrainer(
        ref_model_name=ref_model_name,
        epsilon=config.distillation.epsilon,
        stage1_epochs=config.distillation.stage1_epochs,
        stage2_epochs=config.distillation.stage2_epochs,
        lr=config.get_lr(num_params),
        weight_decay=config.distillation.weight_decay,
        batch_size=config.distillation.batch_size,
        device=config.device,
        use_fp16=config.use_fp16,
        cache_dir=config.data.cache_dir,
    )

    logger.info(
        "Reference model base: %s | params=%d | lr=%.2e",
        ref_model_name,
        num_params,
        trainer.lr,
    )

    save_dir = os.path.join(config.output_dir, "reference_models")
    os.makedirs(save_dir, exist_ok=True)
    reference_models: Dict[int, dict] = {}

    for domain_id in range(config.domain.num_domains):
        if domain_id not in domains:
            raise ValueError(f"Missing domain {domain_id}; expected 0..{config.domain.num_domains - 1}.")

        info = domains[domain_id]
        m_idx = info["member_indices"]
        nm_idx = info["nonmember_indices"]
        if len(m_idx) == 0 or len(nm_idx) == 0:
            raise ValueError(
                f"Domain {domain_id} is invalid for strict paper-aligned training: "
                f"m={len(m_idx)}, nm={len(nm_idx)}."
            )

        ref_tokenizer = trainer.load_base_tokenizer()

        m_ds = TextDataset([texts[i] for i in m_idx], [labels[i] for i in m_idx], ref_tokenizer, config.model.max_seq_length)
        nm_ds = TextDataset([texts[i] for i in nm_idx], [labels[i] for i in nm_idx], ref_tokenizer, config.model.max_seq_length)

        m_loader = DataLoader(
            _RemappedDataset(m_ds, m_idx),
            batch_size=config.distillation.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        nm_loader = DataLoader(
            _RemappedDataset(nm_ds, nm_idx),
            batch_size=config.distillation.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        reference_models[domain_id] = trainer.train_reference_model(
            domain_id,
            m_loader,
            nm_loader,
            target_metadata_dict,
            save_dir,
        )

    if len(reference_models) != config.domain.num_domains:
        raise RuntimeError(
            f"Expected exactly {config.domain.num_domains} reference models, got {len(reference_models)}."
        )

    logger.info("Trained %d/%d reference models", len(reference_models), config.domain.num_domains)
    return reference_models
