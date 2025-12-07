import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .datasets import VQADataset
from .collate import MRANCollator
from .model import MRANVQAModel
from .metrics import compute_classification_accuracy, compute_text_metrics
from .ags import AGSLoss


# ---------------------------------------------------------------------
# Vocab utilities
# ---------------------------------------------------------------------


def build_answer_vocab(train_json: str, vocab_path: str, top_k: int = 3000) -> None:
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for item in data:
        answers = item.get("answers", None)
        if answers:
            counter.update([a.lower() for a in answers])
        else:
            ans = item.get("answer", None)
            if ans:
                counter.update([ans.lower()])

    most_common = counter.most_common(top_k)
    ans2id = {ans: i for i, (ans, _) in enumerate(most_common)}
    id2ans = {i: ans for ans, i in ans2id.items()}

    vocab = {"ans2id": ans2id, "id2ans": id2ans}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"[Vocab] Saved vocab with {len(ans2id)} answers → {vocab_path}")


def load_answer_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    ans2id = vocab["ans2id"]
    # keys for id2ans may be str or int depending on how it was saved
    id2ans = {int(k): v for k, v in vocab["id2ans"].items()}
    return ans2id, id2ans


def attach_answer_ids(
    json_in: str,
    json_out: str,
    ans2id: Dict[str, int],
) -> None:
    with open(json_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        ans = item.get("answer", None)
        if ans is None:
            # fallback: use majority of "answers" if present
            answers = item.get("answers", [])
            if answers:
                # majority vote
                ans = Counter([a.lower() for a in answers]).most_common(1)[0][0]
        if ans is None:
            item["answer_id"] = -1
        else:
            item["answer_id"] = ans2id.get(ans.lower(), -1)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Vocab] Attached answer_id to {len(data)} samples → {json_out}")


# ---------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------


def create_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = VQADataset(cfg.train_json, cfg.image_root)
    val_ds = VQADataset(cfg.val_json, cfg.image_root)

    collator = MRANCollator(
        text_encoder_name=cfg.text_encoder_name,
        max_question_len=cfg.max_question_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_epoch(
    model: MRANVQAModel,
    loader: DataLoader,
    cfg: TrainingConfig,
    id2ans: Dict[int, str],
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    ce_loss = nn.CrossEntropyLoss()

    all_pred_ids: List[int] = []
    all_ref_ids: List[int] = []
    all_pred_str: List[str] = []
    all_ref_strs: List[List[str]] = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answer_ids = batch["answer_ids"].to(device)

            logits, _ = model(pixel_values, input_ids, attention_mask)
            loss = ce_loss(logits, answer_ids)

            B = answer_ids.size(0)
            acc = compute_classification_accuracy(logits, answer_ids)

            total_loss += loss.item() * B
            total_acc += acc * B
            total_count += B

            # text metrics
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            ref_ids = answer_ids.cpu().tolist()
            pred_str = [id2ans.get(i, "") for i in pred_ids]
            ref_str = [[id2ans.get(i, "")] for i in ref_ids]

            all_pred_ids.extend(pred_ids)
            all_ref_ids.extend(ref_ids)
            all_pred_str.extend(pred_str)
            all_ref_strs.extend(ref_str)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_acc / max(total_count, 1)
    text_metrics = compute_text_metrics(all_pred_str, all_ref_strs)

    metrics = {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }
    metrics.update(text_metrics)
    return metrics


def train_mran(cfg: TrainingConfig) -> None:
    set_seed(cfg.seed)

    # build vocab if needed
    if cfg.answer_vocab is None:
        raise ValueError("cfg.answer_vocab must be set (path to vocab json).")
    vocab_path = Path(cfg.answer_vocab)
    if not vocab_path.exists():
        assert cfg.train_json is not None, "train_json required to build vocab"
        build_answer_vocab(cfg.train_json, cfg.answer_vocab, top_k=cfg.num_answers)

    ans2id, id2ans = load_answer_vocab(cfg.answer_vocab)
    cfg.num_answers = len(ans2id)
    print(f"[Config] num_answers = {cfg.num_answers}")

    device = cfg.device
    train_loader, val_loader = create_dataloaders(cfg)

    model = MRANVQAModel(cfg, num_answers=cfg.num_answers).to(device)
    ce_loss = nn.CrossEntropyLoss()
    ags_loss = AGSLoss(lam=0.5) if cfg.use_ags_loss else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    best_ckpt_path = Path(cfg.output_dir) / f"mran_{cfg.dataset}_best.pt"

    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        count = 0

        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            ags_masks = batch["ags_masks"]
            if ags_masks is not None:
                ags_masks = ags_masks.to(device)

            optimizer.zero_grad()
            logits, attn = model(pixel_values, input_ids, attention_mask)
            loss = ce_loss(logits, answer_ids)

            if cfg.use_ags_loss and ags_masks is not None and attn is not None:
                # ensure mask and attn shapes match
                if ags_masks.dim() == 2 and ags_masks.size(1) == attn.size(1):
                    loss_ags = ags_loss(attn, ags_masks)
                    loss = loss + cfg.gamma_ags * loss_ags

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            B = answer_ids.size(0)
            acc = compute_classification_accuracy(logits, answer_ids)
            running_loss += loss.item() * B
            running_acc += acc * B
            count += B
            global_step += 1

            if (i + 1) % cfg.print_every == 0:
                print(
                    f"[Epoch {epoch+1}/{cfg.epochs}] "
                    f"step {i+1}/{len(train_loader)} "
                    f"loss={running_loss/count:.4f} acc={running_acc/count:.4f}"
                )

        # validation
        val_metrics = evaluate_epoch(model, val_loader, cfg, id2ans, device)
        print(
            f"[Val][Epoch {epoch+1}] "
            f"loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"BLEU={val_metrics['BLEU']:.4f} "
            f"METEOR={val_metrics['METEOR']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_dict(),
                    "ans2id": ans2id,
                    "id2ans": id2ans,
                },
                best_ckpt_path,
            )
            print(f"[Checkpoint] Saved best model → {best_ckpt_path}")


def print_mran_stats(cfg: TrainingConfig) -> None:
    """
    Instantiate a model and print:
      - number of parameters
      - single-sample forward latency (approx.)
    """
    ans2id, id2ans = load_answer_vocab(cfg.answer_vocab)
    cfg.num_answers = len(ans2id)

    device = cfg.device
    model = MRANVQAModel(cfg, num_answers=cfg.num_answers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Stats] Trainable parameters: {n_params/1e6:.2f}M")

    # dummy forward
    B = 1
    dummy_img = torch.randn(B, 3, 224, 224, device=device)
    dummy_input_ids = torch.ones(B, cfg.max_question_len, dtype=torch.long, device=device)
    dummy_attn = torch.ones(B, cfg.max_question_len, dtype=torch.long, device=device)

    import time

    model.eval()
    with torch.no_grad():
        # warm-up
        for _ in range(5):
            _ = model(dummy_img, dummy_input_ids, dummy_attn)

        iters = 20
        t0 = time.time()
        for _ in range(iters):
            _ = model(dummy_img, dummy_input_ids, dummy_attn)
        t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"[Stats] Single-sample latency (batch=1): {avg_ms:.2f} ms")
