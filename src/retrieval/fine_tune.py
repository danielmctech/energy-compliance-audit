"""Two-stage LoRA fine-tuning of the dense encoder.

Both stages train the same SentenceTransformer base (MiniLM) with a PEFT
LoRA adapter on (query, positive, negatives) rows using the
MultipleNegativesRanking loss, so every other row in the batch is an extra
negative.

Stage 1 (contrastive)
    rows = pairs_stage1.jsonl (from pair_builder). Each pair is
    unrolled into one row per negative so the explicit negatives are used
    as hard in-batch distractors *in addition to* cross-row negatives.

Stage 2 (hard-negative mining)
    The Stage-1 adapter is loaded and re-trained. For every query, the
    whole corpus is re-ranked with the Stage-1 model and the top-k nearest
    non-positive chunks become hard negatives.

Outputs (gitignored under notebooks/data/retrieval/):
    finetuned_stage1/adapter/   -- Stage 1 LoRA
    finetuned_stage2/adapter/   -- Stage 2 LoRA (continues Stage 1)

Both adapter dirs are a full SentenceTransformer model dir, so
`SentenceTransformer("<that dir>")` loads them for A/B comparison.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from ._corpus import load_corpus
from . import pair_builder as PB

_BASE = "all-MiniLM-L6-v2"
_LORA_R, _LORA_ALPHA, _DROPOUT = 16, 32, 0.05


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rows_from_pairs(pairs: List[dict]) -> List[dict]:
    """Unroll (query, positive, negatives*) into one row per negative so the
    explicit negatives are in-batch hard distractors too."""
    rows = []
    for p in pairs:
        pool = [p["positive"]] + p["negatives"]
        for i in range(len(p["negatives"])):
            negs = pool[:i] + pool[i + 1:]
            rows.append({
                "anchor": p["query"],
                "positive": pool[i]["text"],
                "negatives": [n["text"] for n in negs],
                "positive_id": pool[i]["lineage_id"],
                "kind": p["kind"],
                "doc": p["doc_id"],
            })
    return rows


def _train(rows: List[dict], out_dir: Path, stage: str,
           start_adapter: Optional[Path] = None,
           epochs: int = 3, batch_size: int = 32, lr: float = 2e-5,
           seed: int = 2024) -> Path:
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from sentence_transformers import (SentenceTransformer,
                                       SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(_BASE)
    if start_adapter is not None:
        # continue training the previous stage's adapter
        model.load_adapter(str(start_adapter))
        for n, p in model[0].auto_model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
    else:
        model.add_adapter(LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=_LORA_R, lora_alpha=_LORA_ALPHA, lora_dropout=_DROPOUT))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{stage}] {len(rows)} rows, trainable params = {trainable:,}")

    ds = Dataset.from_list(rows)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=ds,
        loss=MultipleNegativesRankingLoss(model))
    trainer.train()

    adapter = out_dir / f"finetuned_{stage}" / "adapter"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter))
    return adapter


def run(
    out_dir: Optional[Path] = None,
    pairs_file: Optional[Path] = None,
    max_texts: Optional[int] = None,
    stage1_epochs: int = 3,
    stage2_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    stage2_k: int = 6,
    stage2_max: Optional[int] = None,
    seed: int = 2024,
) -> dict:
    import common as c

    out = out_dir or (c.NOTEBOOKS_DATA / "retrieval")
    pairs_file = pairs_file or (out / "pairs_stage1.jsonl")
    if not pairs_file.exists():
        pairs = PB.build_pairs()
        PB.save_pairs(pairs, out)
        print("  built + saved", len(pairs), "pairs ->", pairs_file.name)

    pairs = [json.loads(l) for l in pairs_file.read_text().splitlines()
             if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_texts:
        pairs = pairs[:max_texts]

    rows1 = _rows_from_pairs(pairs)
    _seed_everything(seed)
    s1 = _train(rows1, out, "stage1", epochs=stage1_epochs,
                batch_size=batch_size, lr=learning_rate, seed=seed)

    # ---- stage 2: mine hard negatives with the Stage-1 model -----------
    corpus = load_corpus()
    base = "all-MiniLM-L6-v2"
    from sentence_transformers import SentenceTransformer as _ST
    st_model = _ST(_BASE)
    st_model.load_adapter(str(s1))
    # reuse stage2_rows against the loaded adapter
    rows2 = _stage2_rows_mined(rows1, corpus, st_model, stage2_k)
    rng2 = random.Random(seed + 1)
    rng2.shuffle(rows2)
    if stage2_max:
        rows2 = rows2[:stage2_max]
    print(f"  [stage2] hard-negative rows: {len(rows2)}")

    _seed_everything(seed)
    s2 = _train(rows2, out, "stage2", start_adapter=s1,
                epochs=stage2_epochs, batch_size=batch_size,
                lr=learning_rate, seed=seed)
    return {"stage1": str(s1), "stage2": str(s2)}


def _stage2_rows_mined(dataset_rows, corpus, st_model, k: int) -> List[dict]:
    import faiss

    texts = [c.text for c in corpus["chunks"]]
    qv = st_model.encode([r["anchor"] for r in dataset_rows],
                         normalize_embeddings=True, convert_to_numpy=True,
                         batch_size=32, show_progress_bar=False)
    cv = st_model.encode(texts, normalize_embeddings=True,
                         convert_to_numpy=True, batch_size=128,
                         show_progress_bar=False)
    idx = faiss.IndexFlatIP(cv.shape[1])
    idx.add(np.asarray(cv, dtype="float32"))
    scores, nbrs = idx.search(qv, 1 + k)
    id_to_i = {c.lineage_id: i for i, c in enumerate(corpus["chunks"])}
    out: List[dict] = []
    for i, r in enumerate(dataset_rows):
        pos_i = id_to_i.get(r["positive_id"])
        hard = []
        for j, s in zip(nbrs[i], scores[i]):
            j = int(j)
            if j == pos_i or corpus["chunks"][j].lineage_id == r["positive_id"]:
                continue
            hard.append(texts[j])
            if len(hard) >= 4:
                break
        if not hard:
            continue
        out.append({"anchor": r["anchor"], "positive": r["positive"],
                    "negatives": hard, "positive_id": r["positive_id"],
                    "kind": r["kind"], "doc": r["doc"]})
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-texts", type=int, default=None)
    ap.add_argument("--stage1-epochs", type=int, default=3)
    ap.add_argument("--stage2-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--stage2-k", type=int, default=6)
    args = ap.parse_args()
    res = run(max_texts=args.max_texts,
              stage1_epochs=args.stage1_epochs,
              stage2_epochs=args.stage2_epochs,
              batch_size=args.batch_size, learning_rate=args.lr,
              stage2_k=args.stage2_k)
    for k, v in res.items():
        print(k, "->", v)
