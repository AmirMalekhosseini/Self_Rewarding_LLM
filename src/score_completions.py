import argparse
import json
import os
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_pair_text(prompt: str, completion: str) -> str:
    # Simple, RM-friendly formatting
    return f"Human: {prompt}\n\nAssistant: {completion}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True, help="JSONL with {'prompt','completion'} records")
    parser.add_argument("--out_file", required=True, help="Where to save the scored JSONL")
    parser.add_argument("--rm_model", default="OpenAssistant/reward-model-deberta-v3-large-v2",
                        help="Reward model name (HF hub path)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading reward model: {args.rm_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.rm_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.rm_model).to(device)
    model.eval()

    print(f"Reading: {args.in_file}")
    rows = read_jsonl(args.in_file)

    texts = [build_pair_text(r["prompt"], r["completion"]) for r in rows]

    all_scores: List[float] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Scoring"):
            batch_texts = texts[i:i + args.batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            ).to(device)

            logits = model(**enc).logits  # shape: (B, 1) for OA RM (regression)
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1).tolist()
            else:
                # Fallback if it's a classifier with >1 labels: take the first logit or softmax
                probs = torch.softmax(logits, dim=-1)
                # Take "helpful"/"good" class prob assuming label 1 if binary
                scores = probs[:, -1].tolist()

            all_scores.extend(scores)

    assert len(all_scores) == len(rows)

    # Attach scores & save
    for r, s in zip(rows, all_scores):
        r["score"] = float(s)

    write_jsonl(args.out_file, rows)
    print(f"✅ Wrote {len(rows)} scored rows to {args.out_file}")


if __name__ == "__main__":
    main()
