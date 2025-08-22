import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
import os
sys.path.append("/content/drive/MyDrive/Self-Rewarding-LLM")


from src.tokenizer import load_tokenizer, to_chat
from src.lora import add_lora



SYSTEM_PROMPT = (
    'Respond to the following user query in a comprehensive and detailed way. '
    'But first write down your internal thoughts. This must include your draft response '
    'and its evaluation. After this, write your final response after "<R>".'
)

def collate(tokenizer, batch, max_length=1024):
    texts = []
    for x in batch:
        text = to_chat(tokenizer, SYSTEM_PROMPT, x["prompt"], x["response"])
        texts.append(text)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model = add_lora(model)
    model.train()

    ds = load_dataset("json", data_files=args.dataset)["train"]
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(tokenizer, b, args.max_length)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(50, num_training_steps // 10),
        num_training_steps=num_training_steps
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")
            pbar.update(1)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

if __name__ == "__main__":
    main()

