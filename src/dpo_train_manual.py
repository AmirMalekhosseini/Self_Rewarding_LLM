import argparse
import json
import math
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import sys
import os

project_root = '/content/drive/MyDrive/Self-Rewarding-LLM'
if project_root not in sys.path:
    sys.path.append(project_root)


from src.lora import add_lora
from src.tokenizer import to_chat

# Define the base model name for reliability
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = (
    'Respond to the following user query in a comprehensive and detailed way. '
    'But first write down your internal thoughts. This must include your draft response '
    'and its evaluation. After this, write your final response after "<R>".'
)

class PreferenceDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if "prompt" in ex and "chosen" in ex and "rejected" in ex:
                    self.rows.append(ex)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def total_logprob(model, tokenizer, prompt, answer, device, max_length=1024):
    text = to_chat(tokenizer, SYSTEM_PROMPT, prompt, answer)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    labels = enc["input_ids"].clone()

    # Add a check to prevent empty labels
    if labels.shape[1] == 0:
        return torch.tensor(0.0)

    with torch.no_grad():
        out = model(**enc, labels=labels)

    n_tok = labels.numel()
    return -out.loss.item() * n_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_model", required=True)
    parser.add_argument("--policy_model", required=True)
    parser.add_argument("--prefs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FIX: Load the tokenizer from the original base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32

    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model, torch_dtype=dtype, device_map="auto")
    ref_model.eval()

    policy_model = AutoModelForCausalLM.from_pretrained(args.policy_model, torch_dtype=dtype, device_map="auto")
    policy_model = add_lora(policy_model)
    policy_model.train()

    ds = PreferenceDataset(args.prefs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(policy_model.parameters(), lr=args.lr)
    total_steps = len(dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(100, total_steps // 10), num_training_steps=total_steps)

    pbar = tqdm(total=total_steps)
    beta = args.beta

    for epoch in range(args.epochs):
        for batch in dl:
            prompt = batch["prompt"][0]
            chosen = batch["chosen"][0]
            rejected = batch["rejected"][0]

            # Use a separate forward pass for the policy model to compute gradients
            policy_model.train()

            # Policy model forward pass for chosen response
            chosen_text = to_chat(tokenizer, SYSTEM_PROMPT, prompt, chosen)
            chosen_enc = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            chosen_labels = chosen_enc["input_ids"].clone()

            # Policy model forward pass for rejected response
            rejected_text = to_chat(tokenizer, SYSTEM_PROMPT, prompt, rejected)
            rejected_enc = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            rejected_labels = rejected_enc["input_ids"].clone()

            # Reference model log probabilities (no gradients)
            lp_ref_c = total_logprob(ref_model, tokenizer, prompt, chosen, device, args.max_length)
            lp_ref_r = total_logprob(ref_model, tokenizer, prompt, rejected, device, args.max_length)

            # Policy model log probabilities (with gradients)
            policy_chosen_out = policy_model(**chosen_enc, labels=chosen_labels)
            policy_rejected_out = policy_model(**rejected_enc, labels=rejected_labels)

            lp_pi_c = -policy_chosen_out.loss * chosen_labels.numel()
            lp_pi_r = -policy_rejected_out.loss * rejected_labels.numel()

            # DPO loss calculation
            diff_pi = lp_pi_c - lp_pi_r
            diff_ref = lp_ref_c - lp_ref_r
            adv = diff_pi - diff_ref

            loss = -torch.nn.functional.logsigmoid(beta * adv).mean()
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(f"epoch {epoch} dpo_loss {loss.item():.4f}")
            pbar.update(1)

    os.makedirs(args.out, exist_ok=True)
    policy_model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"✅ Saved DPO policy to {args.out}")


if __name__ == "__main__":
    main()
