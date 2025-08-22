
import sys
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup Project Path
project_root = '/content/drive/MyDrive/Self-Rewarding-LLM'
if project_root not in sys.path:
    sys.path.append(project_root)

from src.tokenizer import to_chat

# Configuration
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = os.path.join(project_root, "checkpoints/sft")
DATA_PATH = os.path.join(project_root, "data/sft_openassistant.jsonl")
OUTPUT_PATH = os.path.join(project_root, "data/sft_completions_100.jsonl") # Changed output file name
MAX_NEW_TOKENS = 256
MAX_LENGTH = 2048
SYSTEM_PROMPT = "Respond to the following user query in a helpful way."


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model and tokenizer loaded successfully.")

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]

    examples = examples[:100]

    print(f"Loaded and reduced to {len(examples)} prompts.")

except FileNotFoundError:
    print(f"[ERROR] Data file not found at {DATA_PATH}. Please check the path.")
    sys.exit(1)

results = []
with torch.no_grad():
    for i, ex in enumerate(tqdm(examples, desc="Generating Completions")):
        try:
            prompt = ex.get("prompt")
            if not prompt:
                continue

            chat_input = to_chat(tokenizer, SYSTEM_PROMPT, prompt)
            inputs = tokenizer(chat_input, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if inputs["input_ids"].shape[1] == 0:
                continue

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_length:]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

            results.append({
                "prompt": prompt,
                "completion": decoded.strip()
            })

        except Exception as e:
            print(f"[ERROR] Generation failed for example #{i}: {e}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(results)} completions to {OUTPUT_PATH}")
