import json
import argparse
import os
from collections import defaultdict
from typing import List, Dict


def read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", required=True, help="Input JSONL with prompt, completion, score")
    parser.add_argument("--out_file", required=True, help="Output preference JSONL")
    args = parser.parse_args()

    rows = read_jsonl(args.in_file)

    # Group by prompt
    prompt_map = defaultdict(list)
    for row in rows:
        prompt_map[row["prompt"]].append(row)

    pref_rows = []
    for prompt, completions in prompt_map.items():
        if len(completions) < 2:
            continue

        # Sort by score descending
        sorted_comps = sorted(completions, key=lambda x: x["score"], reverse=True)
        chosen = sorted_comps[0]["completion"]
        rejected = sorted_comps[-1]["completion"]

        if chosen.strip() == rejected.strip():
            continue

        pref_rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    write_jsonl(args.out_file, pref_rows)
    print(f"✅ Wrote {len(pref_rows)} preference pairs to {args.out_file}")


if __name__ == "__main__":
    main()
