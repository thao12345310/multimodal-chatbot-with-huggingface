from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)

print("Downloading dataset...")
ds = load_dataset("bkai-foundation-models/vi-alpaca", split="train")
ds = ds.filter(lambda x: len(x["output"]) > 80)
ds = ds.shuffle(seed=42).select(range(8000))
split = ds.train_test_split(test_size=0.05, seed=42)

def to_chatml(sample) -> str:
    instr = sample["instruction"]
    inp   = sample.get("input", "")
    out   = sample["output"]
    user  = instr + (f"\n\nNgữ cảnh: {inp}" if inp else "")
    return (
        "<|im_start|>system\nBạn là trợ lý AI hữu ích.<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )

def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps({"text": to_chatml(item)}, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} samples → {path}")

save_jsonl(split["train"], "data/train.jsonl")
save_jsonl(split["test"],  "data/valid.jsonl")
print("Done!")