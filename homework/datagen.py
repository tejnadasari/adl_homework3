import json
import time
import torch
from pathlib import Path
from typing import List, Tuple

from .cot import CoTModel
from .data import Dataset

def extract_answer(text: str) -> float | None:
    import re
    m = re.search(r"<answer>(.*?)</answer>", text)
    if not m:
        return None
    try:
        return float(m.group(1).strip())
    except:
        return None


def is_correct(pred: float, true: float, tol: float = 1e-3) -> bool:
    return abs(pred - true) <= tol


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    trainset = Dataset("train")

    cot_llm = CoTModel()

    
    MAX_OVERSAMPLE = 6        
    BATCH_SIZE = 4             
    MAX_NEW_TOKENS = 64       

    effective_oversample = min(oversample, MAX_OVERSAMPLE)
    print(f"\nEffective oversample = {effective_oversample}")
    print(f"Batch size           = {BATCH_SIZE}")
    print(f"Max new tokens       = {MAX_NEW_TOKENS}")
    print("temperature          =", temperature)
    print("\nGenerating...\n")

    results: List[Tuple[str, float, str]] = []

    total = len(trainset)
    for start in range(0, total, BATCH_SIZE):
        batch = trainset[start:start + BATCH_SIZE]

        formatted_prompts = []
        gold_answers = []

        for q, ans in batch:
            formatted_prompts.append(cot_llm.format_prompt(q))
            gold_answers.append(ans)

        try:
            outputs = cot_llm.batched_generate(
                prompts=formatted_prompts,
                num_return_sequences=effective_oversample,
                temperature=temperature
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            time.sleep(2)

            safe_oversample = 3
            outputs = cot_llm.batched_generate(
                prompts=formatted_prompts,
                num_return_sequences=safe_oversample,
                temperature=temperature
            )

        for i, (q, true_ans) in enumerate(batch):
            completions = outputs[i]
            accepted = None

            for comp in completions:
                pred = extract_answer(comp)
                if pred is not None and is_correct(pred, true_ans):
                    accepted = comp
                    break

            if accepted is not None:
                results.append([q, float(true_ans), accepted])

        print(f"Processed {min(start+BATCH_SIZE, total)}/{total} — "
              f"Accepted so far: {len(results)}")

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n============================================")
    print(f"RFT dataset saved to: {output_json}")
    print(f"Total ORFT accepted samples: {len(results)}")
    print("============================================\n")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)
