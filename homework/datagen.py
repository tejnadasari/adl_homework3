import json
import time
import torch
import gc
import re
from pathlib import Path
from typing import List, Tuple

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def extract_answer(text: str) -> float | None:
    """
    Extract <answer>...</answer> and convert to float.
    """
    m = re.search(r"<answer>(.*?)</answer>", text)
    if not m:
        return None
    try:
        return float(m.group(1).strip())
    except:
        return None


def generate_dataset(
    output_json: str,
    oversample: int = 12,
    temperature: float = 0.4
):

    trainset = Dataset("train")
    cot_llm = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    BATCH_SIZE = 6 

    print("\n=======================================")
    print("Starting RFT Dataset Generation")
    print(f"Train questions:      {len(trainset)}")
    print(f"Oversample:           {oversample}")
    print(f"Temperature:          {temperature}")
    print("=======================================\n")

    accepted_samples: List[Tuple[str, float, str]] = []

    for start in range(0, len(trainset), BATCH_SIZE):

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch = trainset[start:start + BATCH_SIZE]

        prompts = [cot_llm.format_prompt(q) for q, _ in batch]
        gold = [ans for _, ans in batch]

        try:
            completions = cot_llm.batched_generate(
                prompts=prompts,
                num_return_sequences=oversample,
                temperature=temperature,
            )
        except torch.cuda.OutOfMemoryError:
            completions = cot_llm.batched_generate(
                prompts=prompts,
                num_return_sequences=6,
                temperature=temperature,
            )

        for i, (q, true_ans) in enumerate(batch):
            chains = completions[i]

            accepted = None
            for chain in chains:

                pred = extract_answer(chain)
                if pred is None:
                    continue

                if is_answer_valid(pred, true_ans):
                    accepted = chain
                    break

            if accepted:
                accepted_samples.append([q, float(true_ans), accepted])

        print(
            f"Processed: {min(start + BATCH_SIZE, len(trainset))}/{len(trainset)}   "
            f"Accepted so far: {len(accepted_samples)}"
        )

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(accepted_samples, f, indent=2)

    print("\n=======================================")
    print(f"Saved dataset: {output_json}")
    print(f"Total accepted: {len(accepted_samples)}")
    print("=======================================\n")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)

