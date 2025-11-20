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
    output_json: str = "data/rft.json",
    oversample: int = 25,
    temperature: float = 0.7
):
    trainset = Dataset("train")
    cot_llm = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    
    # CRITICAL: Reduced batch size for oversample=25
    BATCH_SIZE = 3  # 3 questions × 25 sequences = 75 (manageable)
    
    print("\n" + "="*60)
    print("RFT Dataset Generation")
    print("="*60)
    print(f"Train questions:      {len(trainset)}")
    print(f"Batch size:           {BATCH_SIZE}")
    print(f"Oversample:           {oversample}")
    print(f"Temperature:          {temperature}")
    print(f"Parallel generations: {BATCH_SIZE * oversample}")
    print("="*60 + "\n")
    
    # Clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB\n")
    
    accepted_samples: List[Tuple[str, float, str]] = []
    start_time = time.time()
    
    for start in range(0, len(trainset), BATCH_SIZE):
        # Aggressive memory cleanup before each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        batch = trainset[start:start + BATCH_SIZE]
        prompts = [cot_llm.format_prompt(q) for q, _ in batch]
        gold = [ans for _, ans in batch]
        
        try:
            completions = cot_llm.batched_generate(
                prompts=prompts,
                num_return_sequences=oversample,
                temperature=temperature,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n⚠️  OOM at batch {start}, attempting recovery...")
            
            # Aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(3)  # Give GPU time to clear
            
            # Try with reduced oversample
            reduced_oversample = max(10, oversample // 2)
            print(f"   Retrying with oversample={reduced_oversample}...")
            
            try:
                completions = cot_llm.batched_generate(
                    prompts=prompts,
                    num_return_sequences=reduced_oversample,
                    temperature=temperature,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"   Still OOM, processing one by one...")
                # Last resort: process questions one at a time
                completions = []
                for single_prompt in prompts:
                    try:
                        single_completion = cot_llm.batched_generate(
                            [single_prompt],
                            num_return_sequences=5,  # Very conservative
                            temperature=temperature,
                        )
                        completions.append(single_completion[0])
                    except:
                        completions.append([])  # Empty list for failed generations
        
        # Process results
        for i, (q, true_ans) in enumerate(batch):
            if i >= len(completions):
                continue
            
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
        
        # Progress update
        progress = min(start + BATCH_SIZE, len(trainset))
        acceptance_rate = len(accepted_samples) / progress if progress > 0 else 0
        elapsed = time.time() - start_time
        eta = (elapsed / progress) * (len(trainset) - progress) if progress > 0 else 0
        
        print(f"[{progress:4d}/{len(trainset)}] "
              f"Accepted: {len(accepted_samples):4d} "
              f"({acceptance_rate*100:.1f}%) "
              f"| ETA: {eta/60:.1f} min")
    
    # Save results
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(accepted_samples, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    acceptance_rate = len(accepted_samples) / len(trainset)
    
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Total accepted:       {len(accepted_samples)}")
    print(f"Acceptance rate:      {acceptance_rate*100:.1f}%")
    print(f"Total time:           {total_time/60:.1f} minutes")
    print(f"Saved to:             {output_json}")
    print("="*60)
    
    # Quality check
    if len(accepted_samples) >= 900:
        print("\n✅ EXCELLENT: 900+ pairs! Expected RFT: 0.80-0.85+")
    elif len(accepted_samples) >= 850:
        print("\n✅ GOOD: 850+ pairs! Expected RFT: 0.78-0.82")
    elif len(accepted_samples) >= 750:
        print("\n⚠️  OK: 750+ pairs. Expected RFT: 0.73-0.77")
        print("   Consider rerunning with higher temperature")
    else:
        print("\n❌ WARNING: <750 pairs may not be enough!")
        print("   Recommend rerunning with improved CoT prompt")
    
    print("\n")


if __name__ == "__main__":
    from fire import Fire
    Fire(generate_dataset)