from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct"):
        super().__init__(checkpoint)
    
    def format_prompt(self, question: str) -> str:
      messages = [
          {
              "role": "system",
              "content": (
                  "Convert units step-by-step. "
                  "CRITICAL: KB and MB use 1000, not 1024. "
                  "Show calculation. Put final number in <answer>number</answer>."
              )
          },
          # Example 1: kB→bit with 1000 (most critical failure)
          {
              "role": "user",
              "content": "What is 3 KB in bit?"
          },
          {
              "role": "assistant",
              "content": "1 KB = 1000 bytes (not 1024). 1 byte = 8 bits. 3 × 1000 × 8 = 24000. <answer>24000</answer>"
          },
          # Example 2: Division conversion (ft→yd type)
          {
              "role": "user",
              "content": "Convert 9 feet to yards."
          },
          {
              "role": "assistant",
              "content": "1 yard = 3 feet. Divide: 9 ÷ 3 = 3. <answer>3</answer>"
          },
          # Example 3: mph→m/s with explicit units
          {
              "role": "user",
              "content": "Convert 10 mi/h to m/s."
          },
          {
              "role": "assistant",
              "content": "1 mile = 1609.34 m. 1 hour = 3600 s. 10 × 1609.34 ÷ 3600 = 4.470. <answer>4.470</answer>"
          },
          # Example 4: Years→weeks (addresses failure #1)
          {
              "role": "user",
              "content": "How many weeks in 2 years?"
          },
          {
              "role": "assistant",
              "content": "1 year = 52.1775 weeks. 2 × 52.1775 = 104.355. <answer>104.355</answer>"
          },
          # Example 5: Basic multiplication
          {
              "role": "user",
              "content": "How many grams in 5 kg?"
          },
          {
              "role": "assistant",
              "content": "1 kg = 1000 g. 5 × 1000 = 5000. <answer>5000</answer>"
          },
          {
              "role": "user",
              "content": question.strip()
          }
      ]
      return self.tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=True,
          tokenize=False,
      )


def load():
    return CoTModel()


def test():
    """Test the CoT model accuracy"""
    from .data import Dataset, benchmark
    
    print("Loading CoT model...")
    model = load()
    
    print("Loading validation dataset...")
    dataset = Dataset("valid")
    
    print(f"Testing on {len(dataset)} examples...")
    result = benchmark(model, dataset, max_question=100)
    
    print(f"\n{'='*50}")
    print(f"CoT Model Results:")
    print(f"  Accuracy:    {result.accuracy:.4f}")
    print(f"  Answer Rate: {result.answer_rate:.4f}")
    print(f"{'='*50}\n")
    
    # Show some failed examples
    failed = [s for s in result.samples if not s.is_correct][:5]
    if failed:
        print(f"First {len(failed)} failed examples:")
        for i, sample in enumerate(failed, 1):
            print(f"\n{i}. Q: {sample.question}")
            print(f"   Expected: {sample.correct_answer}")
            print(f"   Got: {sample.answer}")
    
    return result


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test, "load": load})