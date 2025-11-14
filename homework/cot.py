from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct"):
        super().__init__(checkpoint)

    def format_prompt(self, question: str) -> str:
        """
        Highly structured CoT format optimized for 360M models.
        - Explicit step-by-step reasoning
        - Explicit unit equivalence statement
        - Explicit multiply operation
        - Always ends with <answer>...</answer>
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful math assistant. "
                    "Solve unit conversion questions using the correct conversion factor. "
                    "Show the steps: state the unit relationship, multiply, and output the final number "
                    "INSIDE <answer></answer>. Only produce one answer."
                )
            },

            # Example 1
            {
                "role": "user",
                "content": "How many grams are in 4 kilograms?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilogram = 1000 grams. "
                    "Compute: 4 × 1000 = 4000. "
                    "<answer>4000.0</answer>"
                )
            },

            # Example 2
            {
                "role": "user",
                "content": "How many centimeters are in 3.2 meters?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 meter = 100 centimeters. "
                    "Compute: 3.2 × 100 = 320. "
                    "<answer>320.0</answer>"
                )
            },

            # Example 3
            {
                "role": "user",
                "content": "How many milligrams are in 1.75 grams?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 gram = 1000 milligrams. "
                    "Compute: 1.75 × 1000 = 1750. "
                    "<answer>1750.0</answer>"
                )
            },

            # Final user question
            {
                "role": "user",
                "content": question
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark
    testset = Dataset("valid")
    model = CoTModel()
    result = benchmark(model, testset, 100)
    print(f"{result.accuracy=}  {result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model, "load": load})