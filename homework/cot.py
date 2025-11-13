from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        super().__init__(checkpoint)

    def format_prompt(self, question: str) -> str:
        """
        High-quality chain-of-thought prompt optimized for SmolLM2-1.7B-Instruct.
        Includes:
        - strong system message
        - 3 diverse user–assistant examples (mass, length, time)
        - strict reasoning style
        - consistent <answer></answer> tag usage
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert unit-conversion assistant. "
                    "For every question, you ALWAYS:\n"
                    "1. Identify the units involved.\n"
                    "2. Write a short chain-of-thought showing the conversion formula.\n"
                    "3. Perform the arithmetic cleanly.\n"
                    "4. Provide the FINAL numeric answer inside <answer>number</answer>.\n\n"
                    "Rules:\n"
                    "- No text after the answer tag.\n"
                    "- Answer must be a pure number.\n"
                    "- Keep reasoning concise but explicit: show the formula then compute."
                )
            },

            # === EXAMPLE 1 (mass) ===
            {
                "role": "user",
                "content": "How many grams are in 3.5 kilograms?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 kg = 1000 g. So 3.5 kg = 3.5 × 1000 = 3500 g.\n"
                    "<answer>3500</answer>"
                )
            },

            # === EXAMPLE 2 (length) ===
            {
                "role": "user",
                "content": "How many centimeters are in 2.4 meters?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 m = 100 cm. So 2.4 m = 2.4 × 100 = 240 cm.\n"
                    "<answer>240</answer>"
                )
            },

            # === EXAMPLE 3 (time) ===
            {
                "role": "user",
                "content": "How many seconds are in 2.5 hours?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 hour = 3600 seconds. So 2.5 hours = 2.5 × 3600 = 9000.\n"
                    "<answer>9000</answer>"
                )
            },

            # === The real question ===
            {
                "role": "user",
                "content": question.strip()
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        return prompt


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
