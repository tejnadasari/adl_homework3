from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        super().__init__(checkpoint)

    def format_prompt(self, question: str) -> str:
        """
        Convert the question into a chat-style prompt with:
        - clear system instruction
        - 3 high-quality in-context examples
        - the user question
        - proper chat template usage
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise unit conversion assistant. "
                    "For every question: (1) show short step-by-step reasoning, "
                    "(2) ALWAYS end with the final number inside <answer>number</answer>. "
                    "Never add text after the answer tag."
                ),
            },

            # Example 1
            {
                "role": "user",
                "content": "How many grams are in 2 kg?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 kg = 1000 grams.\n"
                    "2 kg = 2 × 1000 = <answer>2000</answer>"
                ),
            },

            # Example 2
            {
                "role": "user",
                "content": "How many meters are in 3 km?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 km = 1000 meters.\n"
                    "3 km = 3 × 1000 = <answer>3000</answer>"
                ),
            },

            # Example 3
            {
                "role": "user",
                "content": "How many centimeters are in 4 meters?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 meter = 100 centimeters.\n"
                    "4 meters = 4 × 100 = <answer>400</answer>"
                ),
            },

            # Actual question
            {
                "role": "user",
                "content": question.strip()
            },
        ]

        # Convert to model-specific prompt format
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
