from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-360M-Instruct"):
        super().__init__(checkpoint)

    def format_prompt(self, question: str) -> str:
        """
        Optimized few-shot CoT prompt that mirrors rft.json style exactly.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an accurate unit conversion assistant. "
                    "For every question, show the correct conversion rule, "
                    "perform the multiplication explicitly, and put ONLY the "
                    "final numeric answer inside <answer></answer>. "
                    "Do not include words or units inside <answer>."
                )
            },

            # Example 1 — matches RFT style
            {
                "role": "user",
                "content": "How many grams are in 4 kilograms?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilogram = 1000 grams. "
                    "Compute 4 × 1000 = 4000. "
                    "<answer>4000</answer>"
                )
            },

            # Example 2 — matches RFT structure (uses Multiply)
            {
                "role": "user",
                "content": "Convert 3 liters to cubic centimeters."
            },
            {
                "role": "assistant",
                "content": (
                    "1 liter = 1000 cubic centimeters. "
                    "Multiply 3 by 1000 = 3000. "
                    "<answer>3000</answer>"
                )
            },

            # Example 3 — includes decimal conversion like RFT
            {
                "role": "user",
                "content": "How many meters per second is 10 feet per second?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 foot = 0.3048 meters. "
                    "Compute 10 × 0.3048 = 3.048. "
                    "<answer>3.048</answer>"
                )
            },

            # Real task
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