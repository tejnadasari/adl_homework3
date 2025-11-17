from typing import overload
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "HuggingFaceTB/SmolLM2-360M-Instruct"


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    def __init__(self, checkpoint=CHECKPOINT):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)

    def format_prompt(self, question: str) -> str:
        """Simple prompt. CoTModel overrides this."""
        return question

    def parse_answer(self, text: str) -> float:
        """Extract <answer>...</answer>."""
        try:
            return float(text.split("<answer>")[1].split("</answer>")[0])
        except:
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode
        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        ...

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
    ) -> list[str] | list[list[str]]:
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 50,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.02,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False

        if num_return_sequences is not None:
            gen_kwargs["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        prompt_len = input_ids.size(1)
        generated_tokens = outputs[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        if num_return_sequences and num_return_sequences > 1:
            grouped = []
            for i in range(0, len(decoded), num_return_sequences):
                grouped.append(decoded[i : i + num_return_sequences])
            return grouped

        return decoded

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        gens = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in gens]


def test_model():
    print("testing generate function")
    m = BaseLLM()
    tests = ["The cat went up", "The dog went down"]
    for t in tests:
        print("input:", t)
        print("output:", m.generate(t))
    print(m.batched_generate(tests))


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})