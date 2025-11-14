from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        This is intentionally left simple.
        CoTModel overrides this method.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Extract <answer>...</answer> float from model output.
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except Exception:
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        Non-batched generation for debugging.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        generated = outputs[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]: ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]: ...

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
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 50,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

        if num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = num_return_sequences

        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        input_len = input_ids.shape[1]
        generated_only = outputs[:, input_len:]

        decoded = self.tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True
        )

        if num_return_sequences and num_return_sequences > 1:
            grouped = []
            for i in range(0, len(decoded), num_return_sequences):
                grouped.append(decoded[i:i + num_return_sequences])
            return grouped

        return decoded

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    m = BaseLLM()
    for t in testset:
        print("input:", t)
        print("output:", m.generate(t))
    print(m.batched_generate(testset))


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})
