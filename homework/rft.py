import json
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from .base_llm import BaseLLM
from .sft import tokenize, test_model


class RFTTorchDataset(TorchDataset):
    """
    Dataset wrapper for RFT json:
    Each entry is [question, answer, full_reasoning_with_answer].
    We treat reasoning+answer as one string.
    """

    def __init__(self, tokenizer, json_path: str):
        with open(json_path, "r") as f:
            raw = json.load(f)

        self.data = raw
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, true_ans, reasoning = self.data[idx]

        return tokenize(self.tokenizer, q, reasoning)


def load():
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / "rft_model"
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def train_model(
    data_path: str,
    output_dir: str,
    lr: float = 2e-4,
    batch_size: int = 8,
    epochs: int = 5,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    **kwargs,
):
    print(f"Loading RFT dataset: {data_path}")
    base = BaseLLM()

    train_dataset = RFTTorchDataset(base.tokenizer, data_path)
    print(f"Loaded {len(train_dataset)} RFT samples")

    print("Attaching LoRA modules…")
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    base.model = get_peft_model(base.model, lora_cfg)
    base.model.print_trainable_parameters()

    print("Preparing Trainer…")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        gradient_accumulation_steps=1,
        fp16=True,
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=base.model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=base.tokenizer,
    )

    print("Training RFT model…")
    trainer.train()

    print(f"Training complete. Saving to {output_dir}")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
