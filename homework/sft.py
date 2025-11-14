# from .base_llm import BaseLLM
# from .data import Dataset, benchmark


# def load() -> BaseLLM:
#     from pathlib import Path

#     from peft import PeftModel

#     model_name = "sft_model"
#     model_path = Path(__file__).parent / model_name

#     llm = BaseLLM()
#     llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
#     llm.model.eval()

#     return llm


# def tokenize(tokenizer, question: str, answer: str):
#     """
#     Tokenize a data element.
#     We first append the <EOS> token to the question / answer pair.
#     Then we tokenize and construct the ground truth `labels`.
#     `labels[i] == -100` for the question or masked out parts, since we only want to supervise
#     the answer.
#     """
#     full_text = f"{question} {answer}{tokenizer.eos_token}"

#     tokenizer.padding_side = "right"
#     tokenizer.pad_token = tokenizer.eos_token
#     full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

#     input_ids = full["input_ids"]
#     question_len = len(tokenizer(question)["input_ids"])

#     labels = [-100] * question_len + input_ids[question_len:]

#     for i in range(len(labels)):
#         if full["attention_mask"][i] == 0:
#             labels[i] = -100

#     full["labels"] = labels
#     return full


# def format_example(prompt: str, answer: str | float) -> dict[str, str]:
#     """
#     Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
#     """
#     ans = f"<answer>{round(float(answer), 4)}</answer>"
#     return {"question": prompt, "answer": ans}


# class TokenizedDataset:
#     def __init__(self, tokenizer, data: Dataset, format_fn):
#         """
#         Use the
#         - BaseLLM.tokenizer
#         - Dataset
#         - format_fn which converts a data element into a dict with entries
#           - question: str
#           - answer: str
#         """
#         self.format_fn = format_fn
#         self.tokenizer = tokenizer
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         formated_data = self.format_fn(*self.data[idx])
#         return tokenize(self.tokenizer, **formated_data)


# def train_model(
#     output_dir: str,
#     **kwargs,
# ):
#     """
#     Fine-tune the model with LoRA adapters using HuggingFace Trainer.
#     """
#     from peft import LoraConfig, get_peft_model
#     from transformers import Trainer, TrainingArguments
#     from pathlib import Path

#     trainset = Dataset("train")
#     base = BaseLLM()
#     tokenizer = base.tokenizer

#     tokenized_dataset = TokenizedDataset(tokenizer, trainset, format_example)

#     config = LoraConfig(
#         r=8,                     
#         lora_alpha=32,          
#         target_modules="all-linear",
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

#     model = get_peft_model(base.model, config).to(base.device)
#     model.enable_input_require_grads()

#     args = TrainingArguments(
#         output_dir=output_dir,
#         logging_dir=output_dir,
#         report_to="tensorboard",
#         per_device_train_batch_size=32,
#         num_train_epochs=5,
#         gradient_checkpointing=True,
#         learning_rate=2e-4,
#         save_strategy="epoch",
#         remove_unused_columns=False,
#     )

#     trainer = Trainer(
#         model=model,
#         args=args,
#         train_dataset=tokenized_dataset,
#     )

#     trainer.train()

#     model.save_pretrained(Path(output_dir))

#     test_model(output_dir)


# def test_model(ckpt_path: str):
#     testset = Dataset("valid")
#     llm = BaseLLM()

#     from peft import PeftModel

#     llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

#     benchmark_result = benchmark(llm, testset, 100)
#     print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


# if __name__ == "__main__":
#     from fire import Fire

#     Fire({"train": train_model, "test": test_model, "load": load})


from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_path = Path(__file__).parent / "sft_model"

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


# -------------------------------------------------
# TOKENIZATION UTIL
# -------------------------------------------------
def tokenize(tokenizer, question: str, answer: str):
    """
    Safe, TA-compatible tokenizer:
    - Left question unlabelled (label = -100)
    - Supervise ONLY the <answer>...</answer>
    - No extra system or chat formatting
    """

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create strict “question + answer” sequence
    full_text = f"{question}{answer}{tokenizer.eos_token}"

    encoded = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    input_ids = encoded["input_ids"]

    # Get length of question (so we can mask it)
    q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    q_len = len(q_ids)

    # Mask question → label = -100
    labels = [-100] * q_len + input_ids[q_len:]

    # Mask padding as well
    for i, m in enumerate(encoded["attention_mask"]):
        if m == 0:
            labels[i] = -100

    encoded["labels"] = labels
    return encoded


# -------------------------------------------------
# FORMAT EXAMPLES FOR SFT
# -------------------------------------------------
def format_example(prompt: str, answer: str | float):
    """
    MUST match parse_answer() expectations.
    TA-safe simple format:
        <question>
        <answer>number</answer>
    """
    ans = f"<answer>{float(answer)}</answer>"
    return {"question": prompt, "answer": f"\n{ans}"}


# -------------------------------------------------
# DATASET WRAPPER
# -------------------------------------------------
class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.data = data
        self.tokenizer = tokenizer
        self.format_fn = format_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        formatted = self.format_fn(q, a)
        return tokenize(self.tokenizer, **formatted)


# -------------------------------------------------
# TRAINING LOGIC
# -------------------------------------------------
def train_model(output_dir: str, **kwargs):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from pathlib import Path

    # Load raw train dataset
    trainset = Dataset("train")

    # Base model + tokenizer
    base = BaseLLM()
    tokenizer = base.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenized dataset
    dataset = TokenizedDataset(tokenizer, trainset, format_example)

    # LoRA configuration (small + safe)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base.model, config).to(base.device)
    if base.device == "cuda":
        model.enable_input_require_grads()

    # TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        learning_rate=1e-4,
        warmup_steps=50,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="epoch",
        logging_steps=20,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(Path(output_dir))
    test_model(output_dir)


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
def test_model(ckpt_path: str):
    from peft import PeftModel

    testset = Dataset("valid")
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    result = benchmark(llm, testset, 100)
    print(f"{result.accuracy=}  {result.answer_rate=}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})