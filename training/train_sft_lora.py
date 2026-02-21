"""
QLoRA SFT Trainer for DTR-Filtered Dataset
==========================================
Trains LoRA adapters on JSONL rows with fields: prompt, completion.
"""
import os
import argparse
import random
import inspect
import numpy as np
import torch
from datasets import load_dataset

# Keep training pipeline PyTorch-only; avoids tf/keras import issues.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_all_linear_names(model):
    # Typical QLoRA strategy: target every linear layer except output head.
    linear_names = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "linear" in class_name:
            linear_names.add(name.split(".")[-1])
    linear_names.discard("lm_head")
    return sorted(linear_names)


def format_example(tokenizer, prompt, completion, system_prompt):
    # Keep format aligned with generation scripts.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL")
    parser.add_argument("--dataset-path", type=str, default="data/dtr_filtered_sft.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/dtr-tuned-1.2b-v1")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-use-4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = find_all_linear_names(model)
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def map_text(example):
        text = format_example(
            tokenizer,
            prompt=example["prompt"],
            completion=example["completion"],
            system_prompt=args.system_prompt,
        )
        return {"text": text}

    dataset = dataset.map(map_text, remove_columns=dataset.column_names)

    def map_tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(map_tokenize, remove_columns=dataset.column_names)

    if 0.0 < args.eval_ratio < 0.5 and len(dataset) > 50:
        split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = dataset
        eval_ds = None

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.micro_batch_size,
        "per_device_eval_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "bf16": bf16,
        "fp16": not bf16,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "seed": args.seed,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    eval_strategy_value = "steps" if eval_ds is not None else "no"
    if "evaluation_strategy" in sig.parameters:
        training_kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in sig.parameters:
        training_kwargs["eval_strategy"] = eval_strategy_value

    if eval_ds is not None and "eval_steps" in sig.parameters:
        training_kwargs["eval_steps"] = args.save_steps

    training_args = TrainingArguments(**training_kwargs)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        # Newer Transformers moved tokenizer/processor wiring to `processing_class`.
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    print("Saving adapter + tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
