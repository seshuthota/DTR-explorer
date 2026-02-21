"""
Merge a LoRA adapter into its base model and save merged weights locally.
"""
import argparse
import os

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-id",
        type=str,
        required=True,
        help="HF repo id or local path for the LoRA adapter.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="",
        help="Optional base model override. If empty, inferred from adapter_config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save merged model.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    args = parser.parse_args()

    peft_cfg = PeftConfig.from_pretrained(args.adapter_id)
    base_model_id = args.base_model or peft_cfg.base_model_name_or_path

    print(f"Base model: {base_model_id}")
    print(f"Adapter: {args.adapter_id}")
    print(f"Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=args.trust_remote_code,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_id)
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Merged model saved successfully.")


if __name__ == "__main__":
    main()
