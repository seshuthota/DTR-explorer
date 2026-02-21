"""
Upload a trained artifact directory (LoRA adapter or full model) to Hugging Face Hub.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local folder to upload (for example models/dtr-tuned-1.2b-v1).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination repo id, e.g. username/dtr-tuned-1.2b-v1-lora",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="HF token. If omitted, uses HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch/revision to upload to.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload trained artifact",
    )
    parser.add_argument(
        "--allow-patterns",
        type=str,
        default="",
        help="Comma-separated glob patterns to include.",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        default="",
        help="Comma-separated glob patterns to exclude.",
    )
    return parser.parse_args()


def split_patterns(raw: str):
    if not raw.strip():
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "No token provided. Set HF_TOKEN or pass --token explicitly."
        )

    local_dir = Path(args.local_dir)
    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError(f"No files found to upload in: {local_dir}")

    api = HfApi(token=token)
    print(f"Ensuring repo exists: {args.repo_id} ({args.repo_type})")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {len(files)} files from {local_dir} ...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=split_patterns(args.allow_patterns),
        ignore_patterns=split_patterns(args.ignore_patterns),
    )
    print(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
