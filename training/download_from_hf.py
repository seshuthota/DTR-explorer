"""
Download files from a Hugging Face repo (model/dataset/space) to a local folder.
"""
import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["model", "dataset", "space"],
    )
    parser.add_argument("--local-dir", type=str, required=True)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="HF token. If omitted, uses HF_TOKEN environment variable.",
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
    os.makedirs(args.local_dir, exist_ok=True)

    print(
        f"Downloading {args.repo_type} repo {args.repo_id} "
        f"(revision={args.revision}) to {args.local_dir}"
    )
    out = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        token=token if token else None,
        allow_patterns=split_patterns(args.allow_patterns),
        ignore_patterns=split_patterns(args.ignore_patterns),
    )
    print(f"Download complete: {out}")


if __name__ == "__main__":
    main()
