#!/usr/bin/env python3
import argparse
import os
from huggingface_hub import snapshot_download

def main():
    p = argparse.ArgumentParser(description="Download a Hugging Face model snapshot for offline use.")
    p.add_argument("--source", required=True, help="Model repo, e.g. bert-base-uncased or org/model")
    p.add_argument("--target", required=True, help="Target folder to store files")
    args = p.parse_args()

    path = snapshot_download(
        repo_id=args.source,
        local_dir=args.target
    )
    print(f"Downloaded snapshot to: {path}")

if __name__ == "__main__":
    main()