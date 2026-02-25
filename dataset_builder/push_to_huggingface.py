"""Push audio dataset to HuggingFace Hub."""

import os
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi, login
from build_dataset import AudioDatasetBuilder


class HuggingFaceUploader:
    """Handle uploading datasets to HuggingFace Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize uploader with HuggingFace token.
        
        Args:
            token: HuggingFace API token. If None, will try to use cached token.
        """
        self.token = token
        if token:
            login(token=token)
        self.api = HfApi()
    
    def push_dataset(
        self,
        dataset: Dataset,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None
    ):
        """Push dataset to HuggingFace Hub.
        
        Args:
            dataset: The dataset to push
            repo_id: Repository ID (username/dataset-name)
            private: Whether to make the dataset private
            commit_message: Optional commit message
        """
        if commit_message is None:
            commit_message = f"Update dataset with {len(dataset)} recordings"
        
        print(f"Pushing dataset to {repo_id}...")
        print(f"Dataset size: {len(dataset)} recordings")
        
        try:
            dataset.push_to_hub(
                repo_id=repo_id,
                token=self.token,
                private=private,
                commit_message=commit_message
            )
            print(f"✓ Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"✗ Failed to push dataset: {e}")
            raise
    
    def update_dataset(
        self,
        data_dir: str,
        repo_id: str,
        private: bool = False
    ):
        """Build and push updated dataset.
        
        Args:
            data_dir: Directory containing audio recordings
            repo_id: HuggingFace dataset repository ID
            private: Whether to make the dataset private
        """
        # Build dataset
        print("Building dataset from recordings...")
        builder = AudioDatasetBuilder(data_dir=data_dir)
        dataset = builder.create_dataset()
        
        # Push to hub
        self.push_dataset(
            dataset=dataset,
            repo_id=repo_id,
            private=private,
            commit_message=f"Update dataset with {len(dataset)} recordings"
        )


def main():
    """Main entry point for pushing dataset to HuggingFace."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Push audio dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "repo_id",
        help="HuggingFace repository ID (username/dataset-name)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/recordings",
        help="Directory containing audio recordings"
    )
    parser.add_argument(
        "--dataset-dir",
        help="Load existing dataset from disk instead of building from recordings"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (or use HF_TOKEN env variable)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Warning: No HuggingFace token provided.")
        print("Please provide via --token argument or HF_TOKEN environment variable")
        print("Or run 'huggingface-cli login' first")
    
    uploader = HuggingFaceUploader(token=token)
    
    if args.dataset_dir:
        # Load existing dataset
        print(f"Loading dataset from {args.dataset_dir}...")
        dataset = load_from_disk(args.dataset_dir)
        uploader.push_dataset(
            dataset=dataset,
            repo_id=args.repo_id,
            private=args.private
        )
    else:
        # Build and push dataset
        uploader.update_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            private=args.private
        )


if __name__ == "__main__":
    main()
