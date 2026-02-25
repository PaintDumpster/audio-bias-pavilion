#!/usr/bin/env python3
"""Convenience script to build and push dataset in one command."""

import argparse
import sys
from pathlib import Path

# Add dataset_builder to path
sys.path.insert(0, str(Path(__file__).parent))

from build_dataset import AudioDatasetBuilder
from push_to_huggingface import HuggingFaceUploader


def main():
    parser = argparse.ArgumentParser(
        description="Build and push audio dataset to HuggingFace in one step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build and push dataset
  python workflow.py myusername/barcelona-audio --token YOUR_TOKEN
  
  # Update existing dataset
  python workflow.py myusername/barcelona-audio --update
  
  # Build dataset but don't push (dry run)
  python workflow.py --no-push --data-dir data/recordings
        """
    )
    
    parser.add_argument(
        "repo_id",
        nargs="?",
        help='HuggingFace repository ID (e.g., "username/dataset-name")'
    )
    parser.add_argument(
        "--data-dir",
        default="data/recordings",
        help="Directory containing audio recordings (default: data/recordings)"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_output",
        help="Where to save the dataset locally (default: dataset_output)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Only build dataset locally, don't push to HuggingFace"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing dataset (automatically set with repeated runs)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.no_push and not args.repo_id:
        parser.error("repo_id is required unless --no-push is specified")
    
    print("=" * 60)
    print("Audio Bias Pavilion - Dataset Workflow")
    print("=" * 60)
    
    # Step 1: Build dataset
    print("\n[Step 1/2] Building dataset...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    
    builder = AudioDatasetBuilder(data_dir=args.data_dir)
    dataset = builder.save_dataset(output_dir=args.output_dir)
    
    print(f"\n✓ Dataset built with {len(dataset)} recordings")
    
    # Step 2: Push to HuggingFace (if requested)
    if not args.no_push:
        print("\n[Step 2/2] Pushing to HuggingFace...")
        print(f"  Repository: {args.repo_id}")
        print(f"  Private: {args.private}")
        
        uploader = HuggingFaceUploader(token=args.token)
        
        commit_msg = "Update dataset" if args.update else "Initial dataset upload"
        commit_msg += f" with {len(dataset)} recordings"
        
        uploader.push_dataset(
            dataset=dataset,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=commit_msg
        )
        
        print("\n" + "=" * 60)
        print("✓ Workflow completed successfully!")
        print(f"View your dataset at: https://huggingface.co/datasets/{args.repo_id}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✓ Dataset built successfully (not pushed to HuggingFace)")
        print(f"Dataset saved at: {args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
