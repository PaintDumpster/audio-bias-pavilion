#!/usr/bin/env python3
"""Example script demonstrating dataset builder usage."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from build_dataset import AudioDatasetBuilder


def example_basic_usage():
    """Basic example: build dataset from recordings."""
    print("=" * 60)
    print("Example 1: Basic Dataset Building")
    print("=" * 60)
    
    builder = AudioDatasetBuilder(data_dir="data/recordings")
    dataset = builder.create_dataset()
    
    print(f"\nDataset created with {len(dataset)} recordings")
    print(f"\nDataset columns: {dataset.column_names}")
    
    if len(dataset) > 0 and dataset[0]["audio"] is not None:
        print("\nFirst recording:")
        print(f"  District: {dataset[0]['district']}")
        print(f"  Day: {dataset[0]['day_of_week']}")
        print(f"  Time: {dataset[0]['time_of_day']}")
        print(f"  Typology: {dataset[0]['typology']}")
        print(f"  Neighborhood: {dataset[0]['neighborhood']}")
    else:
        print("\nNo recordings found yet. Add audio files to data/recordings/")


def example_metadata_extraction():
    """Example: demonstrate metadata extraction from filename."""
    print("\n" + "=" * 60)
    print("Example 2: Metadata Extraction")
    print("=" * 60)
    
    builder = AudioDatasetBuilder()
    
    # Test filenames
    test_filenames = [
        "traffic_lat41.3851_lon2.1734.m4a",
        "ambient_noise_recording.m4a",
        "construction@41.3901,2.1540.mp3",
        "voice_interview.wav",
    ]
    
    print("\nExtracting metadata from example filenames:\n")
    for filename in test_filenames:
        typology = builder.extract_typology_from_filename(filename)
        coords = builder.extract_coordinates_from_filename(filename)
        
        print(f"Filename: {filename}")
        print(f"  Typology: {typology}")
        print(f"  Coordinates: {coords}")
        print()


def example_save_and_load():
    """Example: save dataset to disk."""
    print("=" * 60)
    print("Example 3: Save Dataset")
    print("=" * 60)
    
    builder = AudioDatasetBuilder(data_dir="data/recordings")
    dataset = builder.save_dataset(output_dir="dataset_output")
    
    print("\nDataset saved to dataset_output/")
    print("\nTo load it later:")
    print("  from datasets import load_from_disk")
    print("  dataset = load_from_disk('dataset_output')")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Audio Bias Pavilion - Dataset Builder Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_metadata_extraction()
        example_save_and_load()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60 + "\n")
        
        print("Next steps:")
        print("1. Add audio recordings to data/recordings/<district>/<day>/<time>/")
        print("2. Run: python dataset_builder/build_dataset.py")
        print("3. Run: python dataset_builder/push_to_huggingface.py <username>/<dataset-name>")
        print()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
