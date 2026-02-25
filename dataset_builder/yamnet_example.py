"""Example: Using Audio Bias Pavilion dataset with YAMNet model."""

import numpy as np
from datasets import load_dataset, load_from_disk


def load_audio_dataset(source: str, is_hub: bool = False):
    """Load the audio dataset.
    
    Args:
        source: HuggingFace repo ID or local path
        is_hub: True if loading from HuggingFace Hub, False for local
    
    Returns:
        Loaded dataset
    """
    if is_hub:
        print(f"Loading dataset from HuggingFace Hub: {source}")
        dataset = load_dataset(source)
    else:
        print(f"Loading dataset from disk: {source}")
        dataset = load_from_disk(source)
    
    return dataset


def inspect_audio_sample(dataset, index: int = 0):
    """Inspect a single audio sample from the dataset.
    
    Args:
        dataset: The dataset to inspect
        index: Sample index to inspect
    """
    sample = dataset[index]
    
    print("\n" + "=" * 60)
    print("Sample Metadata:")
    print("=" * 60)
    print(f"Filename: {sample['filename']}")
    print(f"District: {sample['district']}")
    print(f"Neighborhood: {sample['neighborhood']}")
    print(f"Typology: {sample['typology']}")
    print(f"Day: {sample['day_of_week']}")
    print(f"Time: {sample['time_of_day']}")
    print(f"Date: {sample['date']}")
    
    if sample['latitude'] and sample['longitude']:
        print(f"Location: ({sample['latitude']:.4f}, {sample['longitude']:.4f})")
    
    print("\n" + "=" * 60)
    print("Audio Data (YAMNet-compatible):")
    print("=" * 60)
    
    audio = sample['audio']
    waveform = audio['array']
    sample_rate = audio['sampling_rate']
    duration = len(waveform) / sample_rate
    expected_samples = 16000 * 30  # 480,000 samples for 30 seconds at 16kHz
    
    print(f"Sampling Rate: {sample_rate} Hz ✓ (YAMNet expects 16000 Hz)")
    print(f"Channels: Mono ✓ (YAMNet expects mono)")
    print(f"Waveform Shape: {waveform.shape}")
    print(f"Sample Count: {len(waveform)} (expected: {expected_samples})")
    print(f"Duration: {duration:.6f} seconds (expected: 30.0 seconds)")
    print(f"Data Type: {waveform.dtype}")
    print(f"Value Range: [{waveform.min():.4f}, {waveform.max():.4f}]")
    
    # Check if audio meets YAMNet requirements
    is_correct_sr = sample_rate == 16000
    is_mono = waveform.ndim == 1
    is_30_seconds = len(waveform) == expected_samples
    is_valid = is_correct_sr and is_mono and is_30_seconds
    
    print(f"\nValidation:")
    print(f"  ✓ Sampling rate: {'PASS' if is_correct_sr else 'FAIL'}")
    print(f"  ✓ Mono channel: {'PASS' if is_mono else 'FAIL'}")
    print(f"  ✓ Exactly 30 seconds: {'PASS' if is_30_seconds else 'FAIL'}")
    print(f"\n{'✓' if is_valid else '✗'} Audio is YAMNet-compatible: {is_valid}")
    
    return sample


def prepare_for_yamnet(dataset):
    """Prepare dataset for YAMNet training.
    
    Args:
        dataset: The audio dataset
        
    Returns:
        Processed dataset
    """
    print("\nPreparing dataset for YAMNet...")
    
    def extract_waveform(batch):
        """Extract waveform arrays for model input."""
        return {
            'waveform': [audio['array'] for audio in batch['audio']],
            'label': batch['typology']
        }
    
    # The audio is already at 16kHz mono - just extract arrays
    processed = dataset.map(
        extract_waveform,
        batched=True,
        desc="Extracting waveforms"
    )
    
    print(f"✓ Processed {len(processed)} samples")
    return processed


def get_dataset_statistics(dataset):
    """Get statistics about the dataset.
    
    Args:
        dataset: The audio dataset
    """
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Total Samples: {len(dataset)}")
    
    # Get unique values for categorical features
    typologies = set(dataset['typology'])
    districts = set(dataset['district'])
    times = set(dataset['time_of_day'])
    days = set(dataset['day_of_week'])
    
    print(f"\nTypologies ({len(typologies)}): {', '.join(sorted(typologies))}")
    print(f"Districts ({len(districts)}): {', '.join(sorted(districts))}")
    print(f"Times of Day ({len(times)}): {', '.join(sorted(times))}")
    print(f"Days of Week ({len(days)}): {', '.join(sorted(days))}")
    
    # Count by typology
    print("\nSamples by Typology:")
    from collections import Counter
    typology_counts = Counter(dataset['typology'])
    for typ, count in typology_counts.most_common():
        print(f"  {typ}: {count}")


def main():
    """Main example function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Example: Inspect audio dataset for YAMNet training"
    )
    parser.add_argument(
        "--dataset",
        default="dataset_output",
        help="Dataset path or HuggingFace repo ID"
    )
    parser.add_argument(
        "--hub",
        action="store_true",
        help="Load from HuggingFace Hub instead of local disk"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of sample to inspect"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show dataset statistics"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    try:
        dataset = load_audio_dataset(args.dataset, is_hub=args.hub)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTip: Build the dataset first with:")
        print("  python dataset_builder/build_dataset.py")
        return
    
    # Show statistics
    get_dataset_statistics(dataset)
    
    # Inspect a sample (unless stats-only)
    if not args.stats_only and len(dataset) > 0:
        inspect_audio_sample(dataset, args.sample_index)
    elif len(dataset) == 0:
        print("\n⚠ Dataset is empty. Add audio recordings and rebuild.")
    
    print("\n" + "=" * 60)
    print("Next Steps for YAMNet Training:")
    print("=" * 60)
    print("1. The audio is already preprocessed (16kHz, mono)")
    print("2. Load YAMNet model from TensorFlow Hub")
    print("3. Extract waveforms: sample['audio']['array']")
    print("4. Pass directly to YAMNet model")
    print("\nSee YAMNET_TRAINING.md for full training examples")


if __name__ == "__main__":
    main()
