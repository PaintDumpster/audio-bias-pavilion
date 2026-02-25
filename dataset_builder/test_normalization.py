#!/usr/bin/env python3
"""Test audio normalization to verify 30-second duration."""

import numpy as np
import soundfile as sf
from pathlib import Path


def create_test_audio(duration: float, output_path: str):
    """Create a test audio file with a specific duration.
    
    Args:
        duration: Duration in seconds
        output_path: Where to save the test file
    """
    sample_rate = 44100  # Original sample rate (will be resampled to 16kHz)
    samples = int(duration * sample_rate)
    
    # Generate a simple sine wave
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples, False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .wav file
    sf.write(output_path, audio, sample_rate)
    print(f"Created test audio: {output_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Samples: {samples}")


def verify_normalized_audio(audio_path: str, expected_duration: float = 30.0, expected_sr: int = 16000):
    """Verify that normalized audio matches expected specs.
    
    Args:
        audio_path: Path to the normalized audio file
        expected_duration: Expected duration in seconds
        expected_sr: Expected sample rate in Hz
        
    Returns:
        True if audio meets specs, False otherwise
    """
    try:
        audio, sr = sf.read(audio_path)
        actual_duration = len(audio) / sr
        expected_samples = int(expected_duration * expected_sr)
        
        print(f"\nVerifying: {Path(audio_path).name}")
        print(f"  Sample rate: {sr} Hz (expected: {expected_sr} Hz)")
        print(f"  Samples: {len(audio)} (expected: {expected_samples})")
        print(f"  Duration: {actual_duration:.2f}s (expected: {expected_duration:.2f}s)")
        print(f"  Shape: {audio.shape}")
        
        # Check specs
        sr_ok = sr == expected_sr
        duration_ok = abs(actual_duration - expected_duration) < 0.01  # Within 10ms
        samples_ok = len(audio) == expected_samples
        mono_ok = audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1)
        
        print(f"\n  ✓ Sample rate: {'PASS' if sr_ok else 'FAIL'}")
        print(f"  ✓ Duration: {'PASS' if duration_ok else 'FAIL'}")
        print(f"  ✓ Sample count: {'PASS' if samples_ok else 'FAIL'}")
        print(f"  ✓ Mono: {'PASS' if mono_ok else 'FAIL'}")
        
        all_ok = sr_ok and duration_ok and samples_ok and mono_ok
        
        if all_ok:
            print(f"\n✓ Audio meets YAMNet specs!")
        else:
            print(f"\n✗ Audio does not meet YAMNet specs")
        
        return all_ok
        
    except Exception as e:
        print(f"Error verifying audio: {e}")
        return False


def main():
    """Create test files and demonstrate normalization."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Test audio normalization for YAMNet training"
    )
    parser.add_argument(
        "--create-tests",
        action="store_true",
        help="Create test audio files of various durations"
    )
    parser.add_argument(
        "--verify",
        help="Verify a normalized audio file meets YAMNet specs"
    )
    
    args = parser.parse_args()
    
    if args.create_tests:
        print("Creating test audio files...")
        print("=" * 60)
        
        test_dir = Path("test_audio")
        test_dir.mkdir(exist_ok=True)
        
        # Create test files of different durations
        test_cases = [
            (25.0, "short_25s.wav"),    # Shorter than 30s
            (30.0, "exact_30s.wav"),     # Exactly 30s
            (35.0, "long_35s.wav"),      # Longer than 30s
        ]
        
        for duration, filename in test_cases:
            create_test_audio(duration, str(test_dir / filename))
        
        print("\n" + "=" * 60)
        print("Test files created in 'test_audio/' directory")
        print("\nNext steps:")
        print("1. Run the dataset builder on test files:")
        print("   python dataset_builder/build_dataset.py --data-dir test_audio")
        print("\n2. Verify normalized outputs:")
        print("   python dataset_builder/test_normalization.py --verify test_audio/*_normalized.wav")
        
    elif args.verify:
        print("Verifying normalized audio...")
        print("=" * 60)
        verify_normalized_audio(args.verify)
        
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Create test files")
        print("  python dataset_builder/test_normalization.py --create-tests")
        print("\n  # Verify normalized file")
        print("  python dataset_builder/test_normalization.py --verify path/to/audio_normalized.wav")


if __name__ == "__main__":
    main()
