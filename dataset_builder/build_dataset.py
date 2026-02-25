"""Build HuggingFace dataset from audio recordings with metadata extraction.

This dataset builder is configured for YAMNet model training:
- Audio is automatically resampled to 16kHz (YAMNet's expected rate)
- Converted to mono channel
- Normalized to exactly 30 seconds (trimmed or zero-padded)
- Decoded from compressed formats (.m4a, .mp3, etc.)

The HuggingFace datasets library handles all preprocessing automatically.
"""

import os
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from datasets import Dataset, Audio, Features, Value
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


# Constants from foldergenerator.py
DAYS_OF_WEEK = [
    "monday", "tuesday", "wednesday", "thursday", 
    "friday", "saturday", "sunday"
]

DAY_PERIODS = ["morning", "midday", "evening", "night"]

# Supported audio formats
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg"}

# YAMNet audio configuration
YAMNET_SAMPLE_RATE = 16000  # Hz
YAMNET_CHANNELS = 1  # Mono
YAMNET_DURATION = 30.0  # seconds - standardized duration for training


class AudioDatasetBuilder:
    """Build and manage HuggingFace dataset from audio recordings.
    
    Audio is preprocessed for YAMNet compatibility:
    - 16kHz sampling rate
    - Mono channel
    - Exactly 30 seconds (trimmed or zero-padded)
    - Automatic decoding and resampling
    """
    
    def __init__(self, data_dir: str = "data/recordings", target_duration: float = YAMNET_DURATION):
        """Initialize the dataset builder.
        
        Args:
            data_dir: Base directory containing the recordings
            target_duration: Target duration in seconds (default: 30.0 for YAMNet)
        """
        self.data_dir = Path(data_dir)
        self.target_duration = target_duration
        self.target_samples = int(YAMNET_SAMPLE_RATE * target_duration)
        self.geolocator = Nominatim(user_agent="audio-bias-pavilion")
        self._geocode_cache: Dict[Tuple[float, float], str] = {}
    
    @staticmethod
    def get_yamnet_info() -> Dict[str, any]:
        """Get YAMNet audio configuration information.
        
        Returns:
            Dictionary with YAMNet requirements
        """
        return {
            "sampling_rate": YAMNET_SAMPLE_RATE,
            "channels": YAMNET_CHANNELS,
            "channel_format": "mono",
            "duration": f"{YAMNET_DURATION} seconds (exactly)",
            "target_samples": int(YAMNET_SAMPLE_RATE * YAMNET_DURATION),
            "preprocessing": "automatic normalization + HuggingFace resampling",
            "notes": [
                "Audio files are automatically decoded from .m4a/.mp3/etc",
                "Resampling to 16kHz happens during dataset loading",
                "Stereo files are automatically converted to mono",
                f"All clips normalized to exactly {YAMNET_DURATION} seconds",
                "Longer clips are trimmed, shorter clips are zero-padded"
            ]
        }
    
    def normalize_audio_duration(self, audio_path: Path) -> Optional[Path]:
        """Normalize audio file to exact target duration.
        
        Creates a normalized version of the audio file:
        - If longer than target: trim to target duration
        - If shorter than target: pad with zeros
        - Always at 16kHz mono
        
        Args:
            audio_path: Path to the original audio file
            
        Returns:
            Path to normalized audio file, or None if processing fails
        """
        try:
            # Load audio file (librosa will handle .m4a and other formats)
            # Use soundfile which works well with various formats via libsndfile
            import librosa
            
            # Load and resample to 16kHz mono in one step
            audio, sr = librosa.load(
                str(audio_path),
                sr=YAMNET_SAMPLE_RATE,
                mono=True
            )
            
            current_samples = len(audio)
            
            if current_samples > self.target_samples:
                # Trim to exact duration
                audio = audio[:self.target_samples]
            elif current_samples < self.target_samples:
                # Pad with zeros to exact duration
                padding = self.target_samples - current_samples
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Create normalized file in same directory with _normalized suffix
            normalized_path = audio_path.parent / f"{audio_path.stem}_normalized.wav"
            
            # Save as WAV (standard format, lossless)
            sf.write(
                normalized_path,
                audio,
                YAMNET_SAMPLE_RATE,
                subtype='PCM_16'
            )
            
            return normalized_path
            
        except Exception as e:
            print(f"Warning: Failed to normalize {audio_path.name}: {e}")
            return None
        
    def extract_typology_from_filename(self, filename: str) -> Optional[str]:
        """Extract typology from filename using regex patterns.
        
        Expected patterns:
        - typology_<type>_
        - <type>_coords_
        - pattern variations
        
        Args:
            filename: The audio filename
            
        Returns:
            The extracted typology or None
        """
        # Pattern 1: typology explicitly mentioned
        # Example: survey_traffic_lat41.34_lon2.17.m4a
        patterns = [
            r'(?:^|_)(traffic|ambient|voice|music|construction|nature|urban|quiet)(?:_|\.)',
            r'typology[_-](\w+)',
            r'^(\w+?)_(?:lat|lon|coords)',
        ]
        
        filename_lower = filename.lower()
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return match.group(1)
        
        # Default: try to extract first meaningful word
        match = re.match(r'^([a-zA-Z]+)', filename)
        if match and len(match.group(1)) > 2:
            return match.group(1).lower()
            
        return "unknown"
    
    def extract_coordinates_from_filename(self, filename: str) -> Optional[Tuple[float, float]]:
        """Extract latitude and longitude from filename.
        
        Expected patterns:
        - lat41.34_lon2.17
        - 41.34N_2.17E
        - coords_41.34_2.17
        
        Args:
            filename: The audio filename
            
        Returns:
            Tuple of (latitude, longitude) or None
        """
        patterns = [
            r'lat(-?\d+\.?\d*)_lon(-?\d+\.?\d*)',
            r'(-?\d+\.?\d*)N_(-?\d+\.?\d*)E',
            r'coords_(-?\d+\.?\d*)_(-?\d+\.?\d*)',
            r'@(-?\d+\.?\d*),(-?\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    # Validate coordinates (rough check for Barcelona area)
                    if 41.0 <= lat <= 42.0 and 1.5 <= lon <= 2.5:
                        return (lat, lon)
                except ValueError:
                    continue
        
        return None
    
    def get_neighborhood_from_coords(
        self, 
        lat: float, 
        lon: float, 
        max_retries: int = 3
    ) -> str:
        """Get neighborhood name from coordinates using Nominatim.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_retries: Maximum number of retry attempts
            
        Returns:
            Neighborhood name or "unknown"
        """
        coords = (lat, lon)
        
        # Check cache first
        if coords in self._geocode_cache:
            return self._geocode_cache[coords]
        
        # Try geocoding with retries
        for attempt in range(max_retries):
            try:
                # Nominatim requires delay between requests
                if attempt > 0:
                    time.sleep(1)
                
                location = self.geolocator.reverse(
                    f"{lat}, {lon}",
                    language="en",
                    zoom=16  # Neighborhood level
                )
                
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    # Try to get neighborhood, suburb, or district
                    neighborhood = (
                        address.get('neighbourhood') or
                        address.get('suburb') or
                        address.get('quarter') or
                        address.get('district') or
                        address.get('city_district') or
                        "unknown"
                    )
                    self._geocode_cache[coords] = neighborhood
                    return neighborhood
                    
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt == max_retries - 1:
                    print(f"Warning: Geocoding failed for {lat},{lon}: {e}")
                continue
        
        # Cache the failure too
        self._geocode_cache[coords] = "unknown"
        return "unknown"
    
    def scan_recordings(self) -> List[Dict]:
        """Scan the recordings directory and extract metadata.
        
        Returns:
            List of dictionaries with recording metadata
        """
        recordings = []
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return recordings
        
        # Walk through district/day/time_of_day structure
        for district_dir in self.data_dir.iterdir():
            if not district_dir.is_dir():
                continue
            
            district = district_dir.name
            
            for day_dir in district_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                
                day = day_dir.name
                if day.lower() not in DAYS_OF_WEEK:
                    continue
                
                for period_dir in day_dir.iterdir():
                    if not period_dir.is_dir():
                        continue
                    
                    time_of_day = period_dir.name
                    if time_of_day.lower() not in DAY_PERIODS:
                        continue
                    
                    # Scan for audio files
                    for audio_file in period_dir.iterdir():
                        if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                            continue
                        
                        # Skip already normalized files
                        if "_normalized" in audio_file.stem:
                            continue
                        
                        # Normalize audio duration to exactly target duration
                        print(f"Processing: {audio_file.name}")
                        normalized_file = self.normalize_audio_duration(audio_file)
                        
                        if normalized_file is None:
                            print(f"Skipping {audio_file.name} due to processing error")
                            continue
                        
                        # Extract metadata from original filename
                        filename = audio_file.name
                        typology = self.extract_typology_from_filename(filename)
                        coords = self.extract_coordinates_from_filename(filename)
                        
                        neighborhood = "unknown"
                        if coords:
                            lat, lon = coords
                            neighborhood = self.get_neighborhood_from_coords(lat, lon)
                        
                        # Get file modification date as recording date
                        recording_date = datetime.fromtimestamp(
                            audio_file.stat().st_mtime
                        ).strftime("%Y-%m-%d")
                        
                        recordings.append({
                            "audio": str(normalized_file),  # Use normalized file
                            "date": recording_date,
                            "time_of_day": time_of_day,
                            "district": district,
                            "neighborhood": neighborhood,
                            "typology": typology,
                            "day_of_week": day,
                            "filename": filename,  # Keep original filename for reference
                            "latitude": coords[0] if coords else None,
                            "longitude": coords[1] if coords else None,
                        })
        
        return recordings
    
    def create_dataset(self) -> Dataset:
        """Create HuggingFace dataset from recordings.
        
        Returns:
            HuggingFace Dataset object
        """
        print("Scanning recordings...")
        recordings = self.scan_recordings()
        
        if not recordings:
            print("Warning: No recordings found. Creating empty dataset.")
            # Create empty dataset with correct schema
            recordings = [{
                "audio": None,
                "date": None,
                "time_of_day": None,
                "district": None,
                "neighborhood": None,
                "typology": None,
                "day_of_week": None,
                "filename": None,
                "latitude": None,
                "longitude": None,
            }]
        
        print(f"Found {len(recordings)} recordings")
        
        # Define features schema
        # Audio is configured for YAMNet compatibility:
        # - 16kHz sampling rate (YAMNet's expected input)
        # - Mono channel
        # - Exactly 30 seconds (normalized during scanning)
        # - WAV format (from normalized files)
        features = Features({
            "audio": Audio(sampling_rate=YAMNET_SAMPLE_RATE, mono=True),
            "date": Value("string"),
            "time_of_day": Value("string"),
            "district": Value("string"),
            "neighborhood": Value("string"),
            "typology": Value("string"),
            "day_of_week": Value("string"),
            "filename": Value("string"),
            "latitude": Value("float32"),
            "longitude": Value("float32"),
        })
        
        # Create dataset
        dataset = Dataset.from_dict(
            {k: [r[k] for r in recordings] for k in recordings[0].keys()},
            features=features
        )
        
        return dataset
    
    def save_dataset(self, output_dir: str = "dataset_output"):
        """Save dataset to disk.
        
        Args:
            output_dir: Directory to save the dataset
        """
        dataset = self.create_dataset()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving dataset to {output_path}...")
        dataset.save_to_disk(str(output_path))
        print(f"Dataset saved successfully with {len(dataset)} recordings")
        
        return dataset


def main():
    """Main entry point for building the dataset."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Build HuggingFace dataset from audio recordings (YAMNet-compatible)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/recordings",
        help="Base directory containing recordings"
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_output",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display YAMNet audio configuration and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        info = AudioDatasetBuilder.get_yamnet_info()
        print("YAMNet Audio Configuration:")
        print("=" * 50)
        print(f"Sampling Rate: {info['sampling_rate']} Hz")
        print(f"Channels: {info['channels']} ({info['channel_format']})")
        print(f"Duration: {info['duration']}")
        print(f"Target Samples: {info['target_samples']}")
        print(f"Preprocessing: {info['preprocessing']}")
        print("\nNotes:")
        for note in info['notes']:
            print(f"  • {note}")
        print("=" * 50)
        return
    
    builder = AudioDatasetBuilder(data_dir=args.data_dir)
    builder.save_dataset(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
