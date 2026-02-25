# Quick Start Guide

## YAMNet Compatibility ✓

This dataset is **pre-configured for YAMNet training**:
- Audio automatically resampled to 16kHz
- Converted to mono
- **Normalized to exactly 30 seconds** (trimmed or padded)
- No manual preprocessing needed!

See [YAMNET_TRAINING.md](YAMNET_TRAINING.md) for detailed YAMNet usage.

## Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

## Initial Setup

```bash
# Create folder structure
python dataset_builder/foldergenerator.py
```

This creates: `data/recordings/<district>/<day>/<time_of_day>/`

## Recording Your Audio

Your **~30-second .m4a clips are perfect** for YAMNet:
- Recordings around 30 seconds are ideal
- Longer clips will be automatically trimmed to 30s
- Shorter clips will be zero-padded to 30s
- All clips become exactly 30 seconds during build
- No conversion needed!

## Recommended Filename Format

Name your audio files with coordinates and typology for best metadata extraction:

```
typology_lat<latitude>_lon<longitude>.m4a
```

**Example**: `traffic_lat41.3851_lon2.1734.m4a`

**Supported typologies**: traffic, ambient, voice, music, construction, nature, urban, quiet

## Building and Uploading Dataset

### Option 1: One-Step Workflow (Recommended)

```bash
# Build and push in one command
python dataset_builder/workflow.py YOUR_USERNAME/barcelona-audio --token YOUR_HF_TOKEN
```

### Option 2: Separate Steps

```bash
# Step 1: Build dataset locally
python dataset_builder/build_dataset.py

# Step 2: Push to HuggingFace
python dataset_builder/push_to_huggingface.py YOUR_USERNAME/barcelona-audio
```

## Updating with New Recordings

When you add more recordings:

```bash
# Quick update
python dataset_builder/workflow.py YOUR_USERNAME/barcelona-audio --update
```

## Testing with Empty Data

Even with no recordings yet, you can test the workflow:

```bash
# Build empty dataset (for testing)
python dataset_builder/build_dataset.py

# The script will create an empty dataset with the correct schema
```

## Environment Variable Setup

Instead of passing token each time:

```bash
# Set token once
export HF_TOKEN=your_huggingface_token

# Then use without --token
python dataset_builder/workflow.py YOUR_USERNAME/barcelona-audio
```

Or use HuggingFace CLI:

```bash
huggingface-cli login
```

## Common Commands

```bash
# Build only (don't upload)
python dataset_builder/workflow.py --no-push

# Make dataset private
python dataset_builder/workflow.py YOUR_USERNAME/barcelona-audio --private

# Custom directories
python dataset_builder/workflow.py YOUR_USERNAME/barcelona-audio \
    --data-dir /path/to/recordings \
    --output-dir /path/to/output
```

## Directory Structure

Your recordings should be organized as:

```
data/recordings/
├── Ciutat vella/
│   ├── monday/
│   │   ├── morning/
│   │   │   ├── traffic_lat41.3851_lon2.1734.m4a
│   │   │   └── ambient_lat41.3849_lon2.1738.m4a
│   │   ├── midday/
│   │   ├── evening/
│   │   └── night/
│   ├── tuesday/
│   └── ...
├── Eixample/
└── ...
```

## What Gets Extracted

From your audio files, the dataset captures:

- **audio**: The audio file itself
- **date**: File modification date
- **time_of_day**: From folder structure (morning/midday/evening/night)
- **district**: From folder structure
- **neighborhood**: From coordinates using Nominatim geocoding
- **typology**: From filename (traffic, ambient, etc.)
- **day_of_week**: From folder structure
- **filename**: Original filename
- **latitude**: From filename
- **longitude**: From filename

## Verify YAMNet Configuration

Check YAMNet audio settings:

```bash
# Show YAMNet configuration
python dataset_builder/build_dataset.py --info

# Inspect dataset samples
python dataset_builder/yamnet_example.py --dataset dataset_output
```

## Troubleshooting

**No recordings found**: Ensure directory structure is correct and files have supported extensions (.m4a, .mp3, .wav, .flac, .ogg)

**Neighborhood shows "unknown"**: Add proper coordinates to filename

**HuggingFace authentication fails**: Run `huggingface-cli login` or set `HF_TOKEN` environment variable

## Next Steps

1. Add your first recordings to the appropriate folders
2. Run the workflow script
3. View your dataset on HuggingFace
4. Start YAMNet training - audio is already preprocessed! (see [YAMNET_TRAINING.md](YAMNET_TRAINING.md))
5. As you collect more recordings, simply re-run the workflow to update
