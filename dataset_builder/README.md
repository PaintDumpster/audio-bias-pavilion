# Audio Bias Pavilion - Dataset Builder

This repository contains tools for building and managing a HuggingFace dataset of geolocated audio recordings from Barcelona neighborhoods.

## Dataset Structure

The dataset is tabular with the following columns:

| Column | Description |
|--------|-------------|
| `audio` | Audio recording file (.m4a, .mp3, .wav, etc.) |
| `date` | Recording date (YYYY-MM-DD) |
| `time_of_day` | Time period (morning, midday, evening, night) |
| `district` | Barcelona district name |
| `neighborhood` | Specific neighborhood (extracted from coordinates) |
| `typology` | Sound typology (traffic, ambient, voice, etc.) |
| `day_of_week` | Day of the week |
| `filename` | Original filename |
| `latitude` | Recording latitude |
| `longitude` | Recording longitude |

## Directory Structure

```
audio-bias-pavilion/
├── data/
│   └── recordings/
│       └── <district>/
│           └── <day_of_week>/
│               └── <time_of_day>/
│                   └── <recording files>
└── dataset_builder/
    ├── foldergenerator.py      # Creates folder structure
    ├── build_dataset.py         # Builds HuggingFace dataset
    └── push_to_huggingface.py   # Pushes to HuggingFace Hub
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Create Folder Structure

Generate the folder hierarchy for organizing recordings:

```bash
python dataset_builder/foldergenerator.py
```

Or for specific districts:
```bash
python dataset_builder/foldergenerator.py "Ciutat vella" "Eixample" "Gracia"
```

### 2. Add Recordings

Place audio recordings in the appropriate folders following this structure:
```
data/recordings/<district>/<day>/<time_of_day>/<recording>.m4a
```

#### Filename Conventions

For best metadata extraction, name your audio files following these patterns:

**Include coordinates** (for neighborhood detection):
- `lat41.3851_lon2.1734.m4a`
- `traffic_lat41.3851_lon2.1734.m4a`
- `ambient@41.3851,2.1734.m4a`

**Include typology** (sound type):
- `traffic_lat41.3851_lon2.1734.m4a`
- `ambient_recording.m4a`
- `construction_noise.m4a`

Common typologies: `traffic`, `ambient`, `voice`, `music`, `construction`, `nature`, `urban`, `quiet`

### 3. Build Dataset

Build the HuggingFace dataset from your recordings:

```bash
python dataset_builder/build_dataset.py
```

Options:
- `--data-dir`: Base directory with recordings (default: `data/recordings`)
- `--output-dir`: Where to save the dataset (default: `dataset_output`)

Example:
```bash
python dataset_builder/build_dataset.py --data-dir data/recordings --output-dir my_dataset
```

### 4. Push to HuggingFace

Upload your dataset to HuggingFace Hub:

```bash
# First, log in to HuggingFace
huggingface-cli login

# Then push the dataset
python dataset_builder/push_to_huggingface.py <username>/<dataset-name>
```

Options:
- `--data-dir`: Source directory with recordings
- `--dataset-dir`: Load existing built dataset instead of building
- `--token`: HuggingFace API token (or use HF_TOKEN env variable)
- `--private`: Make the dataset private

Examples:
```bash
# Build and push in one command
python dataset_builder/push_to_huggingface.py myusername/audio-bias-barcelona

# Push existing dataset
python dataset_builder/push_to_huggingface.py myusername/audio-bias-barcelona --dataset-dir dataset_output

# Use environment variable for token
export HF_TOKEN=your_token_here
python dataset_builder/push_to_huggingface.py myusername/audio-bias-barcelona --private
```

### 5. Update Dataset with New Recordings

As you add more recordings, simply run the scripts again:

```bash
# Rebuild dataset with new recordings
python dataset_builder/build_dataset.py

# Push updated dataset to HuggingFace
python dataset_builder/push_to_huggingface.py <username>/<dataset-name>
```

## Metadata Extraction

### Typology Detection
The script extracts sound typology from filenames using regex patterns:
- Explicit typology: `traffic_recording.m4a` → `traffic`
- Pattern matching: `ambient_noise_lat41.38_lon2.17.m4a` → `ambient`
- First word: `construction.m4a` → `construction`

### Neighborhood Detection
The script uses Nominatim (OpenStreetMap) to reverse geocode coordinates:
1. Extracts lat/lon from filename
2. Queries Nominatim for neighborhood name
3. Caches results to avoid repeated API calls
4. Falls back to "unknown" if coordinates not found

### Date Extraction
Recording date is derived from:
1. File modification timestamp (default)
2. Future: filename date patterns if implemented

## Example Workflow

```bash
# 1. Set up folders
python dataset_builder/foldergenerator.py

# 2. Record audio and save to appropriate folders
# Example: data/recordings/Eixample/monday/morning/traffic_lat41.3851_lon2.1734.m4a

# 3. Build dataset
python dataset_builder/build_dataset.py

# 4. Log in to HuggingFace
huggingface-cli login

# 5. Push to HuggingFace
python dataset_builder/push_to_huggingface.py myusername/barcelona-audio-bias

# 6. Later, add more recordings and update
python dataset_builder/push_to_huggingface.py myusername/barcelona-audio-bias
```

## Loading the Dataset

Once published, anyone can load your dataset:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("myusername/barcelona-audio-bias")

# Access audio and metadata
for sample in dataset["train"]:
    audio = sample["audio"]  # Audio data
    district = sample["district"]
    typology = sample["typology"]
    # ... etc
```

## Notes

- **Geocoding limits**: Nominatim has rate limits. The script includes delays and retries.
- **Empty dataset**: The script handles empty folders gracefully and creates a valid dataset schema.
- **Audio formats**: Supports .m4a, .mp3, .wav, .flac, .ogg files.
- **Coordinates**: Expected to be in the Barcelona area (lat: 41-42, lon: 1.5-2.5).

## Districts

The default districts configured are:
- Ciutat vella
- Sants-montjuïc
- Gràcia
- Les corts
- Eixample
- Sarrià-Sant Gervasi
- Sant Martí
- Nou Barris
- Horta-Guinardó
- Sant Andreu

## License

See LICENSE file.
