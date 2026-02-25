# YAMNet Training with Audio Bias Pavilion Dataset

## Overview

This dataset is configured for training or fine-tuning the YAMNet audio classification model. All audio preprocessing is handled automatically by the HuggingFace datasets library.

## YAMNet Requirements

YAMNet expects audio with specific characteristics:

- **Sampling Rate**: 16 kHz
- **Channels**: Mono (1 channel)
- **Format**: Waveform (NumPy array)
- **Duration**: Exactly 30 seconds (normalized)

## Automatic Preprocessing

✅ **No manual preprocessing needed!**

The dataset automatically handles:

1. **Decoding**: Converts .m4a, .mp3, .wav, and other formats to raw waveform
2. **Resampling**: Automatically resamples any audio to 16 kHz
3. **Mono Conversion**: Converts stereo recordings to mono
4. **Duration Normalization**: All clips are exactly 30 seconds
   - Longer clips are trimmed to 30s
   - Shorter clips are zero-padded to 30s
5. **Format**: Returns audio as NumPy arrays ready for YAMNet

## Verifying Configuration

Check the YAMNet audio settings:

```bash
python dataset_builder/build_dataset.py --info
```

Output:
```
YAMNet Audio Configuration:
==================================================
Sampling Rate: 16000 Hz
Channels: 1 (mono)
Duration: 30.0 seconds (exactly)
Target Samples: 480000
Preprocessing: automatic normalization + HuggingFace resampling

Notes:
  • Audio files are automatically decoded from .m4a/.mp3/etc
  • Resampling to 16kHz happens during dataset loading
  • Stereo files are automatically converted to mono
  • All clips normalized to exactly 30.0 seconds
  • Longer clips are trimmed, shorter clips are zero-padded
==================================================
```

## Using the Dataset for YAMNet Training

### Loading the Dataset

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("your-username/barcelona-audio")

# Or load from disk
dataset = load_from_disk("dataset_output")

# Audio is automatically preprocessed
sample = dataset['train'][0]
print(sample['audio'])
# Output: {'array': array([...]), 'sampling_rate': 16000, 'path': '...'}
```

### Accessing Audio Data

```python
# Get audio waveform (NumPy array)
audio_array = sample['audio']['array']
sampling_rate = sample['audio']['sampling_rate']

print(f"Shape: {audio_array.shape}")        # (480000,) - exactly 30 seconds
print(f"Sample rate: {sampling_rate} Hz")   # 16000 Hz
print(f"Duration: {len(audio_array) / sampling_rate:.2f} seconds")  # 30.00 seconds
```

**Note**: All audio clips are normalized to exactly 30.0 seconds (480,000 samples at 16kHz) during dataset creation. This happens automatically when you run `build_dataset.py`.

### Example Training Pipeline

```python
import tensorflow as tf
import tensorflow_hub as hub
from datasets import load_dataset

# Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load your dataset
dataset = load_dataset("your-username/barcelona-audio")

def preprocess_for_yamnet(batch):
    """Prepare batch for YAMNet."""
    # Audio is already at 16kHz mono - just extract the array
    waveforms = [audio['array'] for audio in batch['audio']]
    return {'waveform': waveforms, 'label': batch['typology']}

# Process dataset
processed = dataset.map(
    preprocess_for_yamnet,
    batched=True,
    remove_columns=dataset.column_names
)

# The waveforms are now ready for YAMNet!
```

### Fine-tuning YAMNet

```python
import tensorflow as tf
import tensorflow_hub as hub

# Load pre-trained YAMNet
yamnet = hub.KerasLayer('https://tfhub.dev/google/yamnet/1', trainable=False)

# Build classifier on top
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None,), dtype=tf.float32),
    yamnet,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Your audio is already in the correct format!
for batch in dataset:
    audio_waveform = batch['audio']['array']
    # Ready to pass to model
    predictions = model(audio_waveform)
```

## Audio File Recommendations

### File Format
- ✅ .m4a (automatically decoded)
- ✅ .mp3 (automatically decoded)
- ✅ .wav (any sample rate, will be resampled)
- ✅ .flac (lossless, automatically decoded)
- ✅ .ogg (automatically decoded)

### Recording Quality
- **Duration**: ~30 seconds (will be normalized to exactly 30s)
  - Longer clips: automatically trimmed to 30s
  - Shorter clips: automatically zero-padded to 30s
- **Sample Rate**: Any (will be resampled to 16kHz)
- **Channels**: Mono or stereo (stereo → mono automatically)
- **Bit Depth**: Any (16-bit, 24-bit, etc.)

### What NOT to worry about
❌ Manual resampling
❌ Converting to .wav
❌ Mono conversion
❌ Amplitude normalization (YAMNet handles this internally)
❌ Padding/trimming to fixed length (automatically done during build)

## Technical Details

### Two-Stage Processing

**Stage 1: During Dataset Build** (when you run `build_dataset.py`):
1. Load each audio file using librosa
2. Resample to 16kHz
3. Convert to mono
4. Normalize duration to exactly 30 seconds (trim or pad)
5. Save as normalized .wav files
6. Create dataset with normalized files

**Stage 2: During Dataset Loading** (when you load for training):
- HuggingFace loads the pre-normalized .wav files
- Audio is already at correct format (16kHz, mono, 30s)
- Ready for direct input to YAMNet

### HuggingFace Audio Feature

The dataset uses:
```python
from datasets import Audio

Audio(sampling_rate=16000, mono=True)
```

This ensures HuggingFace loads the pre-normalized audio correctly:
1. Reads normalized .wav files
2. Confirms sampling rate (16000 Hz)
3. Confirms mono channel
4. Returns as NumPy array

### Processing Flow

```python
# When you build the dataset (python build_dataset.py):
# 1. Finds original .m4a files
# 2. Loads and decodes each file
# 3. Resamples to 16kHz, converts to mono
# 4. Normalizes to exactly 30 seconds (trim/pad)
# 5. Saves as <filename>_normalized.wav
# 6. Creates dataset pointing to normalized files

# When you load the dataset for training:
sample = dataset[0]
audio = sample['audio']['array']  # <- Loads pre-normalized .wav
# Audio is already: 16kHz, mono, exactly 30 seconds (480,000 samples)
```

### Caching

Since audio is pre-normalized during build, loading is fast:
```python
# Audio files are already processed, so loading is efficient
dataset = load_dataset("your-username/barcelona-audio")
# No additional caching needed - files are already optimized!
```

## Common Issues

### Need to rebuild after adding recordings?
Yes! When you add new audio files, run the build process again:
```bash
python dataset_builder/build_dataset.py
```
This will create normalized versions of new recordings.

### Out of memory during build?
If you have many large recordings, the normalization process might use RAM:
- Process smaller batches at a time
- Ensure you have enough disk space for normalized .wav files

### Out of memory during training?
30-second clips at 16kHz = exactly 480,000 samples per file:
- Use `dataset.select(range(n))` for smaller subset
- Use streaming: `load_dataset(..., streaming=True)`
- Batch your data appropriately

### Audio shape inconsistency?
All audio is guaranteed to be the same shape after normalization:
```python
for sample in dataset:
    assert sample['audio']['array'].shape == (480000,)  # Always true!
    assert sample['audio']['sampling_rate'] == 16000    # Always true!
```

### Where are the normalized files?
Normalized .wav files are created alongside your original recordings:
```
data/recordings/<district>/<day>/<time>/
  ├── traffic_lat41.38_lon2.17.m4a        # Original
  └── traffic_lat41.38_lon2.17_normalized.wav  # Normalized (used in dataset)
```

## Summary

✅ **Your ~30-second .m4a recordings are perfect for YAMNet!**

✅ **Automatic normalization to exactly 30 seconds:**
- Longer clips → trimmed to 30s
- Shorter clips → zero-padded to 30s
- All clips → 16kHz mono, 480,000 samples

✅ **Just build and train:**
```bash
# Build dataset (this normalizes all audio to 30s)
python dataset_builder/build_dataset.py

# Push to HuggingFace
python dataset_builder/push_to_huggingface.py your-username/dataset-name

# Use for training - all audio is exactly 30 seconds!
# Perfect for batch training with consistent shapes
```

## References

- [YAMNet on TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- [HuggingFace Audio Feature](https://huggingface.co/docs/datasets/audio_dataset)
- [YAMNet Paper](https://arxiv.org/abs/2104.00298)
