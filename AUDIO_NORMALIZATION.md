# Audio Normalization for YAMNet Training

## Summary

All audio recordings are automatically normalized to **exactly 30 seconds** during dataset creation for consistent YAMNet training.

## How It Works

### During Dataset Build

When you run `python dataset_builder/build_dataset.py`:

1. **Scans** your recordings from `data/recordings/`
2. **Processes** each audio file:
   - Decodes .m4a/.mp3/.wav/etc → raw waveform
   - Resamples to 16kHz
   - Converts to mono
   - **Normalizes duration**:
     - If > 30s: Trims to exactly 30 seconds
     - If < 30s: Zero-pads to exactly 30 seconds
     - If = 30s: Keeps as is
3. **Saves** normalized .wav files alongside originals
4. **Creates** dataset using normalized files

### Result

All audio in your dataset will have:
- ✓ Sampling rate: 16,000 Hz
- ✓ Channels: 1 (mono)
- ✓ Duration: 30.0 seconds
- ✓ Samples: 480,000
- ✓ Format: WAV (PCM 16-bit)

## File Structure

```
data/recordings/Eixample/monday/morning/
├── traffic_lat41.38_lon2.17.m4a              # Your original recording
└── traffic_lat41.38_lon2.17_normalized.wav   # Auto-generated normalized version
```

The dataset uses the `_normalized.wav` files for training.

## Commands

### Check Configuration

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
==================================================
```

### Build Dataset

```bash
python dataset_builder/build_dataset.py
```

This will:
- Process all audio files
- Create normalized versions
- Build the dataset

### Inspect Dataset

```bash
python dataset_builder/yamnet_example.py --dataset dataset_output
```

Shows validation that audio is exactly 30 seconds.

### Test Normalization

```bash
# Create test audio files of different durations
python dataset_builder/test_normalization.py --create-tests

# This creates:
# - short_25s.wav (will be padded to 30s)
# - exact_30s.wav (unchanged)
# - long_35s.wav (will be trimmed to 30s)
```

## Benefits for Training

### 1. Consistent Batch Shapes
All audio has the same shape `(480000,)`:
```python
for batch in dataloader:
    # All samples are exactly 480,000 samples
    assert batch['audio'].shape == (batch_size, 480000)
```

### 2. No Runtime Padding
No need to pad/trim during training:
```python
# Just load and train!
audio = sample['audio']['array']
predictions = yamnet_model(audio)  # No preprocessing needed
```

### 3. Predictable Memory Usage
Know exactly how much memory each sample uses:
- 1 sample = 480,000 float32 values = ~1.83 MB
- Batch of 32 = ~58.6 MB

## FAQ

### Q: Do I need to record exactly 30 seconds?
**A:** No! Record around 25-35 seconds. The system handles normalization automatically.

### Q: What if my recordings are much shorter (e.g., 10 seconds)?
**A:** They'll be zero-padded to 30 seconds. The model will see silence at the end.

### Q: What if my recordings are much longer (e.g., 60 seconds)?
**A:** They'll be trimmed to the first 30 seconds. Consider splitting long recordings into multiple 30-second clips before building the dataset.

### Q: Can I change the target duration?
**A:** Yes! Edit the `YAMNET_DURATION` constant in `build_dataset.py`. However, 30 seconds is optimal for YAMNet.

### Q: Will this affect my original recordings?
**A:** No! Original files are preserved. Normalized versions are created as separate `_normalized.wav` files.

### Q: Do I need to rebuild when adding new recordings?
**A:** Yes. Run `python dataset_builder/build_dataset.py` again to process new files.

### Q: What happens to the padding (zero-filled areas)?
**A:** YAMNet can handle silence. During training, you may want to use data augmentation or focus on the active audio portions.

## Technical Details

### Normalization Algorithm

```python
def normalize(audio, target_samples=480000):
    current_samples = len(audio)
    
    if current_samples > target_samples:
        # Trim
        return audio[:target_samples]
    elif current_samples < target_samples:
        # Pad with zeros
        padding = target_samples - current_samples
        return np.pad(audio, (0, padding), mode='constant')
    else:
        return audio
```

### Why 30 Seconds?

- **YAMNet frames**: YAMNet processes audio in 0.96-second frames
- **30 seconds** = ~31 frames, providing good temporal context
- **Batch training**: Consistent shapes simplify batch processing
- **Standard practice**: Common duration for audio ML tasks

### Processing Time

- ~0.5-2 seconds per file depending on:
  - Original format (.m4a slower than .wav)
  - Original sample rate (higher = more resampling)
  - File duration
  - CPU performance

For 1000 files: ~10-30 minutes total processing time.

## Troubleshooting

### "Failed to normalize" error
- Check audio file is not corrupted
- Ensure librosa and soundfile are installed
- Verify file format is supported

### Normalized files taking too much space
- .wav files are uncompressed (~5-10 MB per 30-second file)
- Consider archiving original .m4a files after dataset creation
- Or keep only normalized files and metadata

### Different durations in inspect output
- Check that you're loading the built dataset, not original files
- Verify normalized files were created successfully
- Run build process again if files are missing

## Next Steps

1. **Add recordings** to appropriate folders
2. **Build dataset**: `python dataset_builder/build_dataset.py`
3. **Verify**: `python dataset_builder/yamnet_example.py`
4. **Push to HuggingFace**: `python dataset_builder/push_to_huggingface.py username/dataset`
5. **Train YAMNet** with consistent 30-second audio!
