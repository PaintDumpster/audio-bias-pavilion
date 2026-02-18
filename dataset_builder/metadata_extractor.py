"""Tools for reading metadata from iPhone Voice Memo .m4a files."""

from typing import Any, Dict

from mutagen.mp4 import MP4


# Maps the cryptic QuickTime tag names to plain English labels we actually use.
TAG_NAME_OVERRIDES = {
    "\xa9nam": "title",
    "\xa9ART": "artist",
    "\xa9alb": "album",
    "\xa9day": "recorded_on",
    "\xa9gen": "genre",
    "com.apple.voice.recorder.category": "voice_memo_category",
    "com.apple.voice.recorder.device-id": "device_identifier",
    "com.apple.voice.recorder.data-source": "microphone_source",
    "com.apple.voice.recorder.level": "recording_level",
    "com.apple.voice.recorder.location": "location_note",
    "com.apple.voice.recorder.latitude": "latitude",
    "com.apple.voice.recorder.longitude": "longitude",
}


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Reads every available metadata field from an .m4a file."""

    try:
        audio = MP4(file_path)
    except Exception as error:
        raise ValueError(f"Could not open {file_path}: {error}") from error

    tags = audio.tags or {}

    raw_tags: Dict[str, Any] = dict(tags.items())

    normalized_tags: Dict[str, Any] = {}
    for key, value in raw_tags.items():
        readable_key = TAG_NAME_OVERRIDES.get(key, key.replace(" ", "_").lower())
        normalized_tags[readable_key] = value

    info = audio.info
    technical_details = {
        "duration_seconds": getattr(info, "length", None),
        "bitrate": getattr(info, "bitrate", None),
        "sample_rate": getattr(info, "sample_rate", None),
        "channels": getattr(info, "channels", None),
    }

    return {
        "file_path": file_path,
        "raw_tags": raw_tags,
        "normalized_tags": normalized_tags,
        "technical_details": technical_details,
    }

def print_metadata_summary(file_path: str) -> None:
    """Helper that prints the extracted metadata for quick manual checks."""

    metadata = extract_metadata(file_path)

    print(f"File: {metadata['file_path']}")
    print("\nNormalized tags:")
    for key, value in metadata["normalized_tags"].items():
        print(f"  {key}: {value}")

    print("\nTechnical details:")
    for key, value in metadata["technical_details"].items():
        print(f"  {key}: {value}")

    print("\nRaw tags (for reference):")
    for key, value in metadata["raw_tags"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print_metadata_summary("data/recordings/Ciudad bella/Baba Kebab 2.m4a")