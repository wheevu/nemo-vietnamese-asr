#!/usr/bin/env python3
"""
Prepare manifest files for NVIDIA NeMo ASR training.

This script creates train, validation, and test manifest files from
audio files and their corresponding transcripts.
"""

import argparse
import json
import logging
import os
import random
import re
import string
from pathlib import Path

import soundfile as sf

LOGGER = logging.getLogger(__name__)

# Directory paths
AUDIO_DIR = Path("./audio")
TRANSCRIPTS_DIR = Path("./transcripts")

# Output manifest files
TRAIN_MANIFEST = "train_manifest.json"
VAL_MANIFEST = "val_manifest.json"
TEST_MANIFEST = "test_manifest.json"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Audio compliance
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def normalize_text(text: str) -> str:
    """Normalize transcript text for training.

    Why: NeMo CTC training is sensitive to label noise. This normalization keeps
    the content but removes punctuation differences and normalizes whitespace.

    Args:
        text: Raw transcript text.

    Returns:
        Normalized text (lowercase, punctuation removed except apostrophes).
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation except apostrophes
    # Create a translation table that removes punctuation but keeps apostrophes
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punctuation_to_remove))
    
    # Normalize whitespace (collapse multiple spaces, strip leading/trailing)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_audio_duration(audio_path: Path) -> float:
    """Compute audio duration (seconds) using `soundfile`.

    Args:
        audio_path: Path to a WAV file.

    Returns:
        Duration in seconds.
    """
    with sf.SoundFile(audio_path) as audio:
        duration = len(audio) / audio.samplerate
    return duration


def is_audio_compliant(audio_path: Path) -> bool:
    """Check audio format compliance for NeMo training.

    Args:
        audio_path: Path to a WAV file.

    Returns:
        True if audio is 16kHz mono WAV, otherwise False.
    """
    try:
        info = sf.info(audio_path)
    except RuntimeError:
        return False
    return (
        info.samplerate == TARGET_SAMPLE_RATE
        and info.channels == TARGET_CHANNELS
        and info.format.upper() == "WAV"
    )


def create_master_list() -> list:
    """Build a list of valid training samples from local artifacts.

    Why: we validate upfront so GPU training doesn't fail later due to missing
    files, empty transcripts, or unreadable audio.

    Returns:
        List of sample dicts with keys: `audio_filepath` (absolute path),
        `duration` (seconds), and `text` (normalized transcript).
    """
    master_list = []
    
    # Ensure directories exist
    if not AUDIO_DIR.exists():
        LOGGER.error("Audio directory '%s' does not exist.", AUDIO_DIR)
        return master_list

    if not TRANSCRIPTS_DIR.exists():
        LOGGER.error("Transcripts directory '%s' does not exist.", TRANSCRIPTS_DIR)
        return master_list

    # Iterate through all .wav files in the audio directory
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        LOGGER.warning("No .wav files found in '%s'", AUDIO_DIR)
        return master_list

    LOGGER.info("Found %s audio files. Processing...", len(wav_files))
    
    for audio_path in wav_files:
        # Derive base name (e.g., 'video_id_1' from 'video_id_1.wav')
        base_name = audio_path.stem
        
        # Construct expected transcript path
        transcript_path = TRANSCRIPTS_DIR / f"{base_name}.txt"
        
        # Validation: Check if transcript exists
        if not transcript_path.exists():
            LOGGER.warning("Transcript not found for '%s', skipping.", base_name)
            continue

        # Validation: Check if transcript is not empty
        transcript_text = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript_text:
            LOGGER.warning("Transcript is empty for '%s', skipping.", base_name)
            continue

        if not is_audio_compliant(audio_path):
            LOGGER.warning("Audio '%s' is not 16kHz mono WAV, skipping.", base_name)
            continue

        try:
            # Get audio duration
            duration = get_audio_duration(audio_path)

            # Normalize the transcript text
            normalized_text = normalize_text(transcript_text)

            if not normalized_text:
                LOGGER.warning("Normalized transcript is empty for '%s', skipping.", base_name)
                continue

            # Create sample entry with absolute path
            sample = {
                "audio_filepath": str(audio_path.resolve()),
                "duration": round(duration, 3),
                "text": normalized_text,
            }

            master_list.append(sample)

        except Exception as e:
            LOGGER.warning("Error processing '%s': %s, skipping.", base_name, e)
            continue
    
    LOGGER.info("Successfully processed %s valid samples.", len(master_list))
    return master_list


def split_data(master_list: list) -> tuple:
    """Split samples into train/val/test partitions.

    Args:
        master_list: Shuffled list of sample dicts.

    Returns:
        Tuple `(train_set, val_set, test_set)`.
    """
    total = len(master_list)
    
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_set = master_list[:train_end]
    val_set = master_list[train_end:val_end]
    test_set = master_list[val_end:]
    
    return train_set, val_set, test_set


def write_manifest(data: list, filepath: str) -> None:
    """Write a NeMo JSONL manifest to disk.

    Why JSONL: NeMo expects one JSON object per line for streaming-friendly
    loading.

    Args:
        data: List of manifest entries.
        filepath: Output file path.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")
    
    LOGGER.info("Written %s entries to '%s'", len(data), filepath)


def main():
    """Script entry point.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Prepare NeMo ASR manifest files.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic train/val/test split.",
    )
    args = parser.parse_args()

    LOGGER.info("%s", "=" * 60)
    LOGGER.info("NeMo ASR Data Preparation Script")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Audio directory: %s", AUDIO_DIR.resolve())
    LOGGER.info("Transcripts directory: %s", TRANSCRIPTS_DIR.resolve())

    # Create master list of valid samples
    master_list = create_master_list()

    if not master_list:
        LOGGER.error("No valid samples found. Exiting.")
        return 1

    # Shuffle the master list
    if args.seed is not None:
        random.seed(args.seed)
        LOGGER.info("Shuffling data with seed %s...", args.seed)
    else:
        LOGGER.info("Shuffling data...")
    random.shuffle(master_list)

    # Split into train, validation, and test sets
    train_set, val_set, test_set = split_data(master_list)

    LOGGER.info("\nData split:")
    LOGGER.info("  Training:   %s samples (%s%%)", len(train_set), TRAIN_RATIO * 100)
    LOGGER.info("  Validation: %s samples (%s%%)", len(val_set), VAL_RATIO * 100)
    LOGGER.info("  Test:       %s samples (%s%%)", len(test_set), TEST_RATIO * 100)

    # Write manifest files
    write_manifest(train_set, TRAIN_MANIFEST)
    write_manifest(val_set, VAL_MANIFEST)
    write_manifest(test_set, TEST_MANIFEST)

    # Calculate total duration
    total_duration = sum(s["duration"] for s in master_list)
    train_duration = sum(s["duration"] for s in train_set)
    val_duration = sum(s["duration"] for s in val_set)
    test_duration = sum(s["duration"] for s in test_set)

    LOGGER.info("\nTotal audio duration: %.2f hours", total_duration / 3600)
    LOGGER.info("  Training:   %.2f hours", train_duration / 3600)
    LOGGER.info("  Validation: %.2f hours", val_duration / 3600)
    LOGGER.info("  Test:       %.2f hours", test_duration / 3600)

    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("Data preparation complete!")
    LOGGER.info("%s", "=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())

