# tests/conftest.py
"""
Pytest configuration and shared fixtures for the Vietnamese ASR test suite.

This module provides reusable test fixtures for:
- Mock audio files (16kHz mono WAV)
- Mock NeMo-compatible manifest files
- Temporary file management

These fixtures follow the pattern of creating minimal, valid test artifacts
that mirror the real data pipeline outputs.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# =============================================================================
# Audio Fixtures
# =============================================================================

@pytest.fixture
def mock_audio_file(tmp_path: Path) -> Path:
    """
    Create a valid, silent, 1-second, 16kHz mono WAV file.
    
    This fixture generates a minimal audio file that meets NeMo Conformer
    requirements:
    - Sample rate: 16000 Hz (required for NeMo ASR models)
    - Channels: 1 (mono)
    - Duration: 1 second
    - Content: Silence (zeros) - sufficient for format validation
    
    Args:
        tmp_path: pytest built-in fixture providing a temporary directory
        
    Returns:
        Path to the created WAV file
    """
    audio_path = tmp_path / "test_audio.wav"
    
    # Generate 1 second of silence at 16kHz
    sample_rate = 16000
    duration_seconds = 1.0
    num_samples = int(sample_rate * duration_seconds)
    
    # Create silent audio (zeros) - float32 format
    audio_data = np.zeros(num_samples, dtype=np.float32)
    
    # Write the WAV file
    sf.write(audio_path, audio_data, sample_rate)
    
    return audio_path


@pytest.fixture
def mock_audio_file_stereo(tmp_path: Path) -> Path:
    """
    Create an INVALID stereo audio file for negative testing.
    
    This fixture generates a stereo (2-channel) audio file that should
    fail audio compliance tests, demonstrating proper validation.
    
    Returns:
        Path to the created stereo WAV file
    """
    audio_path = tmp_path / "test_audio_stereo.wav"
    
    sample_rate = 16000
    duration_seconds = 1.0
    num_samples = int(sample_rate * duration_seconds)
    
    # Create stereo audio (2 channels)
    audio_data = np.zeros((num_samples, 2), dtype=np.float32)
    
    sf.write(audio_path, audio_data, sample_rate)
    
    return audio_path


@pytest.fixture
def mock_audio_file_wrong_sample_rate(tmp_path: Path) -> Path:
    """
    Create an audio file with incorrect sample rate for negative testing.
    
    This fixture generates a 44.1kHz audio file (CD quality) that should
    fail NeMo compliance tests, which require 16kHz.
    
    Returns:
        Path to the created WAV file with wrong sample rate
    """
    audio_path = tmp_path / "test_audio_44k.wav"
    
    # Use CD-quality sample rate (incorrect for NeMo)
    sample_rate = 44100
    duration_seconds = 1.0
    num_samples = int(sample_rate * duration_seconds)
    
    audio_data = np.zeros(num_samples, dtype=np.float32)
    
    sf.write(audio_path, audio_data, sample_rate)
    
    return audio_path


# =============================================================================
# Manifest Fixtures
# =============================================================================

@pytest.fixture
def mock_manifest(tmp_path: Path, mock_audio_file: Path) -> Path:
    """
    Create a valid, single-line NeMo manifest file (.jsonl).
    
    This fixture creates a minimal but complete manifest entry that
    follows the NeMo ASR manifest format:
    - audio_filepath: absolute path to audio file
    - text: normalized transcript (lowercase, no punctuation)
    - duration: audio duration in seconds
    
    Args:
        tmp_path: pytest built-in fixture providing a temporary directory
        mock_audio_file: fixture providing a valid audio file
        
    Returns:
        Path to the created manifest file
    """
    manifest_path = tmp_path / "test_manifest.json"
    
    # Create a valid manifest entry
    manifest_entry = {
        "audio_filepath": str(mock_audio_file),
        "text": "đây là bản ghi âm tiếng việt",  # Vietnamese text example
        "duration": 1.0
    }
    
    # Write as JSONL (JSON Lines format - one JSON object per line)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    
    return manifest_path


@pytest.fixture
def mock_manifest_multiple_entries(tmp_path: Path, mock_audio_file: Path) -> Path:
    """
    Create a multi-line manifest file for testing batch operations.
    
    Returns:
        Path to the manifest file with multiple entries
    """
    manifest_path = tmp_path / "test_manifest_multi.json"
    
    entries = [
        {
            "audio_filepath": str(mock_audio_file),
            "text": "xin chào các bạn",
            "duration": 1.0
        },
        {
            "audio_filepath": str(mock_audio_file),
            "text": "cảm ơn rất nhiều",
            "duration": 1.0
        },
        {
            "audio_filepath": str(mock_audio_file),
            "text": "tạm biệt nhé",
            "duration": 1.0
        },
    ]
    
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return manifest_path


@pytest.fixture
def mock_manifest_missing_audio(tmp_path: Path) -> Path:
    """
    Create a manifest pointing to a non-existent audio file.
    
    This fixture is for testing error handling when manifest
    references missing files.
    
    Returns:
        Path to the manifest with invalid audio reference
    """
    manifest_path = tmp_path / "test_manifest_broken.json"
    
    manifest_entry = {
        "audio_filepath": "/nonexistent/path/audio.wav",
        "text": "this audio file does not exist",
        "duration": 1.0
    }
    
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    
    return manifest_path


# =============================================================================
# Text File Fixtures
# =============================================================================

@pytest.fixture
def mock_caption_file(tmp_path: Path) -> Path:
    """
    Create a messy caption file for testing text normalization.
    
    This fixture creates a file with common caption issues:
    - Extra whitespace
    - Blank lines
    - HTML entities
    - Duplicate lines
    
    Returns:
        Path to the created caption file
    """
    caption_path = tmp_path / "test_captions.txt"
    
    # Deliberately messy content to test normalization
    messy_content = """
    Xin chào các bạn  
    
       Hôm nay   chúng ta   sẽ học
    
    
    &amp; đây là ký tự HTML
    Xin chào các bạn
    """
    
    caption_path.write_text(messy_content, encoding="utf-8")
    
    return caption_path


@pytest.fixture
def mock_vtt_file(tmp_path: Path) -> Path:
    """
    Create a WebVTT caption file for testing VTT-specific parsing.
    
    Returns:
        Path to the created VTT file
    """
    vtt_path = tmp_path / "test_captions.vtt"
    
    vtt_content = """WEBVTT
Kind: captions
Language: vi

00:00:00.000 --> 00:00:02.000
Xin chào các bạn

00:00:02.000 --> 00:00:04.000
<c>Đây là phụ đề</c>

00:00:04.000 --> 00:00:06.000
Cảm ơn đã xem
"""
    
    vtt_path.write_text(vtt_content, encoding="utf-8")
    
    return vtt_path
