# tests/test_data_integrity.py
"""
Data integrity tests for the Vietnamese ASR pipeline.

This module validates that the data pipeline produces artifacts that meet
NVIDIA NeMo's requirements. These tests catch silent failures that would
otherwise only manifest during training (e.g., wrong sample rate, missing files).

Test Categories:
1. Manifest Structure - Validates JSON schema and required fields
2. Audio Format Compliance - Ensures 16kHz mono WAV format
3. Manifest-Audio Linkage - Integration test for file references

These tests implement the "fail fast" principle: catch data issues during
preparation, not during expensive GPU training.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# =============================================================================
# Manifest Structure Tests
# =============================================================================

class TestManifestStructure:
    """
    Test suite for NeMo manifest file structure validation.
    
    NeMo ASR expects manifests in JSONL format (one JSON object per line)
    with specific required fields:
    - audio_filepath: str - absolute path to WAV file
    - text: str - normalized transcript
    - duration: float - audio duration in seconds
    
    Invalid manifests cause cryptic training errors.
    """
    
    def test_manifest_is_valid_jsonl(self, mock_manifest: Path):
        """
        Test that each line in the manifest is valid JSON.
        
        Malformed JSON will cause NeMo to fail during data loading.
        This test catches encoding issues, trailing commas, etc.
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            line_number = 0
            for line in f:
                line_number += 1
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON on line {line_number}: {e}")
    
    def test_manifest_has_required_keys(self, mock_manifest: Path):
        """
        Test that each manifest entry contains required NeMo fields.
        
        Missing fields will cause KeyError during training or silent
        data corruption.
        """
        required_keys = {"audio_filepath", "text", "duration"}
        
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                missing_keys = required_keys - entry.keys()
                
                assert not missing_keys, (
                    f"Line {line_number} missing required keys: {missing_keys}"
                )
    
    def test_manifest_field_types(self, mock_manifest: Path):
        """
        Test that manifest fields have correct data types.
        
        - audio_filepath: must be string
        - text: must be string
        - duration: must be numeric (int or float)
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                
                # Type checks
                assert isinstance(entry["audio_filepath"], str), (
                    f"Line {line_number}: audio_filepath must be string"
                )
                assert isinstance(entry["text"], str), (
                    f"Line {line_number}: text must be string"
                )
                assert isinstance(entry["duration"], (int, float)), (
                    f"Line {line_number}: duration must be numeric"
                )
    
    def test_manifest_text_not_empty(self, mock_manifest: Path):
        """
        Test that transcript text is not empty.
        
        Empty transcripts are useless for training and indicate
        a data quality issue upstream.
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                text = entry["text"].strip()
                
                assert text, f"Line {line_number}: text field is empty"
    
    def test_manifest_duration_positive(self, mock_manifest: Path):
        """
        Test that duration values are positive.
        
        Zero or negative durations indicate corrupted audio metadata.
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                duration = entry["duration"]
                
                assert duration > 0, (
                    f"Line {line_number}: duration must be positive, got {duration}"
                )
    
    def test_multiple_entries_manifest(self, mock_manifest_multiple_entries: Path):
        """
        Test validation works for manifests with multiple entries.
        """
        entry_count = 0
        required_keys = {"audio_filepath", "text", "duration"}
        
        with mock_manifest_multiple_entries.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                assert required_keys <= entry.keys()
                entry_count += 1
        
        assert entry_count == 3, f"Expected 3 entries, got {entry_count}"


# =============================================================================
# Audio Format Compliance Tests
# =============================================================================

class TestAudioFormatCompliance:
    """
    Test suite for audio file format validation.
    
    NVIDIA NeMo Conformer models require specific audio formats:
    - Sample rate: 16000 Hz (16kHz)
    - Channels: 1 (mono)
    - Format: WAV (PCM)
    
    Incorrect formats cause silent model degradation or runtime errors.
    """
    
    def test_audio_sample_rate_is_16khz(self, mock_audio_file: Path):
        """
        Test that audio files have exactly 16kHz sample rate.
        
        NeMo Conformer models are trained on 16kHz audio. Using audio
        with different sample rates causes:
        - 8kHz: Missing high frequency information
        - 44.1kHz: Unnecessary computation, potential aliasing artifacts
        """
        info = sf.info(mock_audio_file)
        
        assert info.samplerate == 16000, (
            f"Sample rate must be 16000 Hz for NeMo, got {info.samplerate} Hz"
        )
    
    def test_audio_is_mono(self, mock_audio_file: Path):
        """
        Test that audio files are single-channel (mono).
        
        NeMo ASR models expect mono audio. Stereo audio will cause
        dimension mismatches in the feature extraction pipeline.
        """
        info = sf.info(mock_audio_file)
        
        assert info.channels == 1, (
            f"Audio must be mono (1 channel), got {info.channels} channels"
        )
    
    def test_audio_is_wav_format(self, mock_audio_file: Path):
        """
        Test that audio files are in WAV format.
        
        While NeMo supports multiple formats, WAV is the most reliable
        and doesn't require additional decoders.
        """
        info = sf.info(mock_audio_file)
        
        # soundfile reports format as "WAV" for WAV files
        assert info.format == "WAV", (
            f"Audio must be WAV format, got {info.format}"
        )
    
    def test_audio_duration_matches_samples(self, mock_audio_file: Path):
        """
        Test that audio duration is consistent with sample count.
        
        This catches corrupted audio files where metadata doesn't match content.
        """
        info = sf.info(mock_audio_file)
        
        expected_duration = info.frames / info.samplerate
        actual_duration = info.duration
        
        assert abs(expected_duration - actual_duration) < 0.001, (
            f"Duration mismatch: expected {expected_duration}s, got {actual_duration}s"
        )
    
    def test_audio_is_readable(self, mock_audio_file: Path):
        """
        Test that audio data can be loaded without errors.
        
        This catches file corruption that metadata checks might miss.
        """
        try:
            data, sample_rate = sf.read(mock_audio_file)
            assert data is not None
            assert len(data) > 0
        except Exception as e:
            pytest.fail(f"Failed to read audio file: {e}")
    
    # Negative tests - these should fail compliance checks
    
    def test_stereo_audio_fails_mono_check(self, mock_audio_file_stereo: Path):
        """
        Test that stereo audio is correctly identified as non-compliant.
        
        This negative test verifies our validation catches incorrect formats.
        """
        info = sf.info(mock_audio_file_stereo)
        
        # This SHOULD fail - stereo is not compliant
        assert info.channels != 1, "Test fixture should be stereo"
    
    def test_wrong_sample_rate_fails_check(self, mock_audio_file_wrong_sample_rate: Path):
        """
        Test that non-16kHz audio is correctly identified as non-compliant.
        """
        info = sf.info(mock_audio_file_wrong_sample_rate)
        
        # This SHOULD fail - 44.1kHz is not compliant
        assert info.samplerate != 16000, "Test fixture should have wrong sample rate"


# =============================================================================
# Manifest-Audio Linkage Tests
# =============================================================================

class TestManifestAudioLinkage:
    """
    Test suite for manifest-to-audio file linkage validation.
    
    This is an integration test that verifies the manifest correctly
    references existing audio files. Broken links cause training failures.
    """
    
    def test_manifest_audio_files_exist(self, mock_manifest: Path):
        """
        Test that all audio_filepath values point to existing files.
        
        Missing audio files are a common issue when:
        - Moving data between machines
        - Partial downloads
        - Path format differences (Windows vs Unix)
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                audio_path = Path(entry["audio_filepath"])
                
                assert audio_path.exists(), (
                    f"Line {line_number}: Audio file not found: {audio_path}"
                )
    
    def test_missing_audio_detected(self, mock_manifest_missing_audio: Path):
        """
        Test that missing audio files are correctly detected.
        
        This negative test verifies our validation catches broken links.
        """
        with mock_manifest_missing_audio.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                audio_path = Path(entry["audio_filepath"])
                
                # This file should NOT exist
                assert not audio_path.exists(), (
                    "Test fixture audio path should not exist"
                )
    
    def test_audio_files_are_readable_from_manifest(self, mock_manifest: Path):
        """
        Test that audio files referenced in manifest can be read and validated.
        
        End-to-end integration test: read manifest → load audio → verify format.
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                audio_path = entry["audio_filepath"]
                
                # Load and validate audio
                info = sf.info(audio_path)
                assert info.samplerate == 16000, f"Wrong sample rate in {audio_path}"
                assert info.channels == 1, f"Not mono in {audio_path}"
    
    def test_duration_matches_audio(self, mock_manifest: Path):
        """
        Test that manifest duration matches actual audio duration.
        
        Mismatched durations can cause:
        - Truncated audio during training
        - Index out of bounds errors
        - Incorrect batch padding
        """
        tolerance = 0.1  # 100ms tolerance for rounding differences
        
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                manifest_duration = entry["duration"]
                audio_path = entry["audio_filepath"]
                
                # Get actual audio duration
                info = sf.info(audio_path)
                actual_duration = info.duration
                
                assert abs(manifest_duration - actual_duration) <= tolerance, (
                    f"Line {line_number}: Duration mismatch. "
                    f"Manifest: {manifest_duration}s, Actual: {actual_duration}s"
                )


# =============================================================================
# Data Quality Tests
# =============================================================================

class TestDataQuality:
    """
    Additional data quality tests for production readiness.
    """
    
    def test_vietnamese_text_encoding(self, mock_manifest: Path):
        """
        Test that Vietnamese text is properly UTF-8 encoded.
        
        Encoding issues cause garbled text and degraded model performance.
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                text = entry["text"]
                
                # Verify the text can be encoded/decoded as UTF-8
                try:
                    encoded = text.encode("utf-8")
                    decoded = encoded.decode("utf-8")
                    assert decoded == text
                except UnicodeError as e:
                    pytest.fail(f"Line {line_number}: UTF-8 encoding error: {e}")
    
    def test_audio_filepath_is_absolute(self, mock_manifest: Path):
        """
        Test that audio paths are absolute, not relative.
        
        Relative paths break when:
        - Running from different directories
        - Using manifest on different machines
        - Docker/container environments
        """
        with mock_manifest.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                entry = json.loads(line)
                audio_path = Path(entry["audio_filepath"])
                
                assert audio_path.is_absolute(), (
                    f"Line {line_number}: audio_filepath should be absolute: {audio_path}"
                )
