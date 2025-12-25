# tests/test_text_processing.py
"""
Unit tests for text processing utilities in the Vietnamese ASR pipeline.

This module tests the core text and URL processing functions that are critical
for data quality in ASR training. Poor text normalization leads to "training
pollution" - noisy labels that degrade model performance.

Test Categories:
1. YouTube URL Parsing - Validates video ID extraction from various URL formats
2. Vietnamese Text Normalization - Ensures caption cleaning preserves linguistic integrity

These tests demonstrate the "Linguist-Engineer" approach: treating text data
with the same rigor as code.
"""

from pathlib import Path

import pytest

from src.yt_harvester.utils import (
    video_id_from_url,
    clean_caption_lines,
    merge_fragments,
    build_watch_url,
)


# =============================================================================
# YouTube URL Parsing Tests
# =============================================================================

class TestVideoIdFromUrl:
    """
    Test suite for YouTube video ID extraction.
    
    The video_id_from_url function must handle the many URL formats that
    YouTube uses, including:
    - Standard watch URLs
    - Shortened youtu.be URLs
    - Embed URLs
    - Shorts URLs
    - URLs with timestamps and other query parameters
    - Raw 11-character video IDs
    """
    
    @pytest.mark.parametrize("url,expected_id", [
        # Standard watch URLs
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        
        # Shortened URLs (youtu.be)
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        
        # URLs with timestamps
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=120", "dQw4w9WgXcQ"),
        
        # URLs with additional query parameters
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLtest", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share", "dQw4w9WgXcQ"),
        
        # Embed URLs
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        
        # Shorts URLs
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        
        # Raw video ID (11 characters)
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        
        # IDs with underscores and hyphens (valid YouTube ID characters)
        ("https://www.youtube.com/watch?v=abc_123-XYZ", "abc_123-XYZ"),
        ("abc_123-XYZ", "abc_123-XYZ"),
    ])
    def test_valid_url_formats(self, url: str, expected_id: str):
        """
        Test that video IDs are correctly extracted from various valid URL formats.
        
        This parameterized test covers all common YouTube URL patterns that users
        might provide. Correct extraction is critical because the video ID is used
        to fetch transcripts and audio.
        """
        result = video_id_from_url(url)
        assert result == expected_id, f"Failed to extract ID from: {url}"
    
    @pytest.mark.parametrize("invalid_input", [
        # Completely invalid URLs
        "https://example.com/video",
        "https://vimeo.com/123456789",
        
        # Invalid video ID lengths
        "https://www.youtube.com/watch?v=short",  # Too short
        "https://www.youtube.com/watch?v=waytoolongvideoid",  # Too long
        
        # Empty or whitespace
        "",
        "   ",
        
        # URLs without video ID
        "https://www.youtube.com/",
        "https://www.youtube.com/channel/UCtest",
        
        # Malformed URLs
        "not a url at all",
        "youtube.com",  # Missing protocol and path
    ])
    def test_invalid_url_raises_error(self, invalid_input: str):
        """
        Test that invalid URLs raise ValueError with appropriate message.
        
        Graceful error handling prevents silent failures in the data pipeline.
        When a URL can't be parsed, we want a clear error rather than
        corrupted downstream data.
        """
        with pytest.raises(ValueError):
            video_id_from_url(invalid_input)
    
    def test_whitespace_handling(self):
        """
        Test that leading/trailing whitespace is stripped before parsing.
        
        Users often copy-paste URLs with extra whitespace.
        """
        url_with_whitespace = "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  "
        result = video_id_from_url(url_with_whitespace)
        assert result == "dQw4w9WgXcQ"


class TestBuildWatchUrl:
    """Test suite for constructing YouTube watch URLs from video IDs."""
    
    def test_build_watch_url(self):
        """Test that watch URLs are correctly constructed."""
        video_id = "dQw4w9WgXcQ"
        result = build_watch_url(video_id)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    def test_roundtrip_url_construction(self):
        """Test that we can extract an ID and rebuild the same canonical URL."""
        original_id = "abc_123-XYZ"
        url = build_watch_url(original_id)
        extracted_id = video_id_from_url(url)
        assert extracted_id == original_id


# =============================================================================
# Vietnamese Text Normalization Tests
# =============================================================================

class TestCleanCaptionLines:
    """
    Test suite for caption file cleaning and normalization.
    
    The clean_caption_lines function processes raw caption files (VTT/SRT)
    into clean text suitable for ASR training. This is critical for
    Vietnamese ASR because:
    - Diacritics must be preserved (they change meaning)
    - HTML entities must be decoded
    - Duplicate lines must be removed
    - Timing metadata must be stripped
    """
    
    def test_basic_whitespace_normalization(self, tmp_path: Path):
        """
        Test that excessive whitespace is normalized to single spaces.
        
        Uses tmp_path (pytest fixture) to create a temporary file with
        messy whitespace content, then verifies proper cleaning.
        """
        # Create a temporary file with messy content
        caption_file = tmp_path / "test_captions.txt"
        messy_content = "Xin chào    các bạn\n   Hôm nay   chúng ta   học\n"
        caption_file.write_text(messy_content, encoding="utf-8")
        
        # Clean the captions
        result = clean_caption_lines(caption_file)
        
        # Verify whitespace is collapsed
        assert "Xin chào các bạn" in result
        assert "Hôm nay chúng ta học" in result
        
        # Verify no entries have leading/trailing whitespace
        for line in result:
            assert line == line.strip(), f"Line has extra whitespace: '{line}'"
    
    def test_blank_lines_removed(self, tmp_path: Path):
        """
        Test that blank lines are excluded from output.
        
        Blank lines in captions are common but meaningless for training.
        """
        caption_file = tmp_path / "test_blanks.txt"
        content = "Dòng một\n\n\n\nDòng hai\n   \nDòng ba\n"
        caption_file.write_text(content, encoding="utf-8")
        
        result = clean_caption_lines(caption_file)
        
        # Should only have 3 meaningful lines
        assert len(result) == 3
        assert "" not in result
    
    def test_html_entity_decoding(self, tmp_path: Path):
        """
        Test that HTML entities are properly decoded.
        
        YouTube captions often contain HTML-encoded special characters
        like &amp; for & or &gt; for >.
        """
        caption_file = tmp_path / "test_html.txt"
        content = "Tom &amp; Jerry\n&lt;test&gt;\n"
        caption_file.write_text(content, encoding="utf-8")
        
        result = clean_caption_lines(caption_file)
        
        assert "Tom & Jerry" in result
        # Note: <test> might be stripped as HTML tag, which is correct behavior
    
    def test_duplicate_lines_removed(self, tmp_path: Path):
        """
        Test that consecutive duplicate lines are deduplicated.
        
        Caption files often repeat lines for timing purposes.
        """
        caption_file = tmp_path / "test_duplicates.txt"
        content = "Xin chào\nXin chào\nTạm biệt\nTạm biệt\nTạm biệt\n"
        caption_file.write_text(content, encoding="utf-8")
        
        result = clean_caption_lines(caption_file)
        
        # Should only have 2 unique lines
        assert result == ["Xin chào", "Tạm biệt"]
    
    def test_vietnamese_diacritics_preserved(self, tmp_path: Path):
        """
        Test that Vietnamese diacritics are preserved intact.
        
        This is CRITICAL for Vietnamese ASR. Vietnamese uses 12 vowels
        with up to 6 tone marks each. Losing diacritics changes meaning:
        - "ma" (ghost) vs "mà" (but) vs "má" (mother) vs "mả" (tomb)
        
        The normalization must clean formatting without destroying linguistic content.
        """
        caption_file = tmp_path / "test_vietnamese.txt"
        
        # Comprehensive Vietnamese text with all tone marks and special characters
        vietnamese_text = """
        Xin chào các bạn
        Tôi là người Việt Nam
        Đây là bài học tiếng Việt
        Cảm ơn rất nhiều
        Hẹn gặp lại
        ă â đ ê ô ơ ư
        à á ả ã ạ
        ằ ắ ẳ ẵ ặ
        ầ ấ ẩ ẫ ậ
        """
        caption_file.write_text(vietnamese_text, encoding="utf-8")
        
        result = clean_caption_lines(caption_file)
        result_text = " ".join(result)
        
        # Verify critical Vietnamese characters are preserved
        assert "Việt" in result_text, "Vietnamese word 'Việt' was corrupted"
        assert "ơ" in result_text, "Vietnamese vowel 'ơ' was corrupted"
        assert "ư" in result_text, "Vietnamese vowel 'ư' was corrupted"
        assert "đ" in result_text or "Đ" in result_text, "Vietnamese consonant 'đ' was corrupted"
    
    def test_vtt_metadata_stripped(self, mock_vtt_file: Path):
        """
        Test that WebVTT-specific metadata is removed.
        
        VTT files contain headers (WEBVTT), timing cues (00:00:00.000 --> 00:00:02.000),
        and metadata (Kind:, Language:) that should not appear in training text.
        """
        result = clean_caption_lines(mock_vtt_file)
        result_text = " ".join(result)
        
        # Metadata should be stripped
        assert "WEBVTT" not in result_text
        assert "-->" not in result_text
        assert "Kind:" not in result_text
        assert "Language:" not in result_text
        
        # Actual content should remain
        assert "Xin chào các bạn" in result_text
    
    def test_nonexistent_file_returns_empty(self, tmp_path: Path):
        """
        Test graceful handling of missing files.
        
        The function should return an empty list rather than raising
        an exception for missing files.
        """
        nonexistent = tmp_path / "does_not_exist.txt"
        result = clean_caption_lines(nonexistent)
        assert result == []
    
    def test_empty_file_returns_empty(self, tmp_path: Path):
        """
        Test handling of empty files.
        """
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        
        result = clean_caption_lines(empty_file)
        assert result == []


class TestMergeFragments:
    """
    Test suite for merging caption fragments into sentences.
    
    Caption files often split sentences across multiple lines for timing.
    This function reconstructs readable sentences.
    """
    
    def test_merge_simple_fragments(self):
        """Test basic fragment merging."""
        fragments = ["Xin chào", "các bạn."]
        result = merge_fragments(fragments)
        
        assert "Xin chào các bạn." in result
    
    def test_merge_preserves_sentence_boundaries(self):
        """Test that sentence endings are detected correctly."""
        fragments = [
            "Câu một.",
            "Câu hai!",
            "Câu ba?",
        ]
        result = merge_fragments(fragments)
        
        # Each should be a separate sentence
        assert len(result) == 3
    
    def test_merge_handles_empty_fragments(self):
        """Test that empty fragments are ignored."""
        fragments = ["Xin chào.", "", "  ", "Tạm biệt."]
        result = merge_fragments(fragments)
        
        assert len(result) == 2
        assert "Xin chào." in result
        assert "Tạm biệt." in result
    
    def test_merge_normalizes_whitespace(self):
        """Test that HTML entities and extra whitespace are handled."""
        fragments = ["  Xin &amp; chào  ", "  các bạn.  "]
        result = merge_fragments(fragments)
        
        # Should be normalized
        assert "Xin & chào các bạn." in result
