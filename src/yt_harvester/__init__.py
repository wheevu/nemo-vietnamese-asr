"""YouTube Harvester - Extract transcripts and comments from YouTube videos."""

__version__ = "0.1.0"

from .downloader import (
    fetch_metadata,
    fetch_transcript,
    fetch_comments,
    download_audio,
    AUDIO_OUTPUT_DIR,
)

__all__ = [
    "fetch_metadata",
    "fetch_transcript", 
    "fetch_comments",
    "download_audio",
    "AUDIO_OUTPUT_DIR",
]
