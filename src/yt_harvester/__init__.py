"""YouTube Harvester - Extract transcripts and comments from YouTube videos.

Keep this module lightweight: importing `yt_harvester` should not require
optional runtime dependencies (e.g. transcript fetchers) unless those features
are actually used.
"""

from __future__ import annotations

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "fetch_metadata",
    "fetch_transcript",
    "fetch_comments",
    "download_audio",
    "AUDIO_OUTPUT_DIR",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from . import downloader as _downloader

        return getattr(_downloader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
